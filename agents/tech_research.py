"""2. 기술 조사 에이전트 (RAG: O).

선정된 SW/HW 논문 각각에 대해 벡터스토어를 만들고, "개요/접근/한계/TRL"을 근거 기반으로
정리한다. PDF가 없으면 candidates.py의 summary로 폴백한다.

검색 전략 (단일 질의 Top-K에서 Related Work·References가 상위를 차지해 오귀속·누락이
생겼던 문제에 대한 대응):
- Pin      : Abstract / Conclusion 청크는 검색 점수와 무관하게 항상 컨텍스트에 포함한다.
- 필드별 multi-query : overview / approach / limitations / trl 마다 별도 질의 세트로 검색하고
             필드 안에서는 RRF로 융합, 필드 사이에서는 chunk_id로 중복을 제거한다.
             질의는 기술명({tech})만 바꿔 끼우는 범용 템플릿이라 어떤 논문에도 그대로 쓴다.
- 재검색   : LLM이 "발췌에 없어 미확인"으로 남긴 항목(unverified)이 있으면 그 키워드로
             1회 추가 검색한 뒤 다시 판정한다. unverified는 노드 내부에서만 쓰고 결과에서 제거해
             하위 노드가 받는 스키마(overview/approach/limitations/trl/sources)는 그대로 둔다.
"""
from collections import Counter

from langchain_core.documents import Document

from llm import get_llm
from rag.ingest import build_vectorstore
from rag.retriever import search
from agents.utils import load_prompt, parse_json_response

import sys

# {tech}에는 논문 제목의 콜론 앞 약칭(KIVI, ITME ...)이 들어간다. 특정 논문의 어휘는 넣지 않고
# "기술 논문이면 대개 갖고 있는 절"(방법·구현·실험·한계·future work·코드 공개)을 겨냥한다.
# 이 프로젝트의 대상이 모두 LLM 추론 KV cache 기술이므로 KV cache/prefill/decoding 같은
# 도메인 공통 어휘는 사용한다 (SW/HW 논문 양쪽에서 방법 절을 잘 끌어오는 것을 확인함).
FIELD_QUERY_TEMPLATES = {
    "overview": [
        "{tech} main contribution and key results summary",
        "{tech} problem motivation and proposed solution",
    ],
    "approach": [
        "how the proposed method works step by step",
        "{tech} how the KV cache is stored, updated and accessed during inference",
        "{tech} handling of newly generated tokens during decoding",
        "{tech} analysis of why the proposed design works",
    ],
    "limitations": [
        "{tech} limitation accuracy degradation failure case",
        "{tech} future work remaining overhead and limitation",
        "{tech} ablation study sensitivity to hyperparameters",
    ],
    "trl": [
        "{tech} source code implementation publicly available",
        "{tech} real hardware benchmark throughput latency memory measurement",
        "{tech} implementation with custom GPU kernels or hardware prototype",
        "{tech} experimental setup hardware platform GPU used for evaluation",
    ],
}
PIN_SECTION_KINDS = ("abstract", "conclusion")
K_PER_QUERY = 5        # 질의 1개당 Top-K
K_PER_FIELD = 5        # 필드 안에서 RRF 융합 후 남기는 청크 수
K_WIDEN = 2            # 재검색 라운드·evidence 재시도에서 K를 넓히는 폭
MAX_REFINE_ROUNDS = 1  # "미확인" 항목 재검색 횟수
RRF_K = 60


def _tech_short_name(title: str) -> str:
    """'KIVI: A Tuning-Free ...' -> 'KIVI'. 콜론이 없으면 제목 전체를 쓴다."""
    return title.split(":")[0].strip() if ":" in title else title.strip()


def _chunk_key(d: Document) -> str:
    return d.metadata.get("chunk_id") or f"{d.metadata.get('page')}:{d.page_content[:80]}"


def _rrf_fuse(ranked_lists: list[list[Document]], k: int) -> list[Document]:
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for ranked in ranked_lists:
        for rank, d in enumerate(ranked):
            key = _chunk_key(d)
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            docs.setdefault(key, d)
    return [docs[key] for key in sorted(scores, key=lambda key: scores[key], reverse=True)[:k]]


def _pinned_docs(vs) -> list[Document]:
    """Abstract / Conclusion 청크를 docstore에서 직접 꺼낸다 (검색 점수와 무관하게 포함)."""
    docs = [d for d in vs.docstore._dict.values()
            if d.metadata.get("section_kind") in PIN_SECTION_KINDS]
    return sorted(docs, key=_chunk_key)


def _retrieve_fielded(vs, tech: str, k_per_query: int, extra_queries: list[str] = ()
                      ) -> tuple[list[Document], dict[str, list[Document]]]:
    """(핀 고정 청크, {필드: 검색 청크}) — 필드 간 중복은 먼저 나온 필드에만 남긴다."""
    pinned = _pinned_docs(vs)
    seen = {_chunk_key(d) for d in pinned}
    by_field: dict[str, list[Document]] = {}

    def pick(field: str, queries: list[str], k: int):
        fused = _rrf_fuse([search(vs, q, k=k_per_query) for q in queries], k)
        picked = []
        for d in fused:
            if _chunk_key(d) not in seen:
                seen.add(_chunk_key(d))
                picked.append(d)
        by_field[field] = picked

    for field, templates in FIELD_QUERY_TEMPLATES.items():
        pick(field, [t.format(tech=tech) for t in templates], K_PER_FIELD)
    if extra_queries:
        pick("재검색(미확인 항목)", list(extra_queries), K_PER_FIELD + K_WIDEN)
    return pinned, by_field


def _tag(d: Document) -> str:
    m = d.metadata
    tag = f"p.{m.get('page', '?')} | §{m.get('section', '?')}"
    if m.get("block_type") in ("table", "algorithm"):
        tag += f" | {m['block_type'].upper()}"
        if m.get("position_section"):
            tag += f" (본문 참조 기준 귀속, 실제 위치 §{m['position_section']})"
    return f"[{tag}]"


def _body(d: Document) -> str:
    # ingest 단계에서 임베딩용으로 붙인 "[§...]" 첫 줄은 태그와 중복이므로 뗀다
    text = d.page_content.strip()
    if text.startswith("[§") and "\n" in text:
        text = text.split("\n", 1)[1]
    return text.strip()


def _format_context(pinned: list[Document], by_field: dict[str, list[Document]]) -> str:
    parts = ["### 항상 포함되는 발췌 (Abstract / Conclusion)"]
    parts += [f"{_tag(d)}\n{_body(d)}" for d in pinned]
    for field, docs in by_field.items():
        if docs:
            parts.append(f"### '{field}' 질의로 검색된 발췌")
            parts += [f"{_tag(d)}\n{_body(d)}" for d in docs]
    return "\n\n".join(parts)


def _log_retrieval(candidate_id: str, round_no: int, pinned: list[Document],
                   by_field: dict[str, list[Document]]) -> None:
    """검색된 청크의 section 분포를 남긴다 (Abstract 포함 여부 등 검색 품질 확인용)."""
    all_docs = pinned + [d for docs in by_field.values() for d in docs]
    dist = Counter(d.metadata.get("section", "?") for d in sorted(all_docs, key=_chunk_key))
    print(f"[tech_research][{candidate_id}] round={round_no} chunks={len(all_docs)} "
          f"pinned={[d.metadata.get('section') for d in pinned]}")
    for section, n in dist.items():
        print(f"    {n:>2}  §{section}")


def _as_keywords(value) -> list[str]:
    if isinstance(value, str):
        value = [v for v in value.replace(";", ",").split(",")]
    if not isinstance(value, list):
        return []
    return [v.strip() for v in value if isinstance(v, str) and v.strip()]


def _research_one(candidate: dict, warnings: list, widen: bool = False) -> dict:
    vs, warning = build_vectorstore(candidate["id"], candidate["file"])
    if warning:
        warnings.append(warning)

    if vs is None:
        # PDF 없음 -> 요약 메타데이터로 폴백 (RAG 없이)
        return {
            "overview": candidate["summary"],
            "approach": "(원문 미확보로 Doc Pool 요약을 사용함)",
            "limitations": "원문 미확보로 한계점 분석 불가",
            "trl": "원문 미확보로 TRL 근거 확인 불가",
            "sources": [candidate["url"]],
        }

    tech = _tech_short_name(candidate["title"])
    k_per_query = K_PER_QUERY + (K_WIDEN if widen else 0)
    extra_queries: list[str] = []
    refine_note = ""
    result: dict = {}

    for round_no in range(MAX_REFINE_ROUNDS + 1):
        pinned, by_field = _retrieve_fielded(vs, tech, k_per_query, extra_queries)
        _log_retrieval(candidate["id"], round_no, pinned, by_field)

        prompt = load_prompt("tech_research").format(
            title=candidate["title"], year=candidate["year"], tech=tech,
            context=_format_context(pinned, by_field), refine_note=refine_note,
        )
        resp = get_llm(temperature=0.2).invoke(prompt)
        result = parse_json_response(resp.content)

        unverified = _as_keywords(result.pop("unverified", []))
        if not unverified or round_no >= MAX_REFINE_ROUNDS:
            break
        # 미확인 항목 키워드로 1회 재검색 후 재판정
        print(f"[tech_research][{candidate['id']}] 미확인 {len(unverified)}건 -> 재검색: {unverified}")
        warnings.append(f"[tech_research] {candidate['id']}: 미확인 항목 재검색 1회 ({', '.join(unverified)})")
        extra_queries = [f"{tech} {kw}" for kw in unverified]
        k_per_query += K_WIDEN
        refine_note = (
            "\n[재판정 안내] 이전 판정에서 다음 항목이 발췌에 없어 '미확인'이었고, 그 키워드로 추가 검색한 "
            f"발췌('재검색(미확인 항목)' 그룹)를 덧붙였습니다: {', '.join(unverified)}. "
            "추가 발췌에서 확인되면 '확인됨(§x)'으로 반영하고, 여전히 없으면 '미확인'으로 유지하세요.\n"
        )

    # 하위 노드(종합/보고서)가 받는 스키마를 고정한다. LLM이 덧붙인 키는 버린다.
    return {
        **{key: result.get(key, "원문 발췌에서 확인 불가") for key in ("overview", "approach", "limitations", "trl")},
        "sources": [candidate["url"]],
    }


def tech_research_node(state: dict) -> dict:
    # evidence_check의 재검색 라운드(retry_count>0)에서는 K를 넓혀 다시 조사한다.
    widen = state.get("retry_count", 0) > 0
    warnings = list(state.get("warnings", []))
    sw = _research_one(state["selected_sw"], warnings, widen)
    hw = _research_one(state["selected_hw"], warnings, widen)
    return {"tech_research": {"sw": sw, "hw": hw}, "warnings": warnings}
