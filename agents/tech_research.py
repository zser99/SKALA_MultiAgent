"""2. 기술 조사 에이전트 (RAG: O).

선정된 SW/HW 논문 각각에 대해 벡터스토어를 만들고, "개요/접근/한계"를 질의해
근거 기반으로 정리한다. PDF가 없으면 candidates.py의 summary로 폴백한다.
"""
from llm import get_llm
from rag.ingest import build_vectorstore
from rag.retriever import retrieve_context
from agents.utils import load_prompt, parse_json_response

QUERY = "이 기술의 핵심 접근 방식, 적용 범위, 그리고 한계점 또는 트레이드오프"


def _research_one(candidate: dict, warnings: list, transform: bool = False) -> dict:
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

    context = retrieve_context(
        vs, QUERY, k=8 if transform else 5, transform=transform, title=candidate["title"]
    )
    prompt = load_prompt("tech_research").format(
        title=candidate["title"], year=candidate["year"], context=context
    )
    resp = get_llm(temperature=0.2).invoke(prompt)
    result = parse_json_response(resp.content)
    result["sources"] = [candidate["url"]]
    return result


def tech_research_node(state: dict) -> dict:
    # 재검색 라운드(retry_count>0)에서는 Query Transformation을 켜고 K를 넓힌다.
    transform = state.get("retry_count", 0) > 0
    warnings = list(state.get("warnings", []))
    sw = _research_one(state["selected_sw"], warnings, transform)
    hw = _research_one(state["selected_hw"], warnings, transform)
    return {"tech_research": {"sw": sw, "hw": hw}, "warnings": warnings}
