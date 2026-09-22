"""3-c. 도메인 적용 평가 에이전트 (RAG: O) — 병렬 실행되는 노드 중 하나.

팀 최종 결정 도메인: "Cloud/Data Center 환경의 Long-context LLM Serving".
Memory Efficiency / Latency / Throughput / Quality / Infrastructure Requirement
5개 세부 지표로 평가한다 (설계문서 '④ Domain 적용성 세부 평가 기준' 표 기반).
"""
from llm import get_llm
from rag.ingest import build_vectorstore
from rag.retriever import retrieve_context
from agents.utils import load_prompt, parse_json_response

DEFAULT_DOMAIN = "Cloud/Data Center 환경의 Long-context LLM Serving"
QUERY = (
    "메모리 사용량/효율, 지연시간(latency), 처리량(throughput), 출력 품질(정확도/perplexity), "
    "기존 인프라 대비 추가로 필요한 하드웨어/소프트웨어 변경 사항"
)


def _domain_one(candidate: dict, domain_focus: str, transform: bool = False) -> str:
    vs, _ = build_vectorstore(candidate["id"], candidate["file"])
    context = (
        retrieve_context(
            vs, QUERY, k=8 if transform else 5, transform=transform, title=candidate["title"]
        )
        if vs
        else ""
    )
    prompt = load_prompt("domain").format(
        title=candidate["title"],
        domain_focus=domain_focus,
        context=context or "(검색된 발췌 없음)",
    )
    resp = get_llm(temperature=0.3).invoke(prompt)
    result = parse_json_response(resp.content)
    return result["summary"]


def domain_node(state: dict) -> dict:
    # 재검색 라운드(retry_count>0)에서는 Query Transformation을 켜고 K를 넓힌다.
    transform = state.get("retry_count", 0) > 0
    domain_focus = state.get("domain_focus") or DEFAULT_DOMAIN
    sw_summary = _domain_one(state["selected_sw"], domain_focus, transform)
    hw_summary = _domain_one(state["selected_hw"], domain_focus, transform)
    return {
        "domain_focus": domain_focus,
        "domain_result": {"sw": sw_summary, "hw": hw_summary},
    }
