"""3-a. 시장성 평가 에이전트 (RAG: O) — 병렬 실행되는 노드 중 하나.

market_result 키에만 쓰기 때문에 stakeholder/domain 노드와 fan-out 시 충돌하지 않는다.
"""
from llm import get_llm
from rag.ingest import build_vectorstore
from rag.retriever import retrieve_context
from agents.utils import load_prompt, parse_json_response

QUERY = "상용화 사례, 산업 채택, 시장 규모, 생태계 지원 프레임워크에 대한 언급"


def _market_one(candidate: dict, transform: bool = False) -> tuple[str, str]:
    vs, _ = build_vectorstore(candidate["id"], candidate["file"])
    context = (
        retrieve_context(
            vs, QUERY, k=7 if transform else 4, transform=transform, title=candidate["title"]
        )
        if vs
        else ""
    )
    prompt = load_prompt("market").format(title=candidate["title"], context=context or "(검색된 발췌 없음)")
    resp = get_llm(temperature=0.3).invoke(prompt)
    result = parse_json_response(resp.content)
    return result["summary"], result.get("basis", "unknown")


def market_node(state: dict) -> dict:
    # 재검색 라운드(retry_count>0)에서는 Query Transformation을 켜고 K를 넓힌다.
    transform = state.get("retry_count", 0) > 0
    sw_summary, sw_basis = _market_one(state["selected_sw"], transform)
    hw_summary, hw_basis = _market_one(state["selected_hw"], transform)
    return {
        "market_result": {
            "sw": sw_summary,
            "hw": hw_summary,
            "sources": [f"sw_basis={sw_basis}", f"hw_basis={hw_basis}"],
        }
    }
