"""3-a. 시장성 평가 에이전트 (RAG: O) — 병렬 실행되는 노드 중 하나.

market_result 키에만 쓰기 때문에 stakeholder/domain 노드와 fan-out 시 충돌하지 않는다.
"""
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm import get_llm
from rag.ingest import build_vectorstore_from_files
from rag.retriever import retrieve_context
from agents.utils import load_prompt, parse_json_response

QUERY = "상용화 사례, 산업 채택, 시장 규모, 생태계 지원 프레임워크에 대한 언급"
MARKET_FIELDS = (
    ("시장 규모·성장성", "market_size_growth"),
    ("상용화·채택 현황", "commercialization_adoption"),
    ("생태계 지지", "ecosystem_support"),
    ("경제적 효과", "economic_effect"),
    ("도입 비용·장벽", "adoption_cost_barrier"),
)


def _format_market_result(result: dict) -> str:
    parts = [f"{label}: {result.get(key, '확인 불가')}" for label, key in MARKET_FIELDS]
    if result.get("summary"):
        parts.append(f"종합: {result['summary']}")
    return "\n".join(parts)


def _market_one(candidate: dict, transform: bool = False) -> tuple[str, str]:
    files = [candidate["file"], *candidate.get("market_files", [])]
    vs, _ = build_vectorstore_from_files(f"market_{candidate['id']}", candidate["id"], files)
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
    return _format_market_result(result), result.get("basis", "unknown")


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


def main() -> None:
    from candidates import FINAL_SELECTION, HW_POOL, SW_POOL

    selected_sw = next(c for c in SW_POOL if c["id"] == FINAL_SELECTION["sw"])
    selected_hw = next(c for c in HW_POOL if c["id"] == FINAL_SELECTION["hw"])
    result = market_node({"selected_sw": selected_sw, "selected_hw": selected_hw})

    print("[market] SW")
    print(result["market_result"]["sw"])
    print()
    print("[market] HW")
    print(result["market_result"]["hw"])
    print()
    print("[market] sources")
    for source in result["market_result"]["sources"]:
        print("-", source)


if __name__ == "__main__":
    main()
