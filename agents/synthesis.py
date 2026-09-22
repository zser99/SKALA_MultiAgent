"""4. 평가 종합 에이전트 (RAG: X).

market_result, stakeholder_result, domain_result 세 병렬 노드가 모두 끝난 뒤에만
실행된다 (LangGraph가 fan-in을 자동으로 처리).
"""
from llm import get_llm
from agents.utils import load_prompt, parse_json_response


EXPECTED_OUTPUT_KEYS = ("agreements", "conflicts", "implications")


def synthesis_node(state: dict) -> dict:
    tr = state["tech_research"]
    stakeholder = state["stakeholder_result"]
    prompt = load_prompt("synthesis").format(
        sw_title=state["selected_sw"]["title"],
        hw_title=state["selected_hw"]["title"],
        trl_sw=tr["sw"].get("trl", "N/A"),
        trl_hw=tr["hw"].get("trl", "N/A"),
        market_sw=state["market_result"]["sw"],
        market_hw=state["market_result"]["hw"],
        stakeholder_sw=stakeholder["sw"],
        stakeholder_hw=stakeholder["hw"],
        stakeholder_summary=stakeholder.get("summary", "제공되지 않음"),
        # app.py의 domain_focus와 설계 State의 domain을 모두 허용한다.
        domain_focus=state.get("domain_focus") or state.get("domain") or "미지정",
        domain_sw=state["domain_result"]["sw"],
        domain_hw=state["domain_result"]["hw"],
        evidence_sufficient=state.get("evidence_sufficient", False),
        evidence_warnings="\n".join(state.get("warnings", [])) or "없음",
    )
    resp = get_llm(temperature=0.2).invoke(prompt)
    result = parse_json_response(resp.content)

    missing = [key for key in EXPECTED_OUTPUT_KEYS if key not in result]
    if missing:
        raise ValueError(f"평가 종합 결과에 필수 항목이 없습니다: {', '.join(missing)}")

    return {"synthesis": result}
