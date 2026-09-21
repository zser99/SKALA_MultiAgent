"""4. 평가 종합 에이전트 (RAG: X).

market_result, stakeholder_result, domain_result 세 병렬 노드가 모두 끝난 뒤에만
실행된다 (LangGraph가 fan-in을 자동으로 처리).
"""
from llm import get_llm
from agents.utils import load_prompt, parse_json_response


def synthesis_node(state: dict) -> dict:
    tr = state["tech_research"]
    prompt = load_prompt("synthesis").format(
        sw_title=state["selected_sw"]["title"],
        hw_title=state["selected_hw"]["title"],
        trl_sw=tr["sw"].get("trl", "N/A"),
        trl_hw=tr["hw"].get("trl", "N/A"),
        market_sw=state["market_result"]["sw"],
        market_hw=state["market_result"]["hw"],
        stakeholder_sw=state["stakeholder_result"]["sw"],
        stakeholder_hw=state["stakeholder_result"]["hw"],
        domain_focus=state.get("domain_focus", ""),
        domain_sw=state["domain_result"]["sw"],
        domain_hw=state["domain_result"]["hw"],
    )
    resp = get_llm(temperature=0.2).invoke(prompt)
    result = parse_json_response(resp.content)
    return {"synthesis": result}
