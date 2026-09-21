"""5. 보고서 생성 에이전트 (RAG: X). 최종 마크다운 보고서를 만든다."""
from llm import get_llm
from agents.utils import load_prompt


def _format_references(state: dict) -> str:
    lines = []
    for key in ("selected_sw", "selected_hw"):
        c = state[key]
        lines.append(f"- 논문: {c['title']} ({c['year']}). *arXiv*. {c['url']}")
    return "\n".join(lines)


def report_node(state: dict) -> dict:
    tr = state["tech_research"]
    syn = state["synthesis"]
    prompt = load_prompt("report").format(
        selection_rationale=state["selection_rationale"],
        sw_research=tr["sw"],
        hw_research=tr["hw"],
        trl_sw=tr["sw"].get("trl", "N/A"),
        trl_hw=tr["hw"].get("trl", "N/A"),
        market_sw=state["market_result"]["sw"],
        market_hw=state["market_result"]["hw"],
        stakeholder_sw=state["stakeholder_result"]["sw"],
        stakeholder_hw=state["stakeholder_result"]["hw"],
        domain_focus=state.get("domain_focus", ""),
        domain_sw=state["domain_result"]["sw"],
        domain_hw=state["domain_result"]["hw"],
        agreements=syn["agreements"],
        conflicts=syn["conflicts"],
        implications=syn["implications"],
        references=_format_references(state),
    )
    resp = get_llm(temperature=0.2).invoke(prompt)
    return {"final_report": resp.content.strip()}
