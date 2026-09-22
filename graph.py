"""그래프 구조 (설계문서 D안 그대로 구현):

기술선정 -> 기술조사 -> (시장성평가, 이해관계자평가, 도메인평가) [병렬]
        -> 근거 충분성 Check -> (부족하면 재검색 1회) -> 평가종합 -> 보고서생성

market_result / stakeholder_result / domain_result 가 서로 다른 State 키에 쓰기 때문에
LangGraph의 기본 병렬 실행(superstep)만으로 fan-out/fan-in이 안전하게 동작한다.
"""
from langgraph.graph import StateGraph, START, END

from state import GraphState
from agents.tech_selection import tech_selection_node
from agents.tech_research import tech_research_node
from agents.market import market_node
from agents.stakeholder import stakeholder_node
from agents.domain import domain_node
from agents.evidence import evidence_check_node, evidence_router, re_retrieval_node
from agents.synthesis import synthesis_node
from agents.report import report_node


def build_graph():
    g = StateGraph(GraphState)

    g.add_node("tech_selection", tech_selection_node)
    g.add_node("tech_research", tech_research_node)
    g.add_node("market", market_node)
    g.add_node("stakeholder", stakeholder_node)
    g.add_node("domain", domain_node)
    g.add_node("evidence_check", evidence_check_node)
    g.add_node("re_retrieval", re_retrieval_node)
    g.add_node("synthesis", synthesis_node)
    g.add_node("report", report_node)

    g.add_edge(START, "tech_selection")
    g.add_edge("tech_selection", "tech_research")

    # fan-out
    g.add_edge("tech_research", "market")
    g.add_edge("tech_research", "stakeholder")
    g.add_edge("tech_research", "domain")

    # fan-in -> 근거 충분성 검증
    g.add_edge("market", "evidence_check")
    g.add_edge("stakeholder", "evidence_check")
    g.add_edge("domain", "evidence_check")

    # 부족하면 재검색(최대 1회) 후 다시 검증, 충분하면 종합으로 진행
    g.add_conditional_edges(
        "evidence_check",
        evidence_router,
        {"re_retrieval": "re_retrieval", "synthesis": "synthesis"},
    )
    g.add_edge("re_retrieval", "evidence_check")

    g.add_edge("synthesis", "report")
    g.add_edge("report", END)

    return g.compile()
