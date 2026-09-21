"""그래프 구조 (설계문서 D안 그대로 구현):

기술선정 -> 기술조사 -> (시장성평가, 이해관계자평가, 도메인평가) [병렬] -> 평가종합 -> 보고서생성

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
from agents.synthesis import synthesis_node
from agents.report import report_node


def build_graph():
    g = StateGraph(GraphState)

    g.add_node("tech_selection", tech_selection_node)
    g.add_node("tech_research", tech_research_node)
    g.add_node("market", market_node)
    g.add_node("stakeholder", stakeholder_node)
    g.add_node("domain", domain_node)
    g.add_node("synthesis", synthesis_node)
    g.add_node("report", report_node)

    g.add_edge(START, "tech_selection")
    g.add_edge("tech_selection", "tech_research")

    # fan-out
    g.add_edge("tech_research", "market")
    g.add_edge("tech_research", "stakeholder")
    g.add_edge("tech_research", "domain")

    # fan-in
    g.add_edge("market", "synthesis")
    g.add_edge("stakeholder", "synthesis")
    g.add_edge("domain", "synthesis")

    g.add_edge("synthesis", "report")
    g.add_edge("report", END)

    return g.compile()
