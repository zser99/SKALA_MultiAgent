"""Orchestrator-Workers with dynamic Send and bounded quality rework."""
from langgraph.graph import StateGraph, START, END
from state import GraphState
from agents.tech_selection import tech_selection_node
from agents.orchestrator import (
    orchestrator_node, dispatch_workers, collect_node, quality_router,
    rework_node, finish_node,
)
from agents.worker import worker_node, research_node
from agents.synthesis import synthesis_node
from agents.report import report_node
from agents.quality import quality_node


def generate_report_node(state: dict) -> dict:
    try:
        return {**report_node(state), "generation_error": None}
    except Exception as exception:
        return {"final_report": state.get("final_report", ""),
                "generation_error": type(exception).__name__}


def build_graph(checkpointer=None):
    graph = StateGraph(GraphState)
    nodes = {
        "tech_selection": tech_selection_node, "tech_research": research_node,
        "orchestrator": orchestrator_node, "worker": worker_node, "collect": collect_node,
        "synthesis": synthesis_node, "report": generate_report_node, "quality": quality_node,
        "rework": rework_node, "finish": finish_node,
    }
    for name, node in nodes.items():
        graph.add_node(name, node)
    graph.add_edge(START, "tech_selection")
    graph.add_edge("tech_selection", "tech_research")
    graph.add_edge("tech_research", "orchestrator")
    graph.add_conditional_edges("orchestrator", dispatch_workers, ["worker", "collect"])
    graph.add_edge("worker", "collect")
    graph.add_edge("collect", "synthesis")
    graph.add_edge("synthesis", "report")
    graph.add_edge("report", "quality")
    graph.add_conditional_edges("quality", quality_router, {"finish": "finish", "rework": "rework"})
    graph.add_edge("rework", "orchestrator")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer)
