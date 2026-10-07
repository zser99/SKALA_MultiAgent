"""Workers return isolated outcomes and external evidence artifacts."""
import json
from contextvars import ContextVar
from pathlib import Path

from agents.market import market_node
from agents.stakeholder import stakeholder_node
from agents.domain import domain_node
from agents.tech_research import tech_research_node

EVIDENCE = ContextVar("worker_evidence", default=None)
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
WORKER_NODES = {"market": market_node, "stakeholder": stakeholder_node, "domain": domain_node,
                "tech_research": tech_research_node}
WORKER_CAPABILITIES = {
    "market": {"perspective": "market", "description": "SW/HW 시장 규모, 상용화, 생태계, 경제 효과와 도입 장벽 평가"},
    "stakeholder": {"perspective": "stakeholder", "description": "SW/HW 이해관계자별 영향, 이익과 부담 비교"},
    "domain": {"perspective": "domain", "description": "SW/HW 도메인 적합성, 지표별 근거와 불확실성 평가"},
}


def record_evidence(documents):
    evidence = EVIDENCE.get()
    if evidence is not None:
        for document in documents:
            key = (document.metadata.get("source_file") or document.metadata.get("source"),
                   document.metadata.get("page"), document.page_content)
            evidence[key] = {"metadata": document.metadata, "text": document.page_content}


def worker_node(state: dict) -> dict:
    task = state["task"]
    worker = task["worker"]
    evidence = {}
    token = EVIDENCE.set(evidence)
    error = None
    try:
        payload = WORKER_NODES[worker]({**state, "worker_instruction": task["instruction"]})
        status = "completed"
    except Exception as exception:
        error = type(exception).__name__
        status = "failed"
        previous = state.get("worker_results", {}).get(task["task_id"], {}).get("payload")
        payload = previous or {f"{worker}_result": {
            "sw": "작업 실패로 판단 유보", "hw": "작업 실패로 판단 유보", "sources": []}}
    finally:
        EVIDENCE.reset(token)
    directory = OUTPUT_DIR / state["run_id"]
    directory.mkdir(parents=True, exist_ok=True)
    evidence_path = directory / f"{task['task_id']}_{task['attempt']}_evidence.json"
    evidence_path.write_text(json.dumps(list(evidence.values()), ensure_ascii=False, default=str), encoding="utf-8")
    if status == "failed" and state.get("worker_results", {}).get(task["task_id"]):
        evidence_path = Path(state["worker_results"][task["task_id"]]["evidence_path"])
    return {"worker_results": {task["task_id"]: {
        "worker": worker, "status": status, "attempt": task["attempt"], "payload": payload,
        "error": error, "evidence_path": str(evidence_path)}}}


def research_node(state: dict) -> dict:
    update = worker_node({**state, "task": {
        "task_id": "tech_research", "worker": "tech_research", "instruction": "", "attempt": 0}})
    result = update["worker_results"]["tech_research"]
    if result["status"] == "failed":
        raise RuntimeError("선행 기술 조사 실패: " + result["error"])
    return {**result["payload"], **update}
