"""실행 스크립트.

Usage:
    python app.py
    python app.py --domain "OnDevice AI"
"""
import argparse
import datetime
import os
import json
from pathlib import Path
from uuid import uuid4
from langgraph.checkpoint.memory import InMemorySaver

from graph import build_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain",
        default=None,
        help="평가 대상 도메인 (예: '데이터센터/클라우드 서빙', 'OnDevice AI', '장문맥 처리 어플리케이션')",
    )
    args = parser.parse_args()

    app = build_graph(checkpointer=InMemorySaver())

    run_id = uuid4().hex
    initial_state = {"run_id": run_id, "round_count": 0, "step_count": 0,
                     "max_rounds": 2, "max_steps": 12, "worker_results": {}}
    if args.domain:
        initial_state["domain"] = args.domain

    print("[graph] 실행 시작...")
    directory = Path(__file__).resolve().parent / "outputs" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    config = {"recursion_limit": 60, "configurable": {"thread_id": run_id},
              "run_name": "orchestrator-workers", "metadata": {"run_id": run_id}}
    with (directory / "trace.jsonl").open("w", encoding="utf-8") as trace:
        for event in app.stream(initial_state, config=config, stream_mode="updates"):
            for node, update in event.items():
                record = {"run_id": run_id, "node": node,
                          "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          "plan": update.get("plan"), "eval_result": update.get("eval_result"),
                          "workers": {name: {"status": result["status"], "attempt": result["attempt"]}
                                      for name, result in update.get("worker_results", {}).items()},
                          "termination_reason": update.get("termination_reason")}
                trace.write(json.dumps(record, ensure_ascii=False) + "\n")
                trace.flush()
                print(f"[graph] {node}", flush=True)
    final_state = app.get_state(config).values
    (directory / "quality.json").write_text(
        json.dumps(final_state["eval_result"], ensure_ascii=False, indent=2), encoding="utf-8")

    if final_state.get("warnings"):
        print("\n[warnings]")
        for w in final_state["warnings"]:
            print(" -", w)

    os.makedirs("outputs", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join("outputs", f"report_{ts}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(final_state["final_report"])

    print(f"\n[done] 보고서 저장: {out_path}")
    print(f" - 선정 SW: {final_state['selected_sw']['title']}")
    print(f" - 선정 HW: {final_state['selected_hw']['title']}")


if __name__ == "__main__":
    main()
