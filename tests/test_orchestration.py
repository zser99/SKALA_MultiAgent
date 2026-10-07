import unittest
from unittest.mock import patch
from copy import deepcopy

from state import merge_worker_results
from agents.orchestrator import (
    orchestrator_node, dispatch_workers, quality_router, WorkerPlan, validate_plan,
)
from tests.mock_state import MOCK_STATE
from graph import build_graph, generate_report_node


class OrchestrationTest(unittest.TestCase):
    def make_plan(self, workers):
        return WorkerPlan(tasks=[{"task_id": name, "worker": name,
                                 "instruction": "근거를 조사해 평가", "reason": "필수 관점 또는 근거 보완"}
                                for name in workers])

    @patch("graph.report_node", side_effect=RuntimeError("unavailable"))
    def test_generation_failure_preserves_draft_for_quality_loop(self, report):
        result = generate_report_node({"final_report": "previous draft"})
        self.assertEqual(result["final_report"], "previous draft")
        self.assertEqual(result["generation_error"], "RuntimeError")

    @patch("agents.orchestrator.get_llm")
    def test_plan_dispatches_three_then_one_then_no_workers(self, llm):
        llm.return_value.with_structured_output.return_value.invoke.side_effect = [
            self.make_plan(["domain", "market", "stakeholder"]),
            self.make_plan(["market"]), self.make_plan([])]
        state = {"worker_results": {}, "round_count": 0}
        state.update(orchestrator_node(state))
        self.assertEqual(len(dispatch_workers(state)), 3)
        state["worker_results"] = {name: {"status": "completed"} for name in ("market", "stakeholder", "domain")}
        state["eval_result"] = {"passed": False, "rework_type": "evidence",
                                "rework_targets": ["market"], "reasons": ["근거 보완"]}
        state.update(orchestrator_node(state))
        self.assertEqual([task.arg["task"]["worker"] for task in dispatch_workers(state)], ["market"])
        state["eval_result"]["rework_targets"] = []
        state.update(orchestrator_node(state))
        self.assertEqual(dispatch_workers(state), "collect")

    @patch("agents.orchestrator.get_llm")
    def test_invalid_plan_is_repaired(self, llm):
        llm.return_value.with_structured_output.return_value.invoke.side_effect = [
            self.make_plan(["unknown"]), self.make_plan(["market"])]
        result = orchestrator_node({"required_perspectives": ["market"]})
        self.assertEqual([task["worker"] for task in result["plan"]], ["market"])

    @patch("agents.orchestrator.get_llm")
    def test_invalid_plan_stops_without_fixed_fallback(self, llm):
        llm.return_value.with_structured_output.return_value.invoke.return_value = self.make_plan([])
        with self.assertRaises(RuntimeError):
            orchestrator_node({})
        self.assertEqual(llm.return_value.with_structured_output.return_value.invoke.call_count, 2)

    def test_required_coverage_and_duplicate_workers(self):
        with self.assertRaises(ValueError):
            validate_plan(self.make_plan(["market"]), {})
        with self.assertRaises(ValueError):
            validate_plan(self.make_plan(["market", "market"]), {})

    def test_rework_plan_must_match_quality_targets(self):
        state = {
            "eval_result": {"passed": False, "rework_type": "evidence",
                            "rework_targets": ["domain"], "reasons": []},
            "required_perspectives": ["market", "stakeholder", "domain"],
            "worker_results": {
                name: {"worker": name, "status": "completed"}
                for name in ("market", "stakeholder", "domain")
            },
        }
        with self.assertRaises(ValueError):
            validate_plan(self.make_plan(["market"]), state)
        validate_plan(self.make_plan(["domain"]), state)

    def test_quality_router_separates_report_and_worker_rework(self):
        base = {"round_count": 0, "max_rounds": 2, "step_count": 1, "max_steps": 12}
        self.assertEqual(quality_router({**base, "eval_result": {
            "passed": False, "rework_type": "report", "rework_targets": [], "reasons": []
        }}), "rewrite_report")
        self.assertEqual(quality_router({**base, "eval_result": {
            "passed": False, "rework_type": "evidence",
            "rework_targets": ["market"], "reasons": []
        }}), "rework_workers")

    def test_rework_preserves_task_identity(self):
        with self.assertRaises(ValueError):
            validate_plan(self.make_plan(["market"]), {
                "required_perspectives": ["market"],
                "worker_results": {"market-review": {"worker": "market", "status": "failed"}}})

    def test_reducer_replaces_only_reworked_result(self):
        self.assertEqual(merge_worker_results({"market": 1, "domain": 2}, {"market": 3}),
                         {"market": 3, "domain": 2})

    def run_mock_graph(self, always_fail=False):
        calls = []
        evaluations = []

        def worker(state):
            task = state["task"]
            calls.append((task["worker"], task["attempt"]))
            name = task["worker"]
            return {"worker_results": {name: {
                "status": "completed", "attempt": task["attempt"], "error": None,
                "evidence_path": "unused", "payload": {f"{name}_result": MOCK_STATE[f"{name}_result"]}}}}

        def quality(state):
            evaluations.append(state["round_count"])
            passed = not always_fail and state["round_count"] > 0
            return {"eval_result": {"passed": passed, "checks": {"groundedness": passed},
                                    "reasons": [] if passed else ["시장성 보완"],
                                    "rework_type": "none" if passed else "evidence",
                                    "rework_targets": [] if passed else ["market"]},
                    "step_count": state["step_count"] + 1}

        with (patch("graph.tech_selection_node", return_value={}),
              patch("graph.research_node", return_value={}),
              patch("graph.worker_node", side_effect=worker),
              patch("graph.synthesis_node", return_value={"synthesis": MOCK_STATE["synthesis"]}),
              patch("graph.report_node", return_value={"final_report": "mock report"}),
              patch("agents.orchestrator.get_llm") as llm,
              patch("graph.quality_node", side_effect=quality)):
            llm.return_value.with_structured_output.return_value.invoke.side_effect = [
                self.make_plan(["market", "stakeholder", "domain"]),
                self.make_plan(["market"]), self.make_plan(["market"])]
            result = build_graph().invoke({**deepcopy(MOCK_STATE), "run_id": "test",
                                          "worker_results": {}, "round_count": 0,
                                          "max_rounds": 2, "step_count": 0},
                                         {"recursion_limit": 60})
        return result, calls, evaluations

    def test_graph_selective_rework_and_success(self):
        result, calls, evaluations = self.run_mock_graph()
        self.assertEqual(result["termination_reason"], "quality_passed")
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[-1], ("market", 1))
        self.assertEqual(evaluations, [0, 1])

    def test_graph_terminates_when_quality_never_passes(self):
        result, calls, evaluations = self.run_mock_graph(always_fail=True)
        self.assertEqual(result["termination_reason"], "rework_limit_reached")
        self.assertEqual(evaluations, [0, 1, 2])
        self.assertIn("품질 평가 한계", result["final_report"])


if __name__ == "__main__":
    unittest.main()
