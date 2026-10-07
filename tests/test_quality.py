import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.quality import QualityAssessment, quality_node
from agents.worker import worker_node


class QualityTest(unittest.TestCase):
    def test_worker_error_returns_failed_fallback(self):
        def fail(state):
            raise RuntimeError("unavailable")

        with tempfile.TemporaryDirectory() as directory:
            with (patch("agents.worker.OUTPUT_DIR", Path(directory)),
                  patch.dict("agents.worker.WORKER_NODES", {"market": fail})):
                result = worker_node({"run_id": "test", "task": {
                    "task_id": "market", "worker": "market", "instruction": "", "attempt": 0}})
            outcome = result["worker_results"]["market"]
            self.assertEqual(outcome["status"], "failed")
            self.assertEqual(outcome["error"], "RuntimeError")
            self.assertIn("판단 유보", outcome["payload"]["market_result"]["sw"])
            self.assertTrue(Path(outcome["evidence_path"]).exists())

    @patch("agents.quality._citation_validation_error", return_value=None)
    @patch("agents.quality._validate_report_structure")
    @patch("agents.quality.get_llm")
    def test_judge_failure_is_not_success(self, model, structure, citations):
        model.side_effect = RuntimeError("offline")
        result = quality_node({"final_report": "## 7. REFERENCE", "worker_results": {}})
        self.assertFalse(result["eval_result"]["passed"])

    @patch("agents.quality._citation_validation_error", return_value=None)
    @patch("agents.quality._validate_report_structure")
    @patch("agents.quality.get_llm")
    def test_judge_failure_target_is_preserved(self, model, structure, citations):
        model.return_value.with_structured_output.return_value.invoke.return_value = QualityAssessment(
            groundedness=False, neutrality=True, bias_control=True, coverage=True,
            reasons=["시장성 근거 미지원"], rework_targets=["market"])
        result = quality_node({"final_report": "## 7. REFERENCE", "worker_results": {}})
        self.assertFalse(result["eval_result"]["passed"])
        self.assertEqual(result["eval_result"]["rework_targets"], ["market"])


if __name__ == "__main__":
    unittest.main()
