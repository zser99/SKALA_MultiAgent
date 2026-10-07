import unittest
from copy import deepcopy
from unittest.mock import patch

from pydantic import ValidationError

from agents.synthesis import SynthesisResult, synthesis_node
from tests.mock_state import MOCK_STATE


class SynthesisTest(unittest.TestCase):
    @patch("agents.synthesis.get_llm")
    def test_structured_output_preserves_state_contract(self, mock_get_llm):
        structured = mock_get_llm.return_value.with_structured_output.return_value
        structured.invoke.return_value = SynthesisResult(
            agreements="공통 평가",
            conflicts="서로 다른 조건",
            implications="조건부 판단",
        )

        result = synthesis_node(deepcopy(MOCK_STATE))

        self.assertEqual(result, {"synthesis": {
            "agreements": "공통 평가",
            "conflicts": "서로 다른 조건",
            "implications": "조건부 판단",
        }})
        mock_get_llm.return_value.with_structured_output.assert_called_once_with(SynthesisResult)
        self.assertIn("[도메인 평가]", structured.invoke.call_args.args[0])

    def test_missing_output_key_is_rejected(self):
        with self.assertRaises(ValidationError):
            SynthesisResult.model_validate({"agreements": "공통 평가", "conflicts": "차이"})


if __name__ == "__main__":
    unittest.main()
