import unittest

from agents.evidence import evidence_check_node


class EvidenceTest(unittest.TestCase):
    def test_unknown_limitations_do_not_trigger_market_retry(self):
        state = {
            "tech_research": {"sw": {"overview": "확인됨"}, "hw": {"overview": "확인됨"}},
            "market_result": {
                "sw": "시장 규모·성장성: 확인 불가\n종합: 판단 보류",
                "hw": "경제적 효과: 확인 불가",
            },
            "domain_result": {"sw": "요약", "hw": "요약"},
        }

        result = evidence_check_node(state)

        self.assertTrue(result["evidence_sufficient"])
        self.assertEqual(result["warnings"], [])

    def test_warning_names_missing_market_and_domain_items(self):
        state = {
            "tech_research": {"sw": {"overview": "확인됨"}, "hw": {"overview": "확인됨"}},
            "market_result": {
                "sw": "시장 규모·성장성: 검색된 발췌 없음\n종합: 판단 보류",
                "hw": "경제적 효과: 검색 결과 없음",
            },
            "domain_result": {
                "sw": "요약", "hw": "요약",
                "domain_analysis": {"evidence_gaps": [
                    {"technology": "sw", "criterion": "latency"},
                    {"technology": "hw", "criterion": "quality"},
                    {"technology": "sw", "criterion": "quality"},
                    {"technology": "hw", "criterion": "latency"},
                    {"technology": "sw", "criterion": "throughput"},
                    {"technology": "hw", "criterion": "throughput"},
                ]},
            },
        }

        result = evidence_check_node(state)

        self.assertFalse(result["evidence_sufficient"])
        warning = result["warnings"][0]
        self.assertIn("market.sw.시장 규모·성장성", warning)
        self.assertIn("market.hw.경제적 효과", warning)
        self.assertIn("domain.sw.latency", warning)
        self.assertIn("domain.hw.quality", warning)

    def test_small_domain_gaps_are_limitations_not_retry_trigger(self):
        state = {
            "tech_research": {"sw": {"overview": "확인됨"}, "hw": {"overview": "확인됨"}},
            "market_result": {"sw": "요약", "hw": "요약"},
            "domain_result": {
                "sw": "요약", "hw": "요약",
                "domain_analysis": {"evidence_gaps": [
                    {"technology": "sw", "criterion": "latency"},
                    {"technology": "hw", "criterion": "quality"},
                ]},
            },
        }

        result = evidence_check_node(state)

        self.assertTrue(result["evidence_sufficient"])
        self.assertEqual(result["warnings"], [])


if __name__ == "__main__":
    unittest.main()
