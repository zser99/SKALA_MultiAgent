import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from agents.report import (
    _strip_reference_section,
    report_node,
)
from tests.mock_state import MOCK_STATE
# 정상 보고서 샘플
VALID_REPORT_BODY = """## 0. SUMMARY
요약

## 1. 분석 배경
배경

## 2. 대상 기술 선정
### 2.1 SW 기술
SW 기술
### 2.2 HW 기술
HW 기술
### 2.3 선정 근거
선정 근거

## 3. 기술 개요
### 3.1 SW 기술 원리 및 한계
SW 개요
### 3.2 HW 기술 원리 및 한계
HW 개요

## 4. 관점별 평가
### 4.1 기술 성숙도
TRL
### 4.2 시장성
시장성
### 4.3 이해관계자
이해관계자
### 4.4 도메인 적용성
도메인

## 5. 관점 간 비교 및 시사점
### 5.1 일치하는 평가
공통점
### 5.2 충돌하는 평가
충돌점
### 5.3 시사점
시사점

## 6. 분석 한계
### 6.1 공개 정보 기반 추정의 한계
공개 정보의 한계
### 6.2 확증편향 방지 방법
확증편향 방지 방법
"""

KIVI_SOURCE = {
    "id": "src_kivi",
    "type": "paper",
    "authors": "Liu, Z., Yuan, J., Jin, H., et al.",
    "year": "2024",
    "title": "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache",
    "venue": "ICML 2024",
    "url": "https://arxiv.org/abs/2402.02750",
}

ITME_SOURCE = {
    "id": "src_itme",
    "type": "paper",
    "authors": "Test Authors",
    "year": "2026",
    "title": "ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories",
    "venue": "Test Conference",
    "url": "https://arxiv.org/abs/2606.12556",
}


def make_valid_state():
    state = deepcopy(MOCK_STATE)

    state["tech_research"]["sw"]["sources"] = [KIVI_SOURCE]
    state["tech_research"]["hw"]["sources"] = [ITME_SOURCE]

    state["market_result"]["sources"] = []
    state["stakeholder_result"]["sources"] = []
    state["domain_result"]["sources"] = []
    state["warnings"] = []

    return state


class ReportTest(unittest.TestCase):
    def test_missing_state_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "보고서 생성에 필요한 State 값",
        ):
            report_node({})

    def test_llm_reference_section_is_removed(self):
        report = (
            "## 0. SUMMARY\n"
            "본문\n\n"
            "## 7. REFERENCE\n"
            "- LLM이 만든 가짜 참고문헌"
        )

        stripped = _strip_reference_section(report)

        self.assertIn("## 0. SUMMARY", stripped)
        self.assertNotIn("가짜 참고문헌", stripped)

    @patch("agents.report.get_llm")
    def test_report_node_appends_verified_references(
        self,
        mock_get_llm,
    ):
        mock_get_llm.return_value.invoke.return_value = SimpleNamespace(
            content=(
                VALID_REPORT_BODY
                + "\n## 7. REFERENCE\n"
                + "- LLM이 만든 가짜 참고문헌"
            )
        )

        result = report_node(make_valid_state())
        report = result["final_report"]

        self.assertEqual(report.count("## 7. REFERENCE"), 1)
        self.assertNotIn("LLM이 만든 가짜 참고문헌", report)
        self.assertIn("KIVI", report)
        self.assertIn("ITME", report)
        self.assertEqual(result["warnings"], [])

        mock_get_llm.assert_called_once_with(temperature=0.2)
        mock_get_llm.return_value.invoke.assert_called_once()

    @patch("agents.report.get_llm")
    def test_report_with_missing_headings_is_rejected(
        self,
        mock_get_llm,
    ):
        mock_get_llm.return_value.invoke.return_value = SimpleNamespace(
            content=(
                "## 0. SUMMARY\n"
                "목차가 일부 누락된 보고서"
            )
        )

        with self.assertRaisesRegex(ValueError, "필수 목차"):
            report_node(make_valid_state())


       
if __name__ == "__main__":
    unittest.main()