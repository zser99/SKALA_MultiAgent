import unittest

from agents.references import (
    audit_sources,
    collect_sources,
    render_references,
)
from tests.mock_state import MOCK_STATE, SOURCE_REGISTRY


class ReferencesTest(unittest.TestCase):
    def test_collect_sources_removes_duplicates(self):
        sources = collect_sources(MOCK_STATE, SOURCE_REGISTRY)

        self.assertEqual(len(sources), 7)
        self.assertEqual(
            len({source["id"] for source in sources}),
            len(sources),
        )

    def test_render_references_groups_source_types(self):
        rendered = render_references(MOCK_STATE, SOURCE_REGISTRY)

        self.assertIn("**논문**", rendered)
        self.assertIn("**특허**", rendered)
        self.assertIn("**기타(웹)**", rendered)
        self.assertIn("KIVI", rendered)
        self.assertIn("ITME", rendered)

    def test_unknown_source_is_reported_as_incomplete(self):
        state = {
            "market_result": {
                "sources": ["src_unknown"],
            }
        }

        result = audit_sources(state)

        self.assertEqual(result["total"], 1)
        self.assertEqual(len(result["incomplete"]), 1)
        self.assertEqual(
            result["incomplete"][0]["id"],
            "src_unknown",
        )

    def test_basis_labels_are_not_treated_as_sources(self):
        state = {
            "market_result": {
                "sources": [
                    "sw_basis=paper",
                    "hw_basis=general knowledge",
                ],
            }
        }

        result = audit_sources(state)

        self.assertEqual(result["total"], 0)
        self.assertEqual(len(result["dropped"]), 2)

    def test_placeholder_sources_are_reported_as_incomplete(self):
        result = audit_sources(MOCK_STATE, SOURCE_REGISTRY)
        incomplete_ids = {
            source["id"]
            for source in result["incomplete"]
        }

        self.assertIn("src_itme", incomplete_ids)
        self.assertIn("src_market_01", incomplete_ids)
        self.assertIn("src_domain_02", incomplete_ids)


if __name__ == "__main__":
    unittest.main()