import unittest

from agents.tech_research import _researchable_keywords


class TechResearchTest(unittest.TestCase):
    def test_retry_excludes_items_outside_paper_scope(self):
        keywords = _researchable_keywords([
            "source code release",
            "GPU model used in experiments",
            "customer deployment",
            "commercial deployment",
            "multi-GPU benchmark",
            "serving framework integration",
        ])

        self.assertEqual(keywords, [
            "source code release",
            "GPU model used in experiments",
            "serving framework integration",
        ])

    def test_empty_retry_when_only_out_of_scope_items_remain(self):
        self.assertEqual(_researchable_keywords(["customer deployment"]), [])


if __name__ == "__main__":
    unittest.main()
