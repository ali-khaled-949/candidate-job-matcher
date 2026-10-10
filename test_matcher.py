import math
import unittest

from matcher import BM25, Doc, evaluate, ndcg_at_k, rank, recall_at_k, skill_overlap

# Synthetic candidates. The keyword-stuffed profile repeats "python" but lacks the skills the query needs.
DOCS = [
    Doc("stuffed", "python python python python python developer python scripts", {"python"}),
    Doc("backend", "backend engineer building python apis with sql and docker", {"python", "sql", "docker", "apis"}),
    Doc("data", "data analyst with sql dashboards and reporting", {"sql", "reporting"}),
    Doc("frontend", "frontend developer react typescript css", {"react", "typescript", "css"}),
]
QUERY = "python developer"
SKILLS = {"python", "apis", "sql", "docker"}


class MatcherTests(unittest.TestCase):
    def test_bm25_scores_matching_docs_higher(self):
        scores = BM25(DOCS).score("react typescript")
        self.assertEqual(scores.index(max(scores)), 3)

    def test_skill_overlap(self):
        self.assertAlmostEqual(skill_overlap({"a", "b"}, {"b", "c"}), 1 / 3)
        self.assertEqual(skill_overlap(set(), {"a"}), 0.0)

    def test_text_only_baseline_is_fooled_by_keyword_stuffing(self):
        self.assertEqual(rank(QUERY, SKILLS, DOCS, alpha=0.0)[0], "stuffed")

    def test_skill_reranking_fixes_it(self):
        self.assertEqual(rank(QUERY, SKILLS, DOCS, alpha=0.6)[0], "backend")

    def test_metrics(self):
        self.assertEqual(recall_at_k(["a", "b", "c"], {"a", "c"}, 2), 0.5)
        self.assertAlmostEqual(ndcg_at_k(["a", "b"], {"a": 3, "b": 1}, 2), 1.0)
        worse = ndcg_at_k(["b", "a"], {"a": 3, "b": 1}, 2)
        self.assertLess(worse, 1.0)
        self.assertTrue(math.isfinite(worse))

    def test_blend_beats_baseline_on_labeled_query(self):
        cases = [{"query": QUERY, "skills": SKILLS, "gains": {"backend": 3, "data": 1}}]
        base = evaluate(cases, DOCS, alpha=0.0, k=2)
        blend = evaluate(cases, DOCS, alpha=0.6, k=2)
        self.assertGreater(blend["ndcg@k"], base["ndcg@k"])


if __name__ == "__main__":
    unittest.main()
