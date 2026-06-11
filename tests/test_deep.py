"""Deeper tests of recall ranking quality and TF-IDF behaviour."""

import unittest

from engram import MemoryStore


class RankingQualityTests(unittest.TestCase):
    def setUp(self):
        # No recency boost so we isolate pure lexical relevance.
        self.store = MemoryStore(
            ":memory:", recency_halflife_days=None, recency_weight=0.0
        )

    def tearDown(self):
        self.store.close()

    def test_rare_term_outranks_common_term(self):
        # 'uranium' is rare across the corpus; 'company' is common. A query for a
        # rare term should rank its memory above generic ones via IDF weighting.
        self.store.remember("the company reported earnings")
        self.store.remember("the company hired staff")
        self.store.remember("the company opened an office")
        self.store.remember("uranium enrichment at the company")
        hits = self.store.recall("uranium", limit=4)
        self.assertTrue(hits)
        self.assertIn("uranium", hits[0].memory.text.lower())

    def test_more_overlap_ranks_higher(self):
        self.store.remember("nuclear power plant safety regulations")
        self.store.remember("nuclear weapons treaty negotiations")
        hits = self.store.recall("nuclear power plant safety", limit=2)
        self.assertEqual(len(hits), 2)
        self.assertIn("power plant safety", hits[0].memory.text.lower())
        self.assertGreater(hits[0].score, hits[1].score)

    def test_min_score_filters(self):
        self.store.remember("alpha beta gamma delta epsilon")
        self.store.remember("alpha")
        # A high threshold should drop the weaker (partial) match.
        hits = self.store.recall("alpha beta gamma delta epsilon", min_score=0.5)
        self.assertTrue(hits)
        for h in hits:
            self.assertGreaterEqual(h.score, 0.5)

    def test_scores_bounded_zero_to_one(self):
        self.store.remember("one two three")
        self.store.remember("three four five")
        for h in self.store.recall("three", limit=5):
            self.assertGreaterEqual(h.score, 0.0)
            self.assertLessEqual(h.score, 1.0 + 1e-9)

    def test_limit_respected(self):
        for i in range(20):
            self.store.remember(f"memory number {i} about trading")
        hits = self.store.recall("trading", limit=5)
        self.assertEqual(len(hits), 5)


class RecencyTests(unittest.TestCase):
    def test_recency_disabled_returns_pure_cosine(self):
        store = MemoryStore(":memory:", recency_weight=0.0)
        store.remember("identical text here")
        store._conn.execute("UPDATE memories SET created_at = created_at - 1000000")
        store._conn.commit()
        store.remember("identical text here")
        hits = store.recall("identical text here", limit=2)
        # With recency off, the two identical memories tie on score.
        self.assertEqual(len(hits), 2)
        self.assertAlmostEqual(hits[0].score, hits[1].score, places=6)
        store.close()


if __name__ == "__main__":
    unittest.main()
