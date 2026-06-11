"""Smoke tests: the package imports, identity is sane, basic round-trip works."""

import unittest

import engram
from engram import MemoryStore, TOOL_NAME, TOOL_VERSION


class SmokeTests(unittest.TestCase):
    def test_identity(self):
        self.assertEqual(TOOL_NAME, "engram")
        self.assertTrue(TOOL_VERSION)
        self.assertEqual(engram.__version__, TOOL_VERSION)

    def test_public_api_present(self):
        for name in ("Memory", "MemoryStore", "RecallHit", "tokenize"):
            self.assertTrue(hasattr(engram, name), name)

    def test_remember_recall_round_trip(self):
        with MemoryStore(":memory:") as mem:
            mem.remember("the user prefers metric units", tags=["preference"])
            hits = mem.recall("what units does the user want")
            self.assertTrue(hits)
            self.assertIn("metric", hits[0].memory.text.lower())


if __name__ == "__main__":
    unittest.main()
