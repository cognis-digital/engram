"""Tests for hardened error-handling and edge-case paths added during production hardening.

Covers:
  - memory.py: bad limit/offset/min_score validation in recall() and list()
  - cli.py: bad --db path, empty remember text, bad --limit value
  - mcp_server.py: bad limit/min_score in tool calls, bad DB path in main()
"""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from engram import MemoryStore
from engram import cli
from engram.mcp_server import EngramMCPServer, serve_stdio


# ---------------------------------------------------------------------------
# memory.py hardening
# ---------------------------------------------------------------------------

class MemoryStoreValidationTests(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore(":memory:")

    def tearDown(self):
        self.store.close()

    # -- recall() --

    def test_recall_negative_limit_raises(self):
        self.store.remember("anything")
        with self.assertRaises(ValueError):
            self.store.recall("anything", limit=0)

    def test_recall_limit_minus_one_raises(self):
        with self.assertRaises(ValueError):
            self.store.recall("anything", limit=-5)

    def test_recall_non_int_limit_raises(self):
        with self.assertRaises(TypeError):
            self.store.recall("anything", limit=2.5)  # type: ignore[arg-type]

    def test_recall_bad_min_score_raises(self):
        with self.assertRaises(ValueError):
            self.store.recall("anything", min_score="high")  # type: ignore[arg-type]

    def test_recall_valid_limit_one_works(self):
        self.store.remember("alpha beta gamma")
        hits = self.store.recall("alpha", limit=1)
        self.assertLessEqual(len(hits), 1)

    # -- list() --

    def test_list_zero_limit_raises(self):
        with self.assertRaises(ValueError):
            self.store.list(limit=0)

    def test_list_negative_offset_raises(self):
        with self.assertRaises(ValueError):
            self.store.list(offset=-1)

    def test_list_non_int_limit_raises(self):
        with self.assertRaises(TypeError):
            self.store.list(limit=1.5)  # type: ignore[arg-type]

    def test_list_empty_store_returns_empty_list(self):
        result = self.store.list(limit=10)
        self.assertEqual(result, [])

    def test_list_offset_beyond_count_returns_empty(self):
        self.store.remember("only one")
        result = self.store.list(limit=10, offset=999)
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# cli.py hardening
# ---------------------------------------------------------------------------

class CLIHardeningTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self._tmp.name, "test.sqlite")

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, *args):
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            code = cli.main(["--db", self.db, *args])
        return code, out_buf.getvalue(), err_buf.getvalue()

    def test_bad_db_path_returns_nonzero_with_message(self):
        """A DB path in a nonexistent directory should print an error and exit 2."""
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        bad_db = os.path.join(self._tmp.name, "no_such_dir", "x.sqlite")
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            code = cli.main(["--db", bad_db, "stats"])
        self.assertEqual(code, 2)
        self.assertIn("error", err_buf.getvalue().lower())

    def test_remember_empty_text_returns_nonzero(self):
        """Passing whitespace-only text should not traceback — clean error + exit 2."""
        code, out, err = self._run("remember", "   ")
        self.assertEqual(code, 2)
        self.assertIn("error", err.lower())

    def test_recall_limit_zero_gives_argparse_error(self):
        """--limit 0 should be rejected by argparse before hitting the store."""
        err_buf = io.StringIO()
        with redirect_stderr(err_buf):
            with self.assertRaises(SystemExit) as ctx:
                cli.main(["--db", self.db, "recall", "query", "--limit", "0"])
        self.assertNotEqual(ctx.exception.code, 0)

    def test_list_negative_offset_gives_argparse_error(self):
        """--offset -1 should be rejected by argparse."""
        err_buf = io.StringIO()
        with redirect_stderr(err_buf):
            with self.assertRaises(SystemExit) as ctx:
                cli.main(["--db", self.db, "list", "--offset", "-1"])
        self.assertNotEqual(ctx.exception.code, 0)

    def test_invalid_metadata_json_exits_nonzero(self):
        """--metadata with invalid JSON should print an error and exit non-zero."""
        with self.assertRaises(SystemExit) as ctx:
            self._run("remember", "some text", "--metadata", "{not-json}")
        self.assertNotEqual(ctx.exception.code, 0)

    def test_remember_valid_text_still_works(self):
        """Sanity: hardening must not break normal operation."""
        code, out, err = self._run("remember", "metric units preferred")
        self.assertEqual(code, 0)
        self.assertIn("remembered", out)


# ---------------------------------------------------------------------------
# mcp_server.py hardening
# ---------------------------------------------------------------------------

class MCPServerHardeningTests(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore(":memory:")
        self.server = EngramMCPServer(self.store)

    def tearDown(self):
        self.store.close()

    def _call(self, tool_name, arguments, msg_id=1):
        resp = self.server.handle({
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        })
        return resp

    def test_recall_negative_limit_is_tool_error(self):
        resp = self._call("recall", {"query": "anything", "limit": 0})
        self.assertTrue(resp["result"]["isError"])
        self.assertIn("limit", resp["result"]["content"][0]["text"])

    def test_recall_non_numeric_min_score_is_tool_error(self):
        resp = self._call("recall", {"query": "anything", "min_score": "high"})
        self.assertTrue(resp["result"]["isError"])

    def test_list_memories_negative_limit_is_tool_error(self):
        resp = self._call("list_memories", {"limit": -1})
        self.assertTrue(resp["result"]["isError"])

    def test_list_memories_zero_limit_is_tool_error(self):
        resp = self._call("list_memories", {"limit": 0})
        self.assertTrue(resp["result"]["isError"])

    def test_broken_pipe_in_serve_stdio_does_not_raise(self):
        """BrokenPipeError on stdout write must be swallowed, not propagated."""
        class BrokenStdout:
            def write(self, data):
                raise BrokenPipeError("broken pipe")
            def flush(self):
                pass

        lines = [
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
        ]
        stdin = io.StringIO("\n".join(lines) + "\n")
        # Should not raise even though stdout always raises BrokenPipeError.
        try:
            serve_stdio(self.store, stdin=stdin, stdout=BrokenStdout())
        except BrokenPipeError:
            self.fail("serve_stdio propagated BrokenPipeError")

    def test_mcp_main_bad_db_returns_nonzero(self):
        """main() with an unwritable DB path should return 2, not traceback."""
        from engram.mcp_server import main as mcp_main
        bad_db = os.path.join(tempfile.gettempdir(), "no_such_dir_xyz", "x.sqlite")
        err_buf = io.StringIO()
        with redirect_stderr(err_buf):
            code = mcp_main(["--db", bad_db])
        self.assertEqual(code, 2)
        self.assertIn("error", err_buf.getvalue().lower())


if __name__ == "__main__":
    unittest.main()
