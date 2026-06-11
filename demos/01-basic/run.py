#!/usr/bin/env python3
"""Demo 01 — seed an engram memory, then recall against natural-language queries.

Standard library only. Runs against a throwaway SQLite file in a temp directory
so it leaves no state behind. Exits 0 on success, non-zero if recall looks wrong.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

# Allow running from a checkout without installing the package.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from engram import MemoryStore  # noqa: E402

SEED = os.path.join(_HERE, "seed_memories.jsonl")

QUERIES = [
    "what units does the user prefer",
    "which ticker is on the watchlist",
    "are we allowed to place live orders",
    "what sectors does the user care about",
]


def main() -> int:
    with open(SEED, "r", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]

    tmpdir = tempfile.mkdtemp(prefix="engram-demo-")
    db_path = os.path.join(tmpdir, "demo.sqlite")
    store = MemoryStore(db_path)
    try:
        for rec in records:
            store.remember(
                rec["text"],
                tags=rec.get("tags"),
                source=rec.get("source"),
            )
        print(f"seeded {store.count()} memories into {db_path}\n")

        ok = True
        for q in QUERIES:
            hits = store.recall(q, limit=3)
            print(f"query: {q!r}")
            if not hits:
                print("  (no relevant memories)")
                ok = False
            for h in hits:
                m = h.memory
                tagstr = f" [{', '.join(m.tags)}]" if m.tags else ""
                print(f"  {h.score:6.3f}  #{m.id}{tagstr}  {m.text}")
            print()

        # Sanity check: the units query must surface the metric-units memory first.
        top = store.recall("what units does the user prefer", limit=1)
        if not top or "metric" not in top[0].memory.text.lower():
            print("ERROR: expected the metric-units memory to rank first", file=sys.stderr)
            return 1
        return 0 if ok else 1
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
