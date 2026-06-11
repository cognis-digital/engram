"""engram — durable, model-agnostic long-term memory for AI agents.

Part of the Cognis Neural Suite.

The public surface is intentionally tiny:

    from engram import MemoryStore

    mem = MemoryStore("agent.sqlite")
    mem.remember("the sky is blue", tags=["fact"])
    hits = mem.recall("what color is the sky")

Everything is backed by a single SQLite file and a self-contained TF-IDF + cosine
similarity ranker. No external services, no embedding APIs, standard library only.
"""

from .memory import Memory, MemoryStore, RecallHit, tokenize

TOOL_NAME = "engram"
TOOL_VERSION = "0.1.0"

__all__ = ["Memory", "MemoryStore", "RecallHit", "tokenize", "TOOL_NAME", "TOOL_VERSION"]

__version__ = TOOL_VERSION
