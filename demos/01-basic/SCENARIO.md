# Demo 01 — Remember and recall across a session

This scenario seeds an `engram` memory from `seed_memories.jsonl`, then asks it
to recall the most relevant items for a few natural-language queries. It shows
the core loop an agent uses: persist durable facts, then pull back the right
ones on demand — with no LLM, embedding API, or network access.

## Run it

```bash
# stdlib only — runs in place, no install needed
python demos/01-basic/run.py
```

The script uses a throwaway SQLite file in a temp directory, so it leaves no
state behind. To drive engram from the CLI directly against a real file:

```bash
python -m engram --db agent.sqlite remember "User prefers metric units" --tags preference
python -m engram --db agent.sqlite recall "what units does the user want"
python -m engram --db agent.sqlite list
python -m engram --db agent.sqlite stats
```

## What it should show

The seed file mixes preferences, ops notes, and unrelated facts. For the query
*"what units does the user prefer"*, the top hit is the metric-units
preference — not the trading-bot note or the geography fact — because TF-IDF
cosine similarity rewards the overlapping, discriminating terms. The demo
prints each query, its ranked hits with scores, and exits 0 on success.
