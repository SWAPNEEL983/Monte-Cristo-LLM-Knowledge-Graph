# Monte-Cristo-LLM-Knowledge-Graph

An agentic LLM pipeline that reads *The Count of Monte Cristo* end-to-end and builds a character-centric knowledge graph — causal event extraction, entity/relationship extraction, alias resolution, and per-character spatial/attribute/causal summarization — using Mistral (`ministral-8b-latest`).

## Overview

This project processes the full novel through a multi-stage agentic pipeline to construct a structured knowledge graph of its characters, their relationships, and their narrative arcs, without any manual annotation of the source text.

**Pipeline:**

1. **Causal skeleton extraction** — the novel is swept with a sliding window (3000 chars, 2800 stride) and an LLM agent flags "causal anchors": identity/origin facts, formative experiences, physical constraints, causal pivots, and world rules. Run in parts with a gap-recovery pass for windows that failed mid-run.
2. **Character profile construction** — causal events are aggregated per character into attribute, location, and timeline lists.
3. **Knowledge graph triplet extraction** — a second LLM pass (LightRAG-style prompt) extracts `(entity, type, description)` and `(source, target, relationship, keywords, strength)` triplets from the causal skeleton.
4. **Character name deduplication** — the novel's characters appear under many aliases, titles, and LLM-hallucinated name variants (e.g. "Edmond Dantès" also appears as "The Count," "Sinbad the Sailor," "Lord Wilmore," "Abbé Busoni," etc.). Resolved via string-similarity audits plus an exhaustive hand-built alias map.
5. **Knowledge graph assembly** — two passes, the second adding fuzzy/thematic name resolution and evidence injection from the extracted triplets.
6. **Final entity resolution & deep merge** — a consolidated alias map and a lossless merge that aggregates every field across all node variants into one hub profile per character.
7. **Hierarchical attribute agents** — three stateful LLM agents deduplicate and normalize each character's locations, traits, and causal timeline into compact, high-density summaries.
8. **Star-graph fusion & visualization** — per-character spatial/attribute/causal pillars are fused into a single graph and rendered as an interactive HTML network (pyvis).

## Example

Character name deduplication alone collapses well over 100 raw name variants (from LLM extraction noise and in-text aliases/titles) down to canonical identities — e.g. "The Count," "Sinbad the Sailor," "Lord Wilmore," "Abbé Busoni," "Zaccone," and a dozen parenthetical variants all resolve to one node: **Edmond Dantès**.

## Limitations

- **Alias resolution is hand-curated, not general.** `MASTER_MAP`, `REPAIR_MAP`, and `alias_map` were built by manually inspecting LLM output for this specific novel. Running this pipeline on a different book means rebuilding these maps from scratch — this is not a plug-and-play entity-resolution system.
- **No ground-truth evaluation.** Extracted relationships and character facts are not checked against a hand-labeled sample of the text — correctness was assessed by spot-checking, not measured.
- **A couple of path/filename inconsistencies remain from iterative development**, flagged inline in the notebook with ⚠️ markers — verify these before running end-to-end on a fresh environment.
- **API-cost and runtime dependent.** The full pipeline makes hundreds of LLM calls (one per text window, plus the agentic hierarchy passes); expect it to take a while and to consume a non-trivial amount of API quota.

## Setup

```bash
pip install -r requirements.txt
```

You'll also need:
- A [Mistral API key](https://console.mistral.ai/) (the notebook reads it via Kaggle Secrets — replace with your own secret-management approach if running outside Kaggle).
- The novel's text file (public domain — available from [Project Gutenberg](https://www.gutenberg.org/ebooks/1184)).

## Repo contents

- `monte-cristo-knowledge-graph.ipynb` — full pipeline, organized into 9 labeled sections following the stages above.

## Possible next steps

- Replace the hand-built alias maps with a generalizable coreference/entity-linking approach (e.g. embedding-based clustering over extracted entity mentions).
- Add a sampled evaluation: hand-check a subset of extracted relationships against the source text for precision.
- Generalize the pipeline to run on an arbitrary novel with minimal manual intervention.
