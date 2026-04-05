# Phase 01: Core Algorithm & Config - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-05
**Phase:** 01-Core Algorithm & Config
**Areas discussed:** Parameter defaults, IDF edge handling, TF precision, Statistics storage

---

## Parameter Defaults

| Option | Description | Selected |
|--------|-------------|----------|
| Lucene defaults (k1=1.2, b=0.75) | Industry standard, battle-tested | ✓ |
| Elasticsearch defaults (k1=1.2, b=0.75) | Same as Lucene, widely used | |
| BM25+ variant | Different k1 recommendation, not needed | |

**User's choice:** Lucene defaults (k1=1.2, b=0.75)
**Notes:** [auto] Selected recommended default — industry standard from Lucene/Elasticsearch

---

## IDF Edge Handling

| Option | Description | Selected |
|--------|-------------|----------|
| Floor at 0 | Prevent negative scores for common terms | ✓ |
| Allow negative IDF | Standard formula, penalizes common terms | |
| Use BM25+ variant | Lower-bounded IDF, adds complexity | |

**User's choice:** Floor at 0
**Notes:** [auto] Selected recommended approach — prevents negative scores that break hybrid search

---

## TF Precision

| Option | Description | Selected |
|--------|-------------|----------|
| f64 intermediate, f32 output | Maximize precision, consistent API | ✓ |
| f32 throughout | Simpler, less precision | |
| f64 throughout | Maximum precision, larger memory | |

**User's choice:** f64 intermediate, f32 output
**Notes:** [auto] Selected recommended approach — balances precision with API consistency

---

## Statistics Storage

| Option | Description | Selected |
|--------|-------------|----------|
| rkyv in bm25_stats/ subdirectory | Zero-copy, segment-aligned | ✓ |
| bincode in segment root | Alternative serialization | |
| Inline in term index | Combined storage | |

**User's choice:** rkyv in bm25_stats/ subdirectory
**Notes:** [auto] Selected recommended approach — consistent with existing posting list format

---

## the agent's Discretion

- Tokenization pipeline design (adapt from existing TermIndex patterns)
- Exact struct layout for BM25Config (follow existing Config patterns)
- Unit test organization (follow existing test patterns)

## Deferred Ideas

None
