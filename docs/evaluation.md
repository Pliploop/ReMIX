# ReMIX-B evaluation: baselines + graded-pool API

Design for the retrieval benchmark harness and the baseline suite. **Design only —
nothing here is implemented yet.** Everything runs against the graded relevance
pool built by `stages/relevance_pool.py` (grading spec: `benchmark_grading_v2.md`).

## 1. Task

A query is a chain step: a **seed** clip `q_a` + an **instruction** `q_i`
(history-unaware; history-aware adds prior turns). Rank the **corpus** (all
test-split clips, artist-disjoint) so that clips satisfying the instruction while
staying compatible with the seed appear first. Scored with graded metrics against
the pool; clips outside the pool count as non-relevant.

## 2. Data contracts

All formats are plain dicts / JSONL so any model can produce them.

**Corpus** — the retrieval index (test split):
```json
{"clip_id": "…::0-30::1", "caption": "...", "tags": ["..."],
 "vocals": "vocal", "speed": "fast", "audio_emb_id": 12345, "text_emb_id": 12345}
```
Audio/text embeddings are reused from `embeddings_reused_audio_text/` (MuQ-MuLan
audio + text towers, EmbeddingGemma text) — the harness loads them by id, it does
not re-embed unless a baseline asks.

**Query** — one per test step:
```json
{"query_id": "chain_00000067#1", "seed_clip_id": "...", "source_clip_id": "...",
 "instruction": "add screamed vocals ...", "instruction_history_aware": "...",
 "history": [{"seed": "...", "instruction": "...", "target": "..."}]}
```

**Graded pool (qrels)** — the ground truth, one per query:
```json
{"query_id": "chain_00000067#1", "grades": {"clipA": 6, "clipB": 5, "clipC": 1}}
```
Grades are the v2 scale (0–6). Clips absent from `grades` are non-relevant (0).

**Results** — what a baseline returns, one per query:
```json
{"query_id": "chain_00000067#1", "ranking": [["clipX", 0.98], ["clipY", 0.71]]}
```
A ranked list of `(clip_id, score)` over the corpus (top-`k` is enough; the tail
is treated as unranked / score $-\infty$).

## 3. `grade_pool` — the benchmark object

The benchmark is built from the pool via `grade_pool(...)`, which returns a
`GradedPool`. It **is** the "correct composed results": it holds, per query, the
graded relevant clips, and it scores any submitted `Results`.

```python
def grade_pool(pool_path_or_records, *, positive_min=3, splits=("test",)) -> GradedPool: ...

class GradedPool:
    qrels: dict[str, dict[str, int]]          # query_id -> {clip_id: grade}   (the "correct" answers)
    def relevant(self, query_id, min_grade=1) -> dict[str, int]: ...   # graded relevant set
    def positives(self, query_id) -> set[str]: ...                     # grade >= positive_min
    def evaluate(self, results, metrics=DEFAULT_METRICS) -> Report: ...  # score a baseline
```

- `grade_pool` accepts the pool in the **compatible format** of §2 (or reads the
  raw `chain_step_relevance_pools.shard*.jsonl` and projects it to qrels).
- `evaluate(results)` returns a `Report` (overall metrics + per-slice breakdowns);
  it never re-reads the pool, so the same `GradedPool` grades every baseline
  identically.

## 4. `Baseline` — the common model API

Every baseline (naive → trained) implements one interface, so the harness treats
them uniformly:

```python
class Baseline(Protocol):
    name: str
    def prepare(self, corpus: Corpus) -> None: ...        # build index / cache corpus embeddings (once)
    def rank(self, query: Query, k: int) -> list[tuple[str, float]]: ...   # ranking for one query
    # optional: rank_batch(queries, k) for models that batch efficiently
```

Harness driver:
```python
def run_benchmark(baseline, corpus, queries, graded_pool, k=1000) -> Report:
    baseline.prepare(corpus)
    results = [{"query_id": q.id, "ranking": baseline.rank(q, k)} for q in queries]
    return graded_pool.evaluate(results)
```
A baseline is thus anything that maps (seed, instruction) → a corpus ranking; the
grading is entirely on the `GradedPool` side.

## 5. Metrics

Graded (grade = gain): **nDCG@{10,100}**, **MAP**, **Recall@{10,100}**, **MRR**
(via `ir_measures`/`pytrec_eval`). Binary-relevance cuts at `positive_min`.
Reported overall and sliced by: verified-grade of the target, edit axis,
history-aware vs history-unaware, and #turns. A **designated-target Recall@k**
(did we find `k(q)`) is reported alongside pooled metrics.

## 6. Baseline catalog (all under §4 API)

| baseline | needs | signal |
|---|---|---|
| `Random` | — | lower bound |
| `TargetCaptionOracle` | true target caption | upper bound |
| `SeedAudioNN` | audio emb | ignores instruction |
| `SeedCaptionNN` | text emb | ignores instruction |
| `InstructionText` | text emb | instruction only, ignores seed |
| `CaptionPlusInstruction` | text emb | naive concat → text retrieval |
| `BM25CaptionPlusInstruction` | lexical | naive lexical |
| `LateFusion(seed_audio + instr_text)` | audio+text emb, weight | composed (CIR-style) |
| `MuLanZeroShot` | MuLan towers | instruction→audio joint space |
| `LLMCaptionRewrite` | LLM + text emb | rewrite target caption → retrieve |
| `LLMPointwiseReranker` | first-stage + LLM | rerank top-k by LLM yes/no |
| `ReMIX-C` (trained) | MuQ + adapter, training loop | audio⊕instruction cross-attn, contrastive with pool hard negatives |
| multi-turn: `LatestTurnOnly`, `FullHistory` | any composed model | history dependence |

All zero-shot baselines share the reused embeddings; only `LLMCaptionRewrite`,
`LLMPointwiseReranker`, and `ReMIX-C` call models at eval/train time.

## 7. Code layout (planned)

```
src/jamendo_instruct/benchmark/
  contracts.py     # Corpus, Query, Results dataclasses + JSONL IO
  graded_pool.py   # grade_pool(), GradedPool, Report, metrics
  baselines/
    base.py        # Baseline protocol + shared embedding index helpers
    trivial.py     # Random, oracle, seed/instruction NN, concat, BM25
    fusion.py      # LateFusion, MuLanZeroShot
    llm.py         # LLMCaptionRewrite, LLMPointwiseReranker
    remix_c.py     # trained adapter (+ train loop under train/)
scripts/
  export_benchmark.py   # pool + run dirs -> corpus.jsonl, queries.jsonl, qrels.jsonl
  run_benchmark.py      # --baseline X -> Report (json + table)
```

## 8. Build order

1. **Contracts + `grade_pool`/`GradedPool` + metrics** (this is the API; a tiny
   synthetic test proves the scoring).
2. **`export_benchmark.py`** — project the pool + run into corpus/queries/qrels.
3. **Trivial baselines** (Random, oracle, seed/instruction NN, concat, BM25) →
   first results table on the partial pool.
4. **Fusion + MuLan zero-shot.**
5. **LLM baselines.**
6. **ReMIX-C** (adapter + contrastive train loop, pool hard negatives).

Runs on the partial pool now; re-run at 100% for final numbers.
