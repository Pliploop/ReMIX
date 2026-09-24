# ReMIX-B relevance grading v2

Replaces the v1 pool grading, whose realized grade scale was `{0,1,2,4,5}` — grade 3
was defined but unreachable and grade 5 was an emergent, undocumented value the LLM
judge free-typed. Root cause: the judge wrote the numeric grade itself and the prompt
never pinned a scale, so it drifted away from the code's own `_pool_type_defaults`.

## Principles

1. **Grade is derived, never free-typed.** The judge emits a categorical `quality`
   label; the integer grade is a fixed function of it. Every level is reachable.
2. **Quality axis + orthogonal diagnostics.** The grade measures how good an answer
   the candidate is to *this* turn's query. *Why* it falls short is recorded
   separately (free-form reason + a multi-label failure-mode taxonomy + satisfied/
   failed constraint lists), on **every** candidate — including strong/good matches
   that missed something minor.
3. **Cheap negatives are not judged.** Grade 0 is assigned by a similarity gate, so
   the LLM judge is only spent on plausible candidates.

## Grade scale (0–6)

| grade | `quality` | criterion |
|---|---|---|
| 6 | `exact`     | the designated target `k(q)` (by construction), or a track the judge finds semantically indistinguishable from it |
| 5 | `strong`    | satisfies **all** requested changes **and** all explicit preservation clauses |
| 4 | `good`      | satisfies the requested change and is seed-compatible, but misses a **minor** preservation / is a looser match (miss recorded) |
| 3 | `partial`   | satisfies **part** of the requested change, misses another part |
| 2 | `soft_fail` | surface constraints (tags/vocal/speed) fit, but the requested **semantics** are not achieved |
| 1 | `hard_fail` | fails the **primary** requested change, still musically on-topic |
| 0 | —           | **similarity-gated**: composite score below floor or outside top-K; never seen by the judge |

- **Designated target → 6 by construction** (`is_exact_target`); the judge cannot
  downgrade it (fixes the v1 bug where it landed at "strong").
- **History candidates are graded on merit** — a prior-turn track may legitimately
  satisfy the instruction — and additionally carry a `history_*` flag. History is a
  *diagnostic*, not an automatic grade.

## Judge output contract

```json
{
  "quality": "strong",
  "failure_modes": ["preservation_violated"],
  "reason": "free-form explanation of the gap",
  "satisfied_constraints": ["..."],
  "failed_constraints": ["..."],
  "confidence": 0.0
}
```

- `quality ∈ {exact, strong, good, partial, soft_fail, hard_fail}` → grade `{6,5,4,3,2,1}`.
- `failure_modes`: ≥1 when `quality < strong`; may be `[]` for exact/strong; a minor
  mode is allowed on strong/good to record what missed.
- The judge never emits grade 0 — that is the gated default.

## Failure-mode taxonomy (multi-label)

- **Change:** `change_missing`, `change_partial`, `change_overshoot`
- **Preservation:** `preservation_violated`, `source_incompatible`
- **Semantics:** `caption_semantics_miss`, `surface_only`
- **History:** `history_shortcut`, `history_dependent_satisfied` (positive flag)
- **Degenerate:** `near_duplicate`, `off_topic`

## Judge gate (both conditions)

A candidate is judged only if its composite/rerank score ≥ `near_miss_threshold`
(0.25) **and** it is within the per-query top-K (`max_judge_candidates`). Otherwise it
is grade 0, `off_topic`, unjudged.

## Pool build

- **Dedup:** at most one snippet per `track_id` in a query's pool (keep the
  highest-scoring); drop the target track's other snippets from the negatives.
- Heterogeneous provenance (target/source/seed/history neighbourhoods + typed
  negatives) is unchanged.

## Downstream

- Grade labels become contiguous 0–6 in `scripts/relevance_pool_analysis.py`; the
  grade-distribution, provenance, similarity, judge-vs-heuristic and by-axis figures
  and the paper text regenerate from the rebuilt pool.
