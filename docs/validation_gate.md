# Validation: instruction validity vs. relevance grading

There are **two distinct grading questions** in this pipeline. Keeping them
separate avoids a lot of confusion.

| Level | Question | Where | Unit |
|---|---|---|---|
| **1. Instruction validity** | Is this `source → target` instruction itself a good, supported query? | `stages/validation.py` (binary) **or** the graded gate below | one step / variant |
| **2. Candidate relevance** | For a retained query, which candidate tracks are relevant ground truth? | `stages/relevance_pool.py` (multi-tier) | query × candidate |

Level 1 is an **upstream filter**: it decides which instructions become
benchmark queries at all. Level 2 (`relevance_pool`) then, for each retained
query, grades a candidate pool (≤96 tracks) with cheap heuristics
(tag/caption/constraint overlap, text-encoder solvability audit) and a final
**LLM candidate judge** on the `validation`/`test` splits — that is the
benchmark ground truth. **The graded gate does not touch Level 2.**

---

## The graded instruction-validity gate (`validation_gate.py`)

Historically Level 1 was the binary `validation` stage
(`validated_instructions.jsonl`, `validation.accepted ∈ {true,false}`), which
`relevance_pool` filters on (`only_keep_validation_passed`). The graded pipeline
replaces the *decision source* while keeping the *output schema identical*, so
`relevance_pool` needs no changes.

### Grade first, cut last (two phases)

We **do not filter while grading.** Every variant is rated and every grade is
recorded; a separate selection step does the cutting. This lets us assemble a
valid chain by *choosing a passing variant per step* instead of dropping steps.

1. **grade** (`grade_records` → `instruction_grades.jsonl`): per *variant*,
   record `overall_score` and `variant_accepted = score ≥ threshold`, plus a
   per-step summary (`passing_variants`, `best_variant`, `has_passing_variant`)
   and per-chain coherence info. Nothing is cut.
2. **select** (`select_chain_variants` → `validated_instructions.jsonl`): per
   step choose the **best passing variant** (variant fallback: if variant 0
   fails, take the next that passes); a step is `accepted` iff it has *any*
   passing variant. For the **contextual** track, truncate the chain at the
   first step with *no* passing variant.

Two axes underpin it:
- **Instruction validity** — graded rubric `overall_validity` (1–5, from
  `llm_validation_judge.py` / the human app), thresholded. Same 1–5 scale humans
  use, so the threshold is **human-calibratable**.
- **Chain coherence** — a later step can be standalone-valid yet contextually
  invalid if an earlier step has no passing variant (broken history). Optional:
  a future graded `chain_coherence` rubric question is used if present.

### Config (`GateConfig`)

| Field | Default | Meaning |
|---|---|---|
| `accept_threshold` | `4.0` | `overall_validity ≥ threshold` → variant accepted |
| `variant_select` | `best` | per-step fallback pick: `best` (highest score) or `first` (lowest variant index) |
| `chain_aggregate` | `min` | chain score = `min` (strict) or `mean` of the per-step best scores |
| `contextual_policy` | `truncate` | contextual track: `truncate` at first step w/ no passing variant, or `drop` whole chain |

### Output schemas

`instruction_grades.jsonl` — one record **per variant**, `validation.phase =
"grades"`, with `variant_accepted`, `overall_score`, `question_scores`, `step`
(`passing_variants`, `has_passing_variant`), `chain` (`score`, `truncate_at`,
`within_coherent_prefix`). Nothing removed.

`validated_instructions.jsonl` — one record **per step** (the selected variant),
consumed by `relevance_pool`:

```jsonc
"validation": {
  "accepted": true,                    // relevance_pool reads THIS (step has a passing variant)
  "phase": "selected",
  "selected_variant_index": 1,
  "used_variant_fallback": true,       // variant 0 failed; fell back
  "overall_score": 5,
  "chain": {"score": 2, "within_coherent_prefix": true, "truncate_at": 2},
  // aliases for old binary consumers:
  "history_unaware": {"passed": true,  "reasons": []},
  "history_aware":   {"passed": false, "reasons": ["history_truncated"]}
}
```

The full instruction record is preserved, so `relevance_pool`'s reads of
`semantic_delta_*` / `instruction_plan` are unchanged. The old binary
`validation` stage still works and is untouched.

### How to produce it

CPU-only, from existing graded ratings:
```bash
PYTHONPATH=src python scripts/build_validated_instructions.py \
  --run-root /gpfs/.../music4all_instruct/music4all_v1 \
  --instructions-folder instructions_axis_focused_5 \
  --threshold 4 --variant-select best --contextual-policy truncate
# grade -> instruction_grades.jsonl ; select -> validated_instructions.jsonl
# re-cut only (new policy, no re-grade):  --phase select --variant-select first
```
Or fold both into the judge run: add `--emit-validated` to `llm_validation_judge.py`.

Point `relevance_pool` at it:
`stage.io.input_validation_jsonl=<folder>/validation/validated_instructions.jsonl`.

---

## Rating the judge: human, cross-LLM, and agreement metrics

The graded ratings come from `scripts/llm_validation_judge.py` (the 8-question
rubric in `demo/validation_rubric.py`, shared verbatim with the human app). To
trust the judge at scale we measure agreement in the human-validation app's
Admin tab:

- **Human vs LLM** — per-question agreement between human ratings and a chosen
  LLM (populates once human ratings land).
- **Cross-LLM** — Model A vs Model B (e.g. Qwen vs Gemma). Already run on both
  datasets, both sidecars (frozen human slice + full-validation split).

Each agreement row reports, over items rated by both sides:
`mean_diff` (bias / leniency), `mae`, `within1_rate`, `accept_agree_rate`,
**Cohen's `accept_kappa`**, **Gwet's `accept_ac1`** (prevalence-robust — use it
when almost everything is accepted, where Cohen's κ collapses),
**`quadratic_kappa`** (ordinal, partial credit for near-misses), and
`pearson_r` / `spearman_r`. Report κ *and* AC1 *and* raw agreement together; the
truth is bracketed between the conservative κ and the liberal AC1.

Observed so far (Qwen3.6-27B-FP8 vs Gemma-4-31B-it): `overall_validity`
Pearson ≈ 0.72, AC1 ≈ 0.88, within-1 ≈ 0.95; Gemma is systematically ~0.2–0.4
more lenient (hence per-model threshold calibration).

## Open items

- **Coverage**: `relevance_pool` builds the benchmark on `validation` **and**
  `test`. The graded judge has run on the `validation` split for both datasets;
  to gate the benchmark you must also judge the `test`-split instructions (build
  a `test` or `val+test` full-coverage sidecar and run the judge), otherwise
  unrated test steps fall to `unrated_policy`.
- **Threshold calibration**: set `accept_threshold` (per model) from the
  human-rating study on the shared slice, not by guess.
- **Chain-coherence as an LLM dimension**: currently structural (truncation). A
  graded `chain_coherence` rubric question (fed prior-turn context) can be added
  later; the gate already consumes it if present.
