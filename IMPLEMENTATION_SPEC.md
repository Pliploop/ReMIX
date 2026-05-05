# Jamendo-Instruct Implementation Spec

Prepared: 2026-04-11

Purpose:
- translate the revised architecture into concrete implementation work for the current codebase
- preserve a clear mapping between the conceptual benchmark stages and the repo's implemented stages
- make the constructive relevance-pool design the target end state

## Status

Implemented in this pass:
- Stage 7 / `instructions.py` now prompts for and parses `semantic_delta_full` and `semantic_delta_verbalized`
- Stage 7 instruction records now write both semantic deltas alongside both instruction variants
- `semantic_constraints` is retained as a backward-compatible alias of `semantic_delta_full`
- Stage 7 verifier prompt was updated to a faithfulness-style rubric with structured failure labels
- Stage 8 / `validation.py` now uses the full semantic delta for faithfulness checks and the verbalized semantic delta for genuine-change checks, with fallback logic for older artifacts
- Stage 8 heuristic and LLM-judge prompts now center on genuine change, contradiction, metadata invention, history coherence, and caption-only grounding rather than exhaustive coverage
- Stage 8 config gained explicit faithfulness-oriented checks
- Stage 7 / Stage 8 now derive typed internal semantic-item views with heuristic `source` and `kind` annotations
- Stage 9 / `relevance_pool.py` was moved toward constructive grading:
- exact targets, strong positives, history shortcuts, caption misses, partials, and hard negatives are now explicit pool types
- candidate metadata now includes `pool_type`, `failure_category`, `failed_constraints`, `satisfied_constraints`, `constraint_satisfaction`, and `label_source`
- candidate grading now uses `semantic_delta_verbalized` as the primary requested target and stores `full_constraint_satisfaction` as a diagnostic
- caption-derived typed items now feed explicit `caption_constraint_satisfaction` in the pool metadata
- caption embeddings from the lookup manifest can now contribute to pool semantics and solvability scoring
- the old scalar score is still retained as an internal ranking proxy within pool types
- target-neighborhood sourcing was added so target-relative similarity is available during pool construction
- optional same-split sidecar reason labeling now writes one reason code per clip per step, with non-frontier clips labeled `failed:too_semantically_far_from_target`
- a text-encoder solvability audit now runs inside `relevance_pool` over the final selected pool and surfaces `solvability_audit.flagged` without auto-discarding
- evaluation-split frontier candidates can now be labeled by an LLM judge in `relevance_pool`, with heuristic and embedding signals treated as advisory context

Still to do:
- enrich the prepared Stage 7 payload artifact with post-generation semantic deltas or add a dedicated enriched sidecar artifact
- strengthen contradiction and metadata-invention heuristics beyond the current lightweight checks
- calibrate and harden the evaluation-split LLM candidate judge so it remains stable at pool-construction scale
- add final diagnostics / QC metrics for caption-only rate, history-shortcut coverage, solvability rate, and judge-confidence distributions
- decide whether to keep the current weighted ranking proxy long-term or replace it with a more type-specific ranking policy inside each constructive pool bucket

Current approximation notes:
- `caption_only_change` now excludes deterministic tag, vocal, and speed deltas, which is the intended correction
- typed semantic-item provenance is inferred heuristically from payload evidence rather than produced directly by the LLM
- the current solvability audit uses instruction text embeddings plus lexical overlap over the selected pool, not a full retrieval backbone
- relevance-pool pool typing can now be assigned by an LLM judge for evaluation-split frontier candidates, while typed caption-item matching and caption embeddings remain supporting evidence
- constructive pool construction currently operates over the available local neighborhood graph rather than synthesizing from a broader ANN candidate frontier

## Resume Checkpoint

The semantic-delta migration is now wired end to end across `instructions`, `validation`, and `relevance_pool`:
- `semantic_delta_full`
- `semantic_delta_verbalized`
- full-step validation against the full semantic delta
- candidate grading against the verbalized semantic delta
- optional full same-split candidate-universe labeling with `failed:too_semantically_far_from_target`

What is already landed:
- `src/jamendo_instruct/stages/instructions.py`
  - prompt and parser were updated to expect `semantic_delta_full` and `semantic_delta_verbalized`
  - `semantic_constraints` is intended to remain as a backward-compatible alias of `semantic_delta_full`
  - typed internal semantic-item views are now derived deterministically and written to the instruction artifact
- `src/jamendo_instruct/stages/validation.py`
  - helper logic now handles full versus verbalized semantic deltas
  - the implemented rule is:
  - use `semantic_delta_full` for faithfulness / contradiction checks
  - use `semantic_delta_verbalized` for genuine-change checks
  - typed internal semantic-item views are preserved in heuristic validation output
- `src/jamendo_instruct/stages/relevance_pool.py`
  - helper functions now load full versus verbalized semantic deltas
  - candidate-term matching allows caption-text matches, not just tag / vocals / speed matches
  - constructive pool outputs now include both semantic deltas
  - optional full-dataset labeling support writes the sidecar artifact when enabled
  - solvability audit now runs against the final pool with a lightweight lexical baseline

First commands to run on resume:

```bash
python -m py_compile \
  src/jamendo_instruct/stages/instructions.py \
  src/jamendo_instruct/stages/validation.py \
  src/jamendo_instruct/stages/relevance_pool.py
```

```bash
rg -n "semantic_constraints|semantic_delta_full|semantic_delta_verbalized|too_semantically_far_from_target" \
  src/jamendo_instruct/stages/instructions.py \
  src/jamendo_instruct/stages/validation.py \
  src/jamendo_instruct/stages/relevance_pool.py \
  README.md IMPLEMENTATION_SPEC.md PLAN.md
```

Exact next implementation steps:

1. Strengthen evaluation-mode caption semantics further.
- calibrate the new caption-embedding thresholds on a real validation slice
- expand targeted judge checks for history-shortcut caption preservation, not only generic caption misses

2. Harden reporting and QC.
- aggregate reason-code histograms from the full-dataset sidecar
- add Stage 14-style diagnostics for caption-only rate, history-shortcut coverage, and solvability rate

3. Re-run focused smoke tests after each follow-up change.
- `py_compile`
- a small helper smoke test for semantic-delta normalization
- if artifacts are available, a tiny end-to-end run over a few chains

Design intent to preserve when resuming:
- deterministic delta = source-of-truth structural transition
- `semantic_delta_full` = exhaustive semantic interpretation of the step
- `semantic_delta_verbalized` = requested subset actually verbalized by both instruction variants
- validation judges instruction faithfulness against the full step semantics
- retrieval candidate grading judges candidate satisfaction against the verbalized request
- auxiliary diagnostics can still compare candidates against the full step semantics

## Repo Mapping

Conceptual benchmark stages versus current repo modules:
- conceptual Stage 11 instruction generation -> `src/jamendo_instruct/stages/instructions.py`
- conceptual Stage 12 validation -> `src/jamendo_instruct/stages/validation.py`
- conceptual Stages 9 and 10 candidate construction / grading -> `src/jamendo_instruct/stages/relevance_pool.py`
- conceptual Stage 14 diagnostics -> reports emitted by `validation` and `relevance_pool`, plus later QC additions

Important repo reality:
- the current executable pipeline is `chains -> instructions -> validation -> relevance_pool`
- the internal verifier inside `instructions.py` is optional and disabled by default
- the real benchmark gate today is `validation.py`

## Non-Negotiable Design Decisions

1. `semantic_constraints` must be added to Stage 7 / conceptual Stage 11 output.
- extraction is exhaustive
- instruction verbalization is selective
- validation should check faithfulness, not exhaustive coverage

2. Caption-level semantic changes are first-class.
- caption-only turns must not collapse into tag-only language
- deterministic tags alone are insufficient for grading these turns

3. Semantic extraction and instruction generation stay in the same LLM call.
- no new standalone Stage 6.5 LLM pass
- `_build_step_payload` remains deterministic and non-LLM

4. The relevance pool should be fully adopted to the constructive design.
- benchmark-facing candidates should be instantiated by pool type
- arbitrary pool assembly followed by opaque grading is not the target design
- simple frontier filtering for obvious negatives is allowed, but retained evaluation candidates should otherwise be graded constructively

5. Validation moves from coverage to faithfulness.
- at least one genuine change
- no contradiction
- no invented metadata
- caption-derived wording for caption-only turns

## Recommended Public Schema

Per-turn Stage 7 / `instructions` output should become:

```json
{
  "chain_id": "...",
  "turn_index": 2,
  "seed_clip_id": "...",
  "source_clip_id": "...",
  "target_clip_id": "...",
  "verbosity": "short",
  "semantic_constraints": {
    "preserved": ["haunting piano texture", "sparse reverb character"],
    "new": ["fast arpeggios", "optimistic triumphant register"],
    "lost": ["melancholic introspective mood"],
    "primary_edit": "emotional register shifts from melancholy to triumph while texture is preserved",
    "caption_only_change": true
  },
  "history_unaware_instruction": "...",
  "history_aware_instruction": "..."
}
```

Implementation note:
- the export contract above is stable and simple
- internally, we should preserve typed provenance for each constraint item so later stages can tell tag-derived items from caption-derived items

Suggested internal representation:

```json
{
  "preserved_items": [
    {"text": "piano", "source": "tag", "kind": "instrument"},
    {"text": "haunting texture", "source": "caption", "kind": "texture"}
  ]
}
```

This internal enrichment can stay private to the implementation if we want the public JSONL to remain compact.

## Schema Adjustments to the Original Spec

These are deliberate design corrections for implementability:

1. `caption_only_change` should not mean only `tags_added == []` and `tags_removed == []`.
- deterministic `vocals` and `speed` changes also explain a turn
- the safer definition is:
- no deterministic tag / vocal / speed delta explains the turn
- captions still differ semantically

2. Short instructions should verbalize one salient genuine change, not necessarily one item from `new`.
- some turns are mostly removals or contrastive edits
- allow the salient change to come from `new` or `lost`

3. The Stage 7 internal verifier is not the main acceptance gate.
- if we change the rubric, the primary implementation target is `validation.py`
- the optional verifier in `instructions.py` should be kept aligned, but it is secondary

## Implementation Order

### Phase 1: Stage 7 / `instructions.py`

Goal:
- introduce `semantic_constraints`
- shift prompting from exhaustive verbalization to exhaustive extraction plus selective verbalization
- keep current artifacts readable by downstream stages

Required changes:
- update `_combined_generation_prompt`
- require JSON keys in this order:
- `semantic_constraints`
- `history_unaware_instruction`
- `history_aware_instruction`
- parse and validate `semantic_constraints` in `_generate_instruction_pair`
- write `semantic_constraints` into `chain_step_instructions.jsonl`
- keep all existing fields so the current artifact shape remains additive rather than breaking

Prompt rules to add:
- extract all semantic constraints exhaustively
- short verbalizes one salient genuine change
- medium verbalizes one change plus one preservation cue
- long verbalizes two salient changes or one change with richer context
- instructions must be faithful to `semantic_constraints`
- instructions do not need to cover every item
- caption-only turns must use caption-derived content

Parser expectations:
- `semantic_constraints.preserved` is a list
- `semantic_constraints.new` is a list
- `semantic_constraints.lost` is a list
- `semantic_constraints.primary_edit` is a non-empty string
- `semantic_constraints.caption_only_change` is a boolean

Backward-compatibility rule:
- do not remove any current keys from `chain_step_instruction_inputs.jsonl` or `chain_step_instructions.jsonl`
- add fields only

### Phase 2: Stage 8 / `validation.py`

Goal:
- make validation compatible with selective verbalization

Required changes:
- remove heuristic and LLM-judge assumptions about exhaustive semantic coverage
- replace with faithfulness-oriented checks
- introduce structured failure codes

Heuristic checks should move toward:
- `failed:no_genuine_change`
- `failed:contradiction`
- `failed:metadata_invention`
- `failed:requires_history`
- `failed:history_incoherent`
- `failed:caption_only_verbalization_missing`
- `failed:format_error`

Implementation detail:
- heuristic prechecks can still remain cheap
- they should no longer count raw constraint-term omission as automatic failure
- term matching should become advisory evidence, not the acceptance definition

LLM judge prompt should evaluate:
- at least one genuine change is expressed
- no contradiction of preserved constraints
- no unsupported metadata invention
- history-unaware solvability
- history-aware coherence
- caption-only verbalization quality where relevant

Reporting changes:
- log counts by failure code
- keep variant-specific counts
- keep accepted counts by verbosity

### Phase 3: Stage 9 and Stage 10 / `relevance_pool.py`

Goal:
- replace weighted attribution-style grading with constructive pool assembly and explicit constraint vectors

This is a true rewrite, not a tune-up.

Required pool types:
- `Type_TARGET`
- `Type_STRONG`
- `Type_H`
- `Type_T`
- `Type_PARTIAL`
- `Type_HARD_NEG`

Allowed pre-filters:
- obvious hard negatives may be sourced or filtered using high audio similarity to the seed
- obvious off-manifold negatives may be filtered when semantic delta is clearly too large

But:
- the retained benchmark-facing pool should be labeled by construction rather than mainly by weighted score thresholds

Required candidate metadata:
- `clip_id`
- `grade`
- `pool_type`
- `failure_category`
- `failed_constraints`
- `satisfied_constraints`
- `audio_sim_to_target`
- `caption_sim_to_target`
- `history_shortcut`
- `label_source`
- `constraint_satisfaction`

Required constraint satisfaction vector:
- `satisfies_new_edit`
- `preserves_accumulated_tags`
- `preserves_accumulated_caption_constraints`
- `matches_target_caption_semantics`
- `satisfies_vocal_status`
- `satisfies_speed_constraint`
- `history_shortcut_detected`

Implementation modes:
- training mode:
- deterministic caption thresholds
- deterministic reasons
- test mode:
- targeted LLM binary checks for ambiguous caption-semantic cases
- optional human-calibrated label confirmation later

Migration strategy:
- easiest path is to add a new constructive code path under a config flag first
- once validated, make constructive mode the default
- preserve the current scorer only as a temporary fallback during transition

### Phase 4: Solvability Audit

Goal:
- detect turns labeled history-aware that are actually solvable from the latest instruction alone

Important dependency:
- this depends on having a usable candidate pool
- therefore it should be implemented after or alongside the constructive relevance-pool rewrite

Output behavior:
- add `solvability_flag` to validation or downstream review artifacts
- do not auto-discard on this flag alone

### Phase 5: Diagnostics

Add reportable metrics:
- `caption_only_change_rate`
- `history_shortcut_coverage`
- `single_turn_solvability_rate`
- `constraint_type_distribution`
- `llm_judge_confidence_distribution`

## Concrete File Targets

Files to change first:
- `src/jamendo_instruct/stages/instructions.py`
- `src/jamendo_instruct/stages/validation.py`
- `conf/stage/instructions.yaml`
- `conf/stage/validation.yaml`

Files for the constructive pool rewrite:
- `src/jamendo_instruct/stages/relevance_pool.py`
- `conf/stage/relevance_pool.yaml`

Files to keep synchronized:
- `PLAN.md`
- `README.md`

## Success Criteria

After Phase 1 and 2:
- every instruction record contains `semantic_constraints`
- caption-only turns appear in the data when present in the corpus
- short instructions remain short
- validation pass rate does not collapse because of the new rubric

After Phase 3:
- each pool contains candidates from the intended constructive types
- history-shortcut negatives are explicitly represented rather than only inferred from scores
- tag-hit / caption-miss cases appear in caption-heavy turns
- candidate labels are inspectable from metadata without reverse-engineering a scalar score

## What Not to Change

- do not add LLM calls inside `_build_step_payload`
- do not remove the verbosity-bucket system
- do not remove paired `history_unaware` and `history_aware` generation
- do not modify Stages 1 to 6 unless a later implementation requirement forces it
