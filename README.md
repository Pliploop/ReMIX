# Jamendo-Instruct

Hydra/OmegaConf stage-wise pipeline for building a multi-turn compositional music retrieval dataset.

## Current Status

Implemented: **Stage 1 (`ingest`)**, **Stage 2 (`caption_join`, optional)**, **Stage 2.5 (`lyrics`)**, **Stage 3 (`structured_view`)**, **Optional Stage 3.5 (`embeddings`)**, **Stage 4 (`neighborhood`)**, **Stage 5 (`graph`)**, **Stage 6 (`chains`)**, **Stage 7 (`instructions`)**, **Stage 8 (`validation`)**, and an initial **Stage 9 (`relevance_pool`)**

The current pipeline is a clip-level dataset builder for multi-turn music retrieval:

- `ingest` reads JamendoMaxCaps metadata shards plus `final_caption30sec.jsonl`, deduplicates shard snapshots on `track_id`, emits one row per captioned clip, normalizes tags/captions, assigns splits, and writes a canonical manifest.
- `caption_join` optionally attaches external rewritten `§§`-separated captions by `track_id` without changing the rest of the artifact contract.
- `lyrics` can transcribe clip-level lyrics with `openai/whisper-large-v3-turbo`, appending lyric text plus transcription metadata while defaulting missing / non-lyrical clips to empty lyric strings.
- `structured_view` creates a lightweight retrieval-facing manifest from normalized captions and tags, without introducing heavier ontology or LLM parsing yet.
- `embeddings` computes or resumes per-clip audio and text embeddings, writing `.npy` vectors plus a lookup manifest keyed by `clip_id`.
- `neighborhood` builds same-split local candidate neighborhoods from audio nearest neighbors and reranks them with a configurable audio/text cosine blend.
- `graph` converts retained neighborhood edges into directed transition candidates with explicit structured deltas, hardness labels, and a scalar `transition_score`.
- `chains` samples short-heavy stochastic walks over the transition graph while banning revisits of prior `clip_id` and `track_id`, and stores accumulated intent state for downstream instruction generation.
- `instructions` consumes full chains, builds deterministic per-step prompt payloads with fuzzy/raw caption-difference signals, and now emits both paired instructions plus `semantic_delta_full`, `semantic_delta_verbalized`, and a backward-compatible `semantic_constraints` alias.
- `validation` runs lightweight heuristic prechecks plus an LLM judge over each instruction pair and now validates faithfulness against the full semantic delta while checking genuine requested change against the verbalized semantic delta.
- `relevance_pool` now uses a constructive pool design where benchmark-facing candidates are instantiated by failure type, with history-shortcut negatives, caption misses, partial matches, and optional whole-split reason labeling by construction.
- `jamendo-pipeline` orchestrates these stages for a shared `run_name` and can resume from the earliest missing output.

The current Jamendo mapping already preserves useful metadata for future enrichment work:
- `artist_id` and `artist_name`
- `release_date`
- `vocals` and `speed`
- genre / instrument / vartag groups from `musicinfo.tags`

## Chain Demo Scaffold

An initial Gradio chain explorer is now scaffolded for browsing sampled chains without any annotation workflow yet.

Install the demo dependency:
- `pip install -e .[demo]`

Launch from a run directory:
- `jamendo-chains-demo --run-root /path/to/<run_name> --max-chains 250`

You can also point it at explicit artifacts:
- `jamendo-chains-demo --manifest-csv .../structured_clip_manifest.csv --chains-jsonl .../sampled_chains.jsonl --instructions-jsonl .../chain_step_instructions.jsonl`

Current focus:
- chain-level navigation plus per-step navigation
- first-class display of `history_unaware_instruction`, `history_aware_instruction`, `semantic_delta_full`, `semantic_delta_verbalized`, and the raw `structured_delta`
- playable audio for seed / source / target clips, with a best-effort sliced clip preview when `soundfile` is available

Intentional non-goals for this scaffold:
- no annotation UI
- no persistence or validation-writing flow yet
- no attempt to load an entire million-chain artifact into memory by default

Planned enrichment direction:
- derive coarse era buckets from `release_date`
- use artist metadata mainly for split control, diagnostics, and optional benchmark subsets rather than default instructions
- treat lyric information as optional sidecar enrichment with explicit provenance, instead of making scraping or ASR part of the core v1 pipeline

## Config Layout

- `conf/config.yaml` root composition
- `conf/dataset/jamendomaxcaps.yaml` dataset field mapping
- `conf/runtime/default.yaml` runtime/output settings
- `conf/stage/ingest.yaml` Stage 1 parameters
- `conf/stage/caption_join.yaml` Stage 2 parameters
- `conf/stage/lyrics.yaml` Stage 2.5 parameters
- `conf/stage/structured_view.yaml` Stage 3 parameters
- `conf/stage/embeddings.yaml` Optional Stage 3.5 parameters
- `conf/stage/neighborhood.yaml` Stage 4 parameters
- `conf/stage/graph.yaml` Stage 5 parameters
- `conf/stage/chains.yaml` Stage 6 parameters
- `conf/stage/instructions.yaml` Stage 7 parameters
- `conf/stage/validation.yaml` Stage 8 parameters
- `conf/stage/relevance_pool.yaml` Stage 9 parameters
- `conf/pipeline.yaml` global pipeline parameters

## Implemented Stages

### Stage 1: `ingest`

Purpose:
- Build the canonical clip-level manifest for a run.

Current behavior:
- Reads date-sharded JamendoMaxCaps metadata JSONL files from `stage.tracks.metadata_dir`.
- Optionally downloads missing metadata shards and the captions file from Hugging Face.
- Excludes `final_caption30sec.jsonl` from the shard scan and loads it separately as the caption source.
- Deduplicates repeated shard snapshots on `track_id`.
- Splits multi-caption entries into one row per captioned clip.
- Preserves parent `track_id` while assigning clip-level rows with `clip_id`, `start_time`, and `end_time` when available.
- Normalizes tags with lowercasing, punctuation stripping, and optional deduplication.
- Normalizes captions with optional deduplication and stores a primary caption according to config.
- Synthesizes `file_path` from the configured audio root and file template.
- Assigns dataset splits either from a grouped custom split (`artist_id` by default) or a source column.
- Optionally writes `dropped_rows.csv` for rows filtered out by caption/tag/audio availability rules.

Default outputs:
- `artifacts/<run_name>/ingest/normalized_track_manifest.csv`
- `artifacts/<run_name>/ingest/ingest_report.json`
- `artifacts/<run_name>/ingest/dropped_rows.csv` when enabled and non-empty

Potential upgrades kept in scope:
- richer tag synonym mapping
- optional audio existence verification at scale
- additional diagnostics around deduplication and split balance

### Stage 2: `caption_join` (optional)

Purpose:
- Attach externally rewritten captions to the ingest manifest without changing the rest of the pipeline.

Current behavior:
- Reads `normalized_track_manifest.csv` from Stage 1.
- Optionally loads a rewrites CSV keyed by `track_id`.
- Splits rewrite strings on `§§`, trims parts, drops empties, and derives a rewritten primary caption.
- Can either preserve original caption fields and append rewrite metadata, or overwrite caption fields if explicitly enabled.
- Falls back to pass-through behavior when no rewrites file is supplied and `require_rewrites_file=false`.

Default outputs:
- `artifacts/<run_name>/caption_join/normalized_track_manifest_with_rewrites.csv`
- `artifacts/<run_name>/caption_join/caption_join_report.json`

Potential upgrades kept in scope:
- clip-level rewrite alignment instead of track-level attachment
- stricter validation of rewrite coverage and formatting
- multiple rewrite variants per track

### Stage 3: `structured_view`

Purpose:
- Produce the retrieval-facing structured manifest used by downstream embedding and graph stages.

Current behavior:
- Reads the Stage 1 ingest manifest by default.
- Can also consume the Stage 2 output via pipeline wiring when caption rewrites are enabled.
- Keeps this stage intentionally lightweight: it normalizes caption text, preserves normalized tags, and carries forward useful clip/track metadata needed later.
- Avoids ontology-heavy parsing, LLM extraction, or confidence scoring in v1.

Default outputs:
- `artifacts/<run_name>/structured_view/structured_clip_manifest.csv`
- `artifacts/<run_name>/structured_view/structured_view_report.json`

Potential upgrades kept in scope:
- richer structured attribute extraction
- optional parser-assisted caption decomposition
- explicit provenance fields for derived attributes
- optional enrichment columns for artist / era / lyric signals when source quality is clear

### Stage 3.5: `embeddings` (optional but required for current Stage 4+)

Purpose:
- Materialize reusable clip-level embedding artifacts for retrieval and graph construction.

Current behavior:
- Reads `structured_clip_manifest.csv`.
- Computes per-clip audio embeddings with `OpenMuQ/MuQ-MuLan-large`.
- Computes per-clip text embeddings with `google/embeddinggemma-300m` over `normalized_caption`.
- Slices audio by `start_time` / `end_time` when clip bounds are present, otherwise falls back to the full track.
- Stores one `.npy` file per clip for both audio and text.
- Writes `embedding_lookup_manifest.csv` as the source of truth linking each `clip_id` to embedding paths.
- Supports resume semantics by trusting `embedding_lookup_manifest.csv` by default, avoiding full per-file existence scans on restart unless verification is enabled.
- Supports a `manifest_only` mode to rebuild the lookup manifest without recomputing vectors.

Default outputs:
- `${runtime.output_root}/audio/${runtime.run_name}/` with one audio `.npy` per clip
- `${runtime.output_root}/text/${runtime.run_name}/` with one text `.npy` per clip
- `artifacts/<run_name>/embeddings/embedding_lookup_manifest.csv`
- `artifacts/<run_name>/embeddings/embeddings_report.json`

Potential upgrades kept in scope:
- ANN-friendly export layouts
- pooled or track-level embedding variants
- alternate text fields or prompt formatting for text embedding

### Stage 4: `neighborhood`

Purpose:
- Build a local retrieval neighborhood around each seed clip for transition mining.

Current behavior:
- Reads the Stage 3 structured manifest plus the audio/text embedding directories by default; the embedding lookup manifest remains available as a fallback for older layouts.
- Uses audio cosine similarity for first-stage retrieval.
- Retrieves `audio_top_k` nearest audio neighbors per seed and reranks the retained neighborhood with a weighted audio/text cosine blend.
- Works at the clip level while preserving `track_id` and split metadata.
- Excludes self by default.
- Excludes same-parent `track_id` neighbors by default to avoid trivial within-track transitions.
- Enforces same-split retrieval by default.
- Can filter out candidates whose changed-tag count already exceeds the configured pre-rerank threshold.
- Writes a node table plus an edge table so later stages can join back to the full structured metadata.

Default outputs:
- `artifacts/<run_name>/neighborhood/neighborhood_nodes.csv`
- `artifacts/<run_name>/neighborhood/neighborhood_edges.csv`
- `artifacts/<run_name>/neighborhood/neighborhood_report.json`

Potential upgrades kept in scope:
- ANN retrieval backend for larger corpora
- duplicate-file-path filtering
- richer hybrid retrieval signals beyond audio/text cosine

### Stage 5: `graph`

Purpose:
- Convert local neighborhoods into a directed transition graph of plausible refinements.

Current behavior:
- Reads `structured_clip_manifest.csv`, `neighborhood_nodes.csv`, and `neighborhood_edges.csv`.
- Joins node ids back to clip metadata from the structured manifest.
- Defensively drops any cross-split edge.
- Retains only edges whose changed-tag count is within `filters.max_changed_tags`.
- Computes a structured delta for each retained edge, including tag-set changes and optional caption text context.
- Assigns a scalar `transition_score` from weighted audio similarity, text similarity, and tag-delta ease.
- Stores `transition_cost = 1 - transition_score` as a secondary diagnostic.
- Labels a coarse hardness bucket and writes a compact transition-edge artifact for downstream sampling.
- Uses chunked intermediate CSV/JSON files during graph construction, then removes the temporary `graph_chunks/` directory after a successful merge by default.
- Resume behavior for this stage now keys off the final `transition_graph_edges.csv` artifact rather than temporary chunk files.

Default outputs:
- `artifacts/<run_name>/graph/transition_graph_edges.csv`
- `artifacts/<run_name>/graph/graph_report.json`

Potential upgrades kept in scope:
- richer preservation and semantic-consistency features
- more expressive hardness scoring
- optional chunked or distributed graph scoring backends

### Stage 6: `chains`

Purpose:
- Sample multi-step retrieval trajectories from the transition graph.

Current behavior:
- Reads `structured_clip_manifest.csv`, `neighborhood_nodes.csv`, and `transition_graph_edges.csv`.
- Samples target chain lengths from a short-heavy log-normal distribution, then clips them to the configured min/max range.
- Chooses seed nodes from graph nodes with at least one outgoing transition.
- Walks stochastically over outgoing edges using normalized `transition_score` values.
- Bans revisiting any prior `clip_id` or `track_id`.
- Keeps shorter chains when a dead end is reached if they still satisfy the minimum chain length.
- Writes per-chain JSONL records with seed metadata, ordered steps, structured deltas, transition scores, and accumulated intent state after each turn.
- Records summary statistics such as attempts, realized chain lengths, and dead-end behavior in the stage report.

Default outputs:
- `artifacts/<run_name>/chains/sampled_chains.jsonl`
- `artifacts/<run_name>/chains/chains_report.json`

Potential upgrades kept in scope:
- alternative chain-length priors
- seed sampling strategies that improve coverage over rare edits
- diversity-aware chain sampling and deduplication

### Stage 7: `instructions`

Purpose:
- Generate paired relative user instructions for each chain step.

Current behavior:
- Reads `structured_clip_manifest.csv` and `sampled_chains.jsonl`.
- Builds a deterministic per-step intermediate object with seed, previous, and target views.
- Computes seed-relative and previous-relative deltas plus fuzzy and/or raw caption-difference signals depending on config.
- Samples an instruction verbosity bucket per step: `short`, `medium`, or `long`.
- Generates both `history_unaware` and `history_aware` instructions in one Hugging Face model call using `google/gemma-4-31B-it` by default.
- The same LLM call now first extracts `semantic_delta_full`, then `semantic_delta_verbalized`, and only then writes the paired instruction variants.
- `semantic_delta_full` is the exhaustive semantic interpretation of the step.
- `semantic_delta_verbalized` is the subset the instruction actually commits to requesting.
- `semantic_constraints` is retained as a compact backward-compatible alias of `semantic_delta_full`.
- the stage also derives typed internal forms, `semantic_delta_full_typed` and `semantic_delta_verbalized_typed`, with heuristic `source` and `kind` annotations for downstream grading
- The intended semantics are now: exhaustive extraction, partial verbalization, faithfulness-only validation.
- The generation prompt now explicitly biases toward short, direct, colloquial requests and asks the model to paraphrase caption wording instead of echoing it verbatim whenever possible.
- Enforces strict JSON parsing and can optionally run a verifier pass that discards non-compliant generations.
- Can also write the prepared intermediate-object JSONL for smoke tests and debugging.
- Records timing information in the stage report, including model load time, total generation time, total verification time, and mean generation time per attempted step.

Default outputs:
- `artifacts/<run_name>/instructions/chain_step_instructions.jsonl`
- `artifacts/<run_name>/instructions/chain_step_instruction_inputs.jsonl` when enabled
- `artifacts/<run_name>/instructions/instructions_report.json`

Potential upgrades kept in scope:
- stronger caption-difference heuristics
- multi-sample generation with later filtering
- dedicated rewrite / paraphrase stage
- instruction-aware verifier and retrieval-side validation

### Stage 8: `validation`

Purpose:
- Validate generated instruction pairs before they are used for benchmarking.

Current behavior:
- Reads `chain_step_instructions.jsonl` and the deterministic prepared-input JSONL from Stage 7.
- Runs cheap heuristic prechecks for relative language, caption-signal usage, and constraint-signal usage.
- Uses `google/gemma-4-31B-it` as an LLM judge by default to make the final acceptance decision.
- Produces machine-readable pass/fail annotations for `history_unaware` and `history_aware` separately.
- Records both heuristic precheck results and final judge results in the validation artifact.
- The current judge artifact reports criterion-level checks such as relative phrasing, genuine requested change, contradiction, caption grounding, unsupported invention, and the variant-specific solvability or history-coherence requirement.
- Validation also preserves the typed semantic-delta views actually used during heuristic checking.
- By default, the stage now calls the LLM judge even when heuristic prechecks fail, so benchmark decisions are driven by the judge rather than by the cheap filter.
- The planned rubric change is to stop requiring exhaustive semantic coverage in the instruction text itself and instead require:
- at least one genuine change
- no contradiction of preserved constraints
- no metadata invention
- caption-derived wording for caption-only turns
- a lightweight solvability audit for suspected fake history dependence

Default outputs:
- `artifacts/<run_name>/validation/validated_instructions.jsonl`
- `artifacts/<run_name>/validation/validation_report.json`

Potential upgrades kept in scope:
- richer faithfulness-oriented LLM judge rubrics
- agreement checks across multiple judges or prompts
- retrieval-grounded validation beyond text-only judging

### Stage 9: `relevance_pool`

Purpose:
- Build per-turn graded candidate pools for non-binary evaluation.

Current behavior:
- Reads the structured manifest, neighborhood node/edge tables, chain records, prepared instruction inputs, and optional validation outputs.
- The current implementation uses a constructive pool builder over a heuristic neighborhood frontier.
- The benchmark-facing design is fully constructive for retained candidates rather than attribution-based.
- The pool should be built by type:
- `Type_TARGET`
- `Type_STRONG`
- `Type_H` history-shortcut negatives
- `Type_T` tag-hit / caption-miss candidates
- `Type_PARTIAL`
- `Type_HARD_NEG`
- Straight negatives may still be filtered with high audio similarity or obviously excessive semantic distance, but the retained evaluation pool should otherwise be graded by construction.
- Each candidate should carry explicit pool metadata such as `pool_type`, `failure_category`, `failed_constraints`, `satisfied_constraints`, `history_shortcut`, and `label_source`.
- Candidate grading now also stores typed caption-constraint satisfaction and caption text-embedding diagnostics, but frontier candidate failure-mode labels can now be set by an LLM judge rather than by similarity alone.
- In optional evaluation labeling mode, every same-split clip also receives a reason code:
- constructive frontier clips keep their constructed success or failure reason
- non-frontier clips receive `failed:too_semantically_far_from_target`
- The stage now also runs a text-encoder-based solvability audit over the final pool using the `history_unaware_instruction`; flagged turns are surfaced for review rather than auto-discarded.
- Evaluation-split frontier candidates can now be labeled by an LLM judge, with semantic similarity features treated as advisory context rather than the final failure-mode source.

Default outputs:
- `artifacts/<run_name>/relevance_pool/chain_step_relevance_pools.jsonl`
- `artifacts/<run_name>/relevance_pool/relevance_pool_report.json`

Potential upgrades kept in scope:
- fully constructive candidate synthesis and grading
- larger candidate sources beyond the local neighborhood graph
- instruction-aware or retrieval-aware pool refinement

## Stage Commands

Base command:

```bash
PYTHONPATH=src python src/jamendo_instruct/run.py stage=<stage_name> runtime.run_name=<run_name>
```

### `ingest`

```bash
PYTHONPATH=src python src/jamendo_instruct/run.py \
  stage=ingest \
  runtime.run_name=<run_name>
```

Useful options:
- `stage.tracks.metadata_dir=/path/to/metadata`
- `stage.download.enabled=true|false`
- `stage.download.max_files=<N>` for smoke runs
- `stage.audio.verify_exists=true|false`
- `stage.split.mode=custom_track_grouped|source_column`
- `stage.filters.drop_missing_caption=true|false`

### `caption_join`

```bash
PYTHONPATH=src python src/jamendo_instruct/run.py \
  stage=caption_join \
  runtime.run_name=<run_name> \
  stage.rewrites.csv_path=/path/to/rewrites.csv
```

Useful options:
- `stage.rewrites.enabled=true|false`
- `stage.rewrites.primary_caption_strategy=longest|first`
- `stage.behavior.require_rewrites_file=true|false`
- `stage.behavior.overwrite_caption_fields=true|false`

### `structured_view`

```bash
PYTHONPATH=src python src/jamendo_instruct/run.py \
  stage=structured_view \
  runtime.run_name=<run_name>
```

Useful options:
- `stage.io.input_manifest_csv=/path/to/normalized_track_manifest.csv`
- `stage.caption_normalization.lowercase=true|false`
- `stage.caption_normalization.collapse_whitespace=true|false`
- `stage.tags.top_n=<N>` to keep only the top-N normalized tags by corpus frequency for downstream deltas

### `embeddings`

```bash
PYTHONPATH=src python src/jamendo_instruct/run.py \
  stage=embeddings \
  runtime.run_name=<run_name>
```

Useful options:
- `stage.behavior.manifest_only=true|false`
- `stage.behavior.overwrite_existing=true|false`
- `stage.behavior.verify_existing_files=true|false`
- `stage.behavior.max_rows=<N>`
- `stage.models.audio_model_id=<hf_model>`
- `stage.models.text_model_id=<hf_model>`
- `stage.runtime.device=auto|cuda|cpu`

### `neighborhood`

```bash
PYTHONPATH=src python src/jamendo_instruct/run.py \
  stage=neighborhood \
  runtime.run_name=<run_name>
```

Useful options:
- `stage.retrieval.audio_backend=auto|brute_force|faiss`
- `stage.retrieval.audio_top_k=<N>`
- `stage.retrieval.retain_top_k=<N>`
- `stage.retrieval.use_text_rerank=true|false`
- `stage.behavior.exclude_same_track=true|false`
- `stage.behavior.enforce_same_split=true|false`
- `stage.behavior.max_rows=<N>`

### `graph`

```bash
PYTHONPATH=src python src/jamendo_instruct/run.py \
  stage=graph \
  runtime.run_name=<run_name>
```

Useful options:
- `stage.filters.max_changed_tags=<N>`
- `stage.scoring.audio_weight=<float>`
- `stage.scoring.text_weight=<float>`
- `stage.scoring.tag_weight=<float>`
- `stage.output.include_structured_delta_json=true|false`

### `chains`

```bash
PYTHONPATH=src python src/jamendo_instruct/run.py \
  stage=chains \
  runtime.run_name=<run_name>
```

Useful options:
- `stage.behavior.target_num_chains=<N>`
- `stage.behavior.min_chain_length=<N>`
- `stage.behavior.max_chain_length=<N>`
- `stage.behavior.max_chain_attempts=<N>`
- `stage.behavior.keep_shorter_on_dead_end=true|false`

### `instructions`

```bash
PYTHONPATH=src python src/jamendo_instruct/run.py \
  stage=instructions \
  runtime.run_name=<run_name>
```

Useful options:
- `stage.behavior.max_chains=<N>`
- `stage.behavior.max_steps=<N>`
- `stage.caption.signal_mode=fuzzy|raw|both`
- `stage.behavior.write_prepared_records=true|false`
- `stage.verification.enabled=true|false`
- `stage.models.model_id=<hf_model>`
- `stage.models.params_b=<B>` to help `stage.runtime.backend=auto` estimate model memory
- `stage.runtime.device=auto|cuda|cpu`
- `stage.runtime.backend=auto|vllm|vllm_local|sglang_local|transformers|transformers_bnb`
- Auto backend policy:
  - direct `vllm` BF16 when the model appears to fit on one visible GPU
  - direct `vllm` tensor parallelism when BF16 fits across visible GPUs
  - direct `vllm` FP8 plus FP8 KV cache on H100/H200/B200-style allocations when BF16 is tight
  - `transformers_bnb` NF4 as the low-memory A100/compatibility fallback
  - `sglang_local` only when explicitly selected, or when `stage.runtime.auto_allow_sglang=true` and auto selects it for Qwen/MoE-style experiments
- Direct offline vLLM backend:
  `stage.runtime.backend=vllm stage.runtime.vllm_tensor_parallel_size=0 stage.runtime.vllm_dtype=bfloat16`
- Transformers NF4 fallback:
  `stage.runtime.backend=transformers_bnb stage.runtime.torch_dtype=bfloat16`
- SGLang OpenAI-compatible endpoint:
  `stage.runtime.backend=sglang_local stage.runtime.sglang_host=<host> stage.runtime.sglang_port=<port>`
- Qwen3.6 normal Transformers backend:
  `stage.models.model_id=Qwen/Qwen3.6-35B-A3B stage.runtime.llm_model_family=qwen3_6`
- Qwen3.6 local vLLM backend:
  `stage.models.model_id=Qwen/Qwen3.6-35B-A3B stage.runtime.backend=vllm_local stage.runtime.vllm_dtype=bfloat16`
- Qwen3.6 FP8 local vLLM backend:
  `stage.models.model_id=Qwen/Qwen3.6-35B-A3B-FP8 stage.runtime.backend=vllm_local stage.runtime.vllm_dtype=auto`
  The `vllm_local` mode launches a local vLLM server for the instruction stage; downstream judge stages can also use an OpenAI-compatible vLLM endpoint via `stage.runtime.backend=vllm_local`. Prefer direct `stage.runtime.backend=vllm` for single-process Slurm preprocessing.

### `validation`

```bash
PYTHONPATH=src python src/jamendo_instruct/run.py \
  stage=validation \
  runtime.run_name=<run_name>
```

Useful options:
- `stage.behavior.max_steps=<N>`
- `stage.behavior.use_llm_judge=true|false`
- `stage.behavior.skip_llm_on_precheck_failure=true|false`
- `stage.behavior.discard_failed=true|false`
- `stage.checks.require_caption_only_grounding=true|false`
- `stage.checks.require_seed_solvable=true|false`
- `stage.models.model_id=Qwen/Qwen3.6-35B-A3B stage.runtime.llm_model_family=qwen3_6`
- `stage.models.model_id=Qwen/Qwen3.6-35B-A3B-FP8 stage.runtime.backend=vllm_local stage.runtime.vllm_host=<host> stage.runtime.vllm_port=<port>`

### `relevance_pool`

```bash
PYTHONPATH=src python src/jamendo_instruct/run.py \
  stage=relevance_pool \
  runtime.run_name=<run_name>
```

Useful options:
- `stage.behavior.require_validated_instructions=true|false`
- `stage.behavior.only_keep_validation_passed=true|false`
- `stage.behavior.write_full_dataset_labels=true|false`
- `stage.behavior.full_dataset_label_splits=[validation,test]`
- `stage.behavior.run_solvability_audit=true|false`
- `stage.models.judge_model_id=Qwen/Qwen3.6-35B-A3B stage.runtime.llm_model_family=qwen3_6`
- `stage.models.judge_model_id=Qwen/Qwen3.6-35B-A3B-FP8 stage.runtime.backend=vllm_local stage.runtime.vllm_host=<host> stage.runtime.vllm_port=<port>`
- `stage.behavior.use_text_encoder_audit=true|false`
- `stage.behavior.use_candidate_llm_judge=true|false`
- `stage.behavior.candidate_llm_judge_splits=[validation,test]`
- `stage.pool.max_candidates_per_step=<N>`

## Run

Install:

```bash
pip install -e .
```

Run ingest stage directly:

```bash
jamendo-ingest
```

Or run orchestrator:

```bash
jamendo-run
```

Run caption join stage (with your rewritten captions CSV):

```bash
python -m jamendo_instruct.run stage=caption_join stage.rewrites.csv_path=/path/to/rewritten_captions.csv
```

Run structured-view stage:

```bash
python -m jamendo_instruct.run stage=structured_view
```

Run embeddings stage:

```bash
python -m jamendo_instruct.run stage=embeddings
```

Run neighborhood stage:

```bash
python -m jamendo_instruct.run stage=neighborhood
```

Run graph stage:

```bash
python -m jamendo_instruct.run stage=graph
```

Run chains stage:

```bash
python -m jamendo_instruct.run stage=chains
```

Run instructions stage:

```bash
python -m jamendo_instruct.run stage=instructions
```

Run validation stage:

```bash
python -m jamendo_instruct.run stage=validation
```

Run relevance-pool stage:

```bash
python -m jamendo_instruct.run stage=relevance_pool
```

## Benchmark Path

Recommended benchmark-oriented order after `chains`:

1. `instructions`
2. `validation`
3. `relevance_pool`

Intended architectural direction after the current refactor:
- `instructions` emits three semantic layers:
- deterministic delta from the chain payload
- `semantic_delta_full` as the exhaustive step oracle
- `semantic_delta_verbalized` as the instruction-conditioned request
- typed internal semantic-item views for both deltas
- `validation` judges instruction faithfulness against the full semantic delta
- `relevance_pool` grades candidate satisfaction primarily against the verbalized semantic delta and stores full-delta satisfaction as a diagnostic
- optional whole-split reason labeling can assign every same-split clip a success or failure reason for each step
- `relevance_pool` also exposes a text-encoder solvability audit per step and an LLM candidate-judge path for frontier failure-mode labeling

Example benchmark-oriented commands for an existing run:

```bash
PYTHONPATH=src /path/to/your/env/bin/python src/jamendo_instruct/run.py \
  stage=instructions \
  runtime.output_root=/gpfs/scratch/acw749/datasets/maxcaps_instruct \
  runtime.run_name=v1 \
  stage.caption.signal_mode=both \
  stage.verification.enabled=false
```

```bash
PYTHONPATH=src /path/to/your/env/bin/python src/jamendo_instruct/run.py \
  stage=validation \
  runtime.output_root=/gpfs/scratch/acw749/datasets/maxcaps_instruct \
  runtime.run_name=v1 \
  stage.behavior.use_llm_judge=true \
  stage.behavior.skip_llm_on_precheck_failure=false
```

```bash
PYTHONPATH=src /path/to/your/env/bin/python src/jamendo_instruct/run.py \
  stage=relevance_pool \
  runtime.output_root=/gpfs/scratch/acw749/datasets/maxcaps_instruct \
  runtime.run_name=v1 \
  stage.behavior.require_validated_instructions=true \
  stage.behavior.only_keep_validation_passed=true \
  stage.behavior.write_full_dataset_labels=true \
  stage.behavior.full_dataset_label_splits=[validation,test] \
  stage.behavior.use_text_encoder_audit=true \
  stage.behavior.use_candidate_llm_judge=true
```

Artifacts to inspect after the benchmark path:
- `instructions/chain_step_instructions.jsonl`
  now includes `semantic_delta_full`, `semantic_delta_verbalized`, typed internal delta views, and `semantic_constraints`
- `validation/validated_instructions.jsonl`
  stores the validation decision plus the semantic deltas and typed views actually used
- `relevance_pool/chain_step_relevance_pools.jsonl`
  stores the constructive candidate pools plus per-step `solvability_audit`
- `relevance_pool/chain_step_candidate_reason_labels.jsonl`
  optional sidecar with one reason code per same-split clip per step

## Current Run Commands

For the current canonical run `v1`, these commands are the canonical rerun commands with the current artifact layout.

Manifest-only embeddings refresh:

```bash
PYTHONPATH=src /data/home/acw749/anaconda3/envs/instruct_embed/bin/python src/jamendo_instruct/run.py \
  stage=embeddings \
  runtime.output_root=/gpfs/scratch/acw749/datasets/maxcaps_instruct \
  runtime.run_name=v1 \
  stage.io.input_manifest_csv=/gpfs/scratch/acw749/datasets/maxcaps_instruct/v1/structured_view/structured_clip_manifest.csv \
  stage.io.output_dir=/gpfs/scratch/acw749/datasets/maxcaps_instruct/v1/embeddings \
  stage.io.audio_embeddings_dir=/gpfs/scratch/acw749/datasets/maxcaps_instruct/audio/v1 \
  stage.io.text_embeddings_dir=/gpfs/scratch/acw749/datasets/maxcaps_instruct/text/v1 \
  stage.behavior.manifest_only=true \
  stage.behavior.overwrite_existing=false
```

Embeddings compute / resume:

```bash
PYTHONPATH=src /data/home/acw749/anaconda3/envs/instruct_embed/bin/python src/jamendo_instruct/run.py \
  stage=embeddings \
  runtime.output_root=/gpfs/scratch/acw749/datasets/maxcaps_instruct \
  runtime.run_name=v1 \
  stage.io.input_manifest_csv=/gpfs/scratch/acw749/datasets/maxcaps_instruct/v1/structured_view/structured_clip_manifest.csv \
  stage.io.output_dir=/gpfs/scratch/acw749/datasets/maxcaps_instruct/v1/embeddings \
  stage.io.audio_embeddings_dir=/gpfs/scratch/acw749/datasets/maxcaps_instruct/audio/v1 \
  stage.io.text_embeddings_dir=/gpfs/scratch/acw749/datasets/maxcaps_instruct/text/v1 \
  stage.behavior.manifest_only=false \
  stage.behavior.overwrite_existing=false
```

Neighborhood retrieval:

```bash
PYTHONPATH=src /data/home/acw749/anaconda3/envs/instruct_embed/bin/python src/jamendo_instruct/run.py \
  stage=neighborhood \
  runtime.output_root=/gpfs/scratch/acw749/datasets/maxcaps_instruct \
  runtime.run_name=v1 \
  stage.io.input_manifest_csv=/gpfs/scratch/acw749/datasets/maxcaps_instruct/v1/structured_view/structured_clip_manifest.csv \
  stage.io.input_lookup_manifest_csv=/gpfs/scratch/acw749/datasets/maxcaps_instruct/v1/embeddings/embedding_lookup_manifest.csv \
  stage.io.output_dir=/gpfs/scratch/acw749/datasets/maxcaps_instruct/v1/neighborhood
```

Graph construction:

```bash
PYTHONPATH=src /data/home/acw749/anaconda3/envs/instruct_embed/bin/python src/jamendo_instruct/run.py \
  stage=graph \
  runtime.output_root=/gpfs/scratch/acw749/datasets/maxcaps_instruct \
  runtime.run_name=v1 \
  stage.io.input_manifest_csv=/gpfs/scratch/acw749/datasets/maxcaps_instruct/v1/structured_view/structured_clip_manifest.csv \
  stage.io.input_nodes_csv=/gpfs/scratch/acw749/datasets/maxcaps_instruct/v1/neighborhood/neighborhood_nodes.csv \
  stage.io.input_edges_csv=/gpfs/scratch/acw749/datasets/maxcaps_instruct/v1/neighborhood/neighborhood_edges.csv \
  stage.io.output_dir=/gpfs/scratch/acw749/datasets/maxcaps_instruct/v1/graph
```

Chain mining:

```bash
PYTHONPATH=src /data/home/acw749/anaconda3/envs/instruct_embed/bin/python src/jamendo_instruct/run.py \
  stage=chains \
  runtime.output_root=/gpfs/scratch/acw749/datasets/maxcaps_instruct \
  runtime.run_name=v1 \
  stage.io.input_manifest_csv=/gpfs/scratch/acw749/datasets/maxcaps_instruct/v1/structured_view/structured_clip_manifest.csv \
  stage.io.input_nodes_csv=/gpfs/scratch/acw749/datasets/maxcaps_instruct/v1/neighborhood/neighborhood_nodes.csv \
  stage.io.input_transitions_csv=/gpfs/scratch/acw749/datasets/maxcaps_instruct/v1/graph/transition_graph_edges.csv \
  stage.io.output_dir=/gpfs/scratch/acw749/datasets/maxcaps_instruct/v1/chains
```

Instruction generation:

```bash
PYTHONPATH=src /data/home/acw749/anaconda3/envs/instruct_embed/bin/python src/jamendo_instruct/run.py \
  stage=instructions \
  runtime.output_root=/gpfs/scratch/acw749/datasets/maxcaps_instruct \
  runtime.run_name=v1 \
  stage.io.input_manifest_csv=/gpfs/scratch/acw749/datasets/maxcaps_instruct/v1/structured_view/structured_clip_manifest.csv \
  stage.io.input_chains_jsonl=/gpfs/scratch/acw749/datasets/maxcaps_instruct/v1/chains/sampled_chains.jsonl \
  stage.io.output_dir=/gpfs/scratch/acw749/datasets/maxcaps_instruct/v1/instructions \
  stage.caption.signal_mode=both \
  stage.verification.enabled=false
```

Instruction validation:

```bash
PYTHONPATH=src /data/home/acw749/anaconda3/envs/instruct_embed/bin/python src/jamendo_instruct/run.py \
  stage=validation \
  runtime.output_root=/gpfs/scratch/acw749/datasets/maxcaps_instruct \
  runtime.run_name=v1 \
  stage.io.input_instructions_jsonl=/gpfs/scratch/acw749/datasets/maxcaps_instruct/v1/instructions/chain_step_instructions.jsonl \
  stage.io.input_prepared_jsonl=/gpfs/scratch/acw749/datasets/maxcaps_instruct/v1/instructions/chain_step_instruction_inputs.jsonl \
  stage.io.output_dir=/gpfs/scratch/acw749/datasets/maxcaps_instruct/v1/validation \
  stage.behavior.use_llm_judge=true \
  stage.behavior.skip_llm_on_precheck_failure=false
```

Constructive relevance pools:

```bash
PYTHONPATH=src /data/home/acw749/anaconda3/envs/instruct_embed/bin/python src/jamendo_instruct/run.py \
  stage=relevance_pool \
  runtime.output_root=/gpfs/scratch/acw749/datasets/maxcaps_instruct \
  runtime.run_name=v1 \
  stage.io.input_manifest_csv=/gpfs/scratch/acw749/datasets/maxcaps_instruct/v1/structured_view/structured_clip_manifest.csv \
  stage.io.input_lookup_manifest_csv=/gpfs/scratch/acw749/datasets/maxcaps_instruct/v1/embeddings/embedding_lookup_manifest.csv \
  stage.io.input_nodes_csv=/gpfs/scratch/acw749/datasets/maxcaps_instruct/v1/neighborhood/neighborhood_nodes.csv \
  stage.io.input_edges_csv=/gpfs/scratch/acw749/datasets/maxcaps_instruct/v1/neighborhood/neighborhood_edges.csv \
  stage.io.input_chains_jsonl=/gpfs/scratch/acw749/datasets/maxcaps_instruct/v1/chains/sampled_chains.jsonl \
  stage.io.input_prepared_jsonl=/gpfs/scratch/acw749/datasets/maxcaps_instruct/v1/instructions/chain_step_instruction_inputs.jsonl \
  stage.io.input_validation_jsonl=/gpfs/scratch/acw749/datasets/maxcaps_instruct/v1/validation/validated_instructions.jsonl \
  stage.io.output_dir=/gpfs/scratch/acw749/datasets/maxcaps_instruct/v1/relevance_pool \
  stage.behavior.write_full_dataset_labels=true \
  stage.behavior.use_text_encoder_audit=true \
  stage.behavior.use_candidate_llm_judge=true
```

Run the global pipeline for a specific run:

```bash
python -m jamendo_instruct.pipeline runtime.run_name=v1
```

Run only Stage 4 from an existing run cache:

```bash
python -m jamendo_instruct.pipeline runtime.run_name=v1 pipeline.run_stages=[neighborhood] pipeline.start_stage=neighborhood pipeline.end_stage=neighborhood
```

Override config values:

```bash
jamendo-ingest stage.tracks.metadata_dir=/gpfs/scratch/acw749/datasets/maxcaps_instruct/metadata
```

Use source split instead of custom grouped split:

```bash
jamendo-ingest stage.split.mode=source_column
```

Limit download for a quick smoke run:

```bash
jamendo-ingest stage.download.max_files=10 runtime.run_name=ingest_smoke
```

## Notes

- Audio is **not** downloaded.
- Metadata JSONLs and `final_caption30sec.jsonl` are expected under `/gpfs/scratch/acw749/datasets/maxcaps_instruct/metadata` by default.
- `file_path` is synthesized from `stage.audio.file_template` and `track_id`.
- Captions are loaded directly in memory during ingest (no SQLite dependency).
- The default ingest dataset is caption-only at clip level: duplicate shard snapshots are collapsed on `track_id`, then one manifest row is written per captioned clip with parent `track_id`, `clip_id`, `start_time`, and `end_time`.
- `structured_view` is intentionally lightweight in v1 and derives its output only from normalized tags and captions, while leaving room for optional richer parsing later.
- `embeddings` uses Hugging Face-hosted models for clip-level similarity search: `OpenMuQ/MuQ-MuLan-large` for audio and `google/embeddinggemma-300m` for text.
- `embeddings` slices audio by `start_time` and `end_time` when clip boundaries are available.
- `embeddings` resumes from `embedding_lookup_manifest.csv`, and pending rows are reconciled with on-disk `.npy` files so interrupted chunked runs can continue; set `stage.behavior.verify_existing_files=true` to force a full disk verification.
- The canonical embedding outputs are per-clip `.npy` files under `audio/` and `text/`, with `embedding_lookup_manifest.csv` recording the exact paths for each `clip_id`.
- By default, embedding vectors are stored under `${runtime.output_root}/audio/${runtime.run_name}` and `${runtime.output_root}/text/${runtime.run_name}`, while the lookup manifest and report live under `${runtime.output_root}/${runtime.run_name}/embeddings`.
- The current canonical run name is `v1`; the older timestamped name `20260405_211504` is kept as a compatibility symlink in the artifact, audio, and text directories.
- `google/embeddinggemma-300m` requires accepting the model license on Hugging Face before first download.
- `caption_join` is optional; if no rewrite CSV is provided, it performs a pass-through join with `has_rewrite=0`.
- `neighborhood` now treats the Stage 3 structured manifest as the metadata source of truth and derives embedding paths from `clip_id` plus the audio/text embedding directories; it can still fall back to the Stage 3.5 lookup manifest for compatibility.
- `neighborhood` enforces same-split retrieval and excludes same-track neighbors by default, so local candidate regions stay within split and avoid trivial same-track transitions.
- `graph` consumes the Stage 4 node/edge artifacts, defensively filters cross-split edges, filters to tag-delta size `<= 6`, and writes retained transition edges with explicit `transition_score`, `transition_cost`, hardness, and `structured_delta_json`.
- `chains` samples short-heavy stochastic walks from the scored graph, explicitly bans revisiting any prior `clip_id` or `track_id`, and stores per-turn accumulated intent state for later instruction generation.
- `jamendo-pipeline` can resume from the earliest missing stage for a `run_name`, or rerun a selected stage range in `pipeline.mode=from_scratch`.
Install the ASR dependency set for the lyrics stage when needed:
- `pip install -e .[asr]`
