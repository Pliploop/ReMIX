# Jamendo-Instruct: Implementation Spec

> **Purpose.** The code map, artifact schemas, and runtime reality — how the
> method in `PLAN.md` is actually implemented, so a new contributor can run,
> extend, or reproduce it. Stage *commands* live in the top-level `README.md`;
> the *method* in `PLAN.md`; instruction *validation* in `validation_gate.md`.

## 1. Repository layout

```
src/jamendo_instruct/
  pipeline.py            # orchestrator: STAGE_ORDER, resume, per-stage wiring
  run.py                 # single-stage entrypoint (STAGE_RUNNERS registry)
  llm_backends.py        # transformers / vLLM (offline + OpenAI-compat) backends
  semantic_delta.py      # typed semantic-delta construction
  validation_gate.py     # graded instruction-validity gate (grade -> select)
  progress.py            # rich progress / stage tracker
  stages/                # one module per pipeline stage (run_<stage>)
    ingest, caption_join, lyrics, structured_view, embeddings,
    neighborhood, graph, chains, instructions, validation, relevance_pool
  demo/
    chains_demo.py       # Streamlit explorer + reusable dataset loader & analysis helpers
    human_validation_app.py  # Streamlit human-rating app + Admin agreement dashboard
    validation_rubric.py # the 8-question rubric + scales + item selection (shared)
conf/
  pipeline.yaml          # orchestration knobs (run_stages, start/end, mode)
  runtime/*.yaml         # output_root, run_name, seed
  stage/*.yaml           # one config per stage
  dataset/*.yaml         # source-dataset descriptors (jamendomaxcaps, mtgjamendo)
scripts/                 # CLIs + SLURM launchers (see §6)
docs/                    # PLAN, IMPLEMENTATION_SPEC, validation_gate, ROADMAP, POSITIONING
```

## 2. Orchestration

- **`pipeline.py`** runs the canonical chain
  `ingest → [caption_join] → lyrics → structured_view → embeddings → neighborhood → graph → chains → instructions → validation → relevance_pool`.
  `pipeline.mode=resume` starts from the earliest missing primary output;
  `pipeline.run_stages` / `start_stage` / `end_stage` restrict the range.
  `_primary_output_path` defines each stage's canonical artifact.
- **`run.py`** runs a single stage: `python -m jamendo_instruct.run stage=<name> …`
  with Hydra overrides (`runtime.output_root=…`, `runtime.run_name=…`,
  `stage.io.*`, `stage.models.*`, `stage.runtime.*`).
- Config resolution: each `conf/stage/<name>.yaml` uses
  `${runtime.output_root}/${runtime.run_name}/…` so a run is identified by
  `(output_root, run_name)`.

## 3. LLM backends & runtime (`llm_backends.py`)

One abstraction over four backends, selected by `stage.runtime.backend`:

- **`transformers`** / **`transformers_bnb`** — `AutoModelForCausalLM` (or
  `AutoModelForImageTextToText` for the `qwen3_6` family); bnb = 4-bit nf4.
- **`vllm`** — offline `vllm.LLM` (`build_vllm_offline_chat_model` +
  `decode_vllm_chat_completions`, **batched**).
- **`vllm_local` / `sglang_local`** — OpenAI-compatible HTTP servers.
- **`auto`** — `choose_auto_backend` picks vLLM/tp/quant from visible GPU memory.

Model family inference (`infer_llm_model_family`) special-cases `qwen3_6`;
everything else is `causal_lm`. Chat templating via `apply_chat_template`.

### Runtime gotchas (learned the hard way — bake these into any new launcher)

1. **libstdc++ / `CXXABI_1.3.15`.** vLLM's import chain
   (`diskcache → sqlite3 → libicui18n.so.78`) needs `CXXABI_1.3.15`, which the
   system libstdc++ and the `cuda/12.6.2-gcc-12.2.0` module (gcc-12) do **not**
   provide. Only the conda env's bundled libstdc++ has it. Every worker script
   must prepend it: `export LD_LIBRARY_PATH=/data/home/acw749/conda-envs/instruct_embed/lib:$LD_LIBRARY_PATH`.
   (In `run_instruction_raw.sh`, `run_pipeline_stage.sh`, `run_llm_validation_judge.sh`.)
2. **fp8 needs Hopper.** *Online* fp8 quantization (`--quantization fp8`, used for
   `gemma-4-31B-it` to fit 2×A100-40GB) compiles an `fp8e4nv` (E4M3) Triton kernel
   that only runs on **sm90 / H100**. On A100 (sm80) it fails at engine init with
   `type fp8e4nv not supported`. Pin such jobs with `--constraint=hopper`.
   Pre-quantized fp8 checkpoints (e.g. `Qwen/Qwen3.6-27B-FP8`) use a marlin path
   that works on Ampere.
3. **sae profile** sets `VLLM_USE_DEEP_GEMM=0` / `VLLM_MOE_USE_DEEP_GEMM=0`
   (`PROFILE=sae` in the worker).

## 4. Key artifact schemas

| Stage | Path (under `<run_root>/`) | Schema notes |
|---|---|---|
| structured_view | `structured_view/structured_clip_manifest.csv` | per-clip; columns in `PLAN.md` §3 |
| embeddings | `embeddings/embedding_lookup_manifest.csv` | clip → embedding lookup |
| neighborhood | `neighborhood/neighborhood_{nodes,edges}.csv` | candidate transitions |
| chains | `chains/sampled_chains.jsonl` | `chain_id, chain_length, split, seed, steps[]`; steps carry `transition_score`, `hardness`, `structured_delta`, `accumulated_intent_state` |
| instructions | `instructions_*/chain_step_instructions.jsonl` | one record per `(chain,turn,variant)`; the merged form of `step_json/*.json` |
| instructions | `instructions_*/chain_step_instruction_inputs.jsonl` | deterministic prepared prompt inputs (read by validation & relevance_pool) |
| validation | `…/validation/validated_instructions.jsonl` | full instruction record + `validation` block; `relevance_pool` reads `validation.accepted` |
| relevance_pool | `relevance_pool/chain_step_relevance_pools.jsonl` | graded candidate pools (benchmark ground truth) |

Instruction records also carry `semantic_delta_full` / `semantic_delta_verbalized`
(`new`/`lost`/`preserved`/`primary_edit`/`caption_only_change` + typed variants),
`semantic_constraints` (back-compat alias of `semantic_delta_full`), and
`instruction_plan`. `relevance_pool` keys validation by `(chain_id, turn_index)`
and grades candidates against `semantic_delta_verbalized`.

## 5. Validation subsystem

- **`demo/validation_rubric.py`** — single source of truth for the 8 rubric
  questions (`meaningful_change`, `target_follows`, `source_support`,
  `source_compatible`, `conservation_supported`, `edit_specificity`,
  `clarity_actionability`, `overall_validity`), the Likert/degree scales,
  scoring, and the sidecar/assignment item-selection helpers. Imported by both
  the human app and the LLM judge so humans and models rate **identical**
  questions on **identical** inputs.
- **`scripts/llm_validation_judge.py`** — text-only graded judge. Loads the same
  item set (run-root or frozen sidecar), sends each instruction the source/target
  caption+tags+metadata, asks all 8 questions in one strict-JSON call, maps
  answers to scores, writes `llm_ratings*.jsonl` (`annotator_id = llm:<model_id>`,
  `modality: text_only`). Resumable; `--emit-validated` also runs the gate.
- **`validation_gate.py`** — `grade_records` (annotate every variant, cut
  nothing) then `select_chain_variants` (per step pick best passing variant with
  **fallback**; truncate contextual chains at the first step with no passing
  variant) → `validated_instructions.jsonl`. See `validation_gate.md`.
- **`demo/human_validation_app.py`** — Streamlit rating app + **Admin** tab with
  Human↔LLM and **cross-LLM** agreement tables. Agreement metrics per question:
  `mean_diff` (bias), `mae`, `within1_rate`, `accept_agree_rate`, Cohen
  `accept_kappa`, **Gwet `accept_ac1`** (prevalence-robust), **`quadratic_kappa`**,
  `pearson_r`, `spearman_r`. Ratings files are globbed and deduped per
  `(model, item)`.

## 6. Scripts inventory

Data prep / build:
- `prepare_music4all_metadata.py`, `prepare_mtg_jamendo_metadata.py`,
  `mtg_jamendo_caption_audio.py`, `music4all_caption_audio.py` — source ingest.
- `merge_instruction_step_json.py` — merge `step_json/*.json` →
  `chain_step_instructions.jsonl` (prerequisite before validation/relevance_pool
  for a `instructions_axis_focused_*` folder).
- `build_human_validation_assignment.py` — build a frozen **sidecar** for
  validation: default = sampled human slice; `--full-coverage --split validation`
  = whole validation split. Writes `assignment_*.jsonl` + `.sidecar.json`.
- `build_validated_instructions.py` — grade → select the graded gate (CPU).

Runtime / launchers:
- `run_pipeline_stage.sh` — generic worker for any stage via
  `python -m jamendo_instruct.run` with the env fixes baked in.
- `run_instruction_raw.sh` + `launch_{andrena,sae}_instruction.sh` — instruction
  generation (vLLM, Qwen3.6-27B-FP8).
- `run_llm_validation_judge.sh` — the graded judge worker (env vars:
  `MODEL_ID, BACKEND, TENSOR_PARALLEL_SIZE, QUANTIZATION, KV_CACHE_DTYPE,
  MAX_MODEL_LEN, GPU_MEMORY_UTILIZATION, FROZEN_SIDECAR_JSON, OUTPUT_NAME,
  PROFILE`).
- `run_human_validation_cloudflared.slurm.sh`, `run_chains_demo_*` — Streamlit
  app deployments.

Analysis:
- `paper_data_stats.py` — publication figures/tables (reuses `chains_demo`
  analysis helpers; caches parquet; genre via a curated tag→genre map). Outputs
  `paper/figures/*.pdf` + `paper/tables/*`.

## 7. Config layout

- `conf/runtime/*.yaml` → `output_root`, `run_name`, `seed=13`.
- `conf/stage/<name>.yaml` → per-stage `io`/`models`/`runtime`/`behavior`. Notable:
  - `instructions.yaml`: `variants_per_step=5`, `clause_budget`, `axis_guidance`
    (`soft|sampled_focus|hybrid`), `generation` sampling.
  - `neighborhood.yaml`: `audio_top_k=500`, `retain_top_k=100`, rerank weights,
    `enforce_same_split=true`.
  - `chains.yaml`: `target_num_chains=1_000_000`, length 1–6 log-normal.
  - `validation.yaml`: heuristic checks + judge; `relevance_pool.yaml`:
    `max_candidates_per_step=96`, `candidate_llm_judge_splits=[validation,test]`,
    solvability audit, text encoder `embeddinggemma-300m`.
- Stage overrides: `python -m jamendo_instruct.run stage=<name> stage.io.output_dir=… stage.models.model_id=…`.

## 8. Reproducing a run (music4all example)

1. `pipeline` (or per-stage) through `chains` → `sampled_chains.jsonl`.
2. `instructions` (vLLM, Qwen3.6-27B-FP8, `axis_guidance=sampled_focus`,
   `write_step_json=true`) → `instructions_axis_focused_5/step_json`.
3. `merge_instruction_step_json.py` → `chain_step_instructions.jsonl`.
4. Build sidecars (`build_human_validation_assignment.py`), run the graded judge
   (`run_llm_validation_judge.sh`) on validation+test, then
   `build_validated_instructions.py` → `validated_instructions.jsonl`.
5. `run.py stage=relevance_pool` (inputs pointed at the axis folder) → benchmark.

Current status and what remains: `ROADMAP.md`.
