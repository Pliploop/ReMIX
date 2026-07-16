# Roadmap: from dataset to benchmark + baselines

Status of the remaining work to a complete dataset + benchmark paper. The
generation pipeline (`ingest → … → chains → instructions`) is built and has been
run at scale; what remains is the validation → benchmark → modelling tail plus
two net-new components (a retrieval eval harness and a contrastive training
pipeline). See `PLAN.md` for the method and `IMPLEMENTATION_SPEC.md` for the code
map.

## Two levels of "validation" (don't conflate — see `validation_gate.md`)

1. **Instruction validity** (upstream filter): is a `source → target` instruction
   a good query? → graded rubric judge → grade → select → `validated_instructions.jsonl`.
2. **Candidate relevance** (`relevance_pool`, the benchmark ground truth): for a
   retained query, which candidate tracks are relevant? Multi-tier: cheap
   heuristics (tag/caption/constraint overlap, text-encoder solvability) → LLM
   candidate judge on `validation`/`test`.

## Ordered plan

0. **(prereq, per dataset)** merge `step_json → chain_step_instructions.jsonl`;
   point `validation`/`relevance_pool` at the `instructions_axis_focused_5`
   folder. *(music4all + mtg merged; maxcaps has no instructions yet.)*
1. **Cross-LLM + human validation** *(parallelisable — justifies the judge)*:
   - ✅ **Cross-LLM done** on both sidecars (frozen human slice + full-validation
     split) for **both** datasets with **Qwen3.6-27B-FP8** and **Gemma-4-31B-it**.
     Qwen↔Gemma agreement is strong (overall_validity: Pearson ≈ 0.72, Gwet AC1
     ≈ 0.88, within-1 ≈ 0.95); Gemma is systematically ~0.2–0.4 more lenient.
   - ⏳ **Human ratings**: collect on the frozen human slice → 3-way
     human/Qwen/Gemma agreement (Admin tab) → **calibrate the accept threshold**.
2. **Full-scale instruction validation + filtering**: run the graded judge over
   the benchmark splits (`validation` **and** `test`), then grade → select →
   `validated_instructions.jsonl` (threshold + variant fallback + chain
   truncation). This is the "filtering" step `relevance_pool` requires.
3. **Relevance pool → benchmark** (`relevance_pool`): construct candidate pools
   and grade relevance (heuristics → LLM candidate judge). This is the benchmark.
4. **Eval harness (net-new)**: load the pools, score candidates with a query
   model, compute retrieval metrics (Recall@k, mAP/nDCG). Shared by baselines
   and the finetuned model.
5. **Baselines**: zero-shot encoders (CLAP/MuLan/`embeddinggemma`), BM25/random,
   and the pre-finetune contrastive model, on the eval harness.
6. **Finetune contrastive model (net-new, largest piece)**: training pipeline
   that composes `source embedding + instruction text → target` pairs from the
   `train` split; contrastive/InfoNCE loss; eval on the benchmark.

## Gaps / decisions

- **Retrieval task definition** underpins 4–6: query = source-clip embedding +
  instruction text → retrieve target; choose encoders/fusion. Decide before 4.
- **Benchmark coverage**: the graded judge has run on `validation` for both
  datasets; the `test` split still needs judging before the benchmark gate is
  complete.
- **maxcaps**: instructions not yet generated (largest catalog, most headroom).
- **Not-pipeline but needed for "done"**: dataset packaging/release (HF, splits)
  + audio **licensing** for Music4All / MTG-Jamendo; paper writing (data-section
  figures done via `scripts/paper_data_stats.py`).

## Per-dataset status (chains / instructions / validation)

| Dataset | Chain pool | Instruction variants (`axis_focused_5`) | Merged jsonl | Val-split LLM ratings |
|---|---:|---:|:--:|:--:|
| music4all (`music4all_v1`) | 1,000,000 | 88,204 (7,554 chains, 18,744 steps) | ✅ | ✅ Qwen + Gemma (slice 450, full 5,112) |
| mtg_jamendo (`v1`) | 1,000,000 | 26,737 (2,186 chains, 5,373 steps) | ✅ | ✅ Qwen + Gemma (slice 494, full 5,597) |
| jamendo maxcaps (`v1`) | 1,000,000 | 0 | — | — |
