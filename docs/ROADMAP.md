# Roadmap: status and execution plan to submission

Last updated 2026-09-25. Method: `PLAN.md`. Code map: `IMPLEMENTATION_SPEC.md`.
Benchmark API: `evaluation.md`. Grading spec: `benchmark_grading_v2.md`.
Trained baseline: `src/remix_c/README.md`. Paper: Overleaf project `6a4ff76b…`
(local clone `paper/6a4ff76b87fdf283ad5b5d68/`, push token in `~/.overleaf_token`).

## Status

| Component | Music4All | MTG-Jamendo |
|---|---|---|
| Chains (1M sampled) + instructions (`axis_focused_5`) | done: 72,769 chains · 181,627 steps · 894,981 variants | done: 81,429 chains · 202,411 steps · 1,006,962 variants |
| LLM validation, test split | done (Qwen all variants, Gemma subset) | partial |
| LLM validation, val split | done (both judges) | running (Qwen + Gemma, resumes queued) |
| LLM validation, train split | running (Qwen ×2 on sae after val, Gemma ×1 on andrena; ~4–7 days) | not started |
| Gate (`validated_instructions.jsonl`) | built from partial ratings; rebuild when judging completes | rebuild when judging completes |
| ReMIX-B relevance pool (v2 grading) | done: 9,107 test queries, 866k graded candidates | 43.9% of test; relaunch after the test gate is final |
| Benchmark export + 14 baselines | done (`results/remix_b/music4all/`) | pending pool |
| ReMIX-C | small model (frozen towers, 2-layer fusion) trained on filtered and on all data; all-data run still training | not trained |
| Paper | full-data numbers, figures, results table pushed | — |

Current ReMIX-B headline (Music4All test, nDCG@10): LLM rerank 0.196 · hybrid fusion
0.128 · LLM caption rewrite 0.121 · **ReMIX-C (all data) 0.113** · instruction text
0.090 · seed audio 0.048 · random 0.002 · text oracle 0.649. ReMIX-C is on the
cost/quality Pareto frontier and beats every non-LLM baseline.

## Remaining work to a submittable paper

Ordered by dependency. `[gate]` marks items that must finish before the final evaluation.

### A. Data and ground truth

1. **Finish LLM validation on all splits, both catalogues** `[gate]`. Music4All train
   is running; MTG val is running; MTG train and the rest of MTG test are not started.
   GPU budget: andrena is capped at 8 GPUs per user (Gemma needs 2, ReMIX-C 2);
   sae runs Qwen on 1 GPU. Relaunch with `SPLIT=… TIME_LIMIT=4-12:00:00
   scripts/launch_llm_validation_judge.sh`; jobs share work through per-item parts.
2. **Decide the gate policy** `[gate]`. The gate currently keeps one rating per variant
   (Qwen's where it exists, else Gemma's). Options: keep it, or require both judges to
   accept (as the ReMIX-C filter does). Changing it changes the benchmark query set,
   so decide before rebuilding pools and before the final benchmark run.
3. **Rebuild the gates** (`scripts/build_validated_instructions.py`) for both catalogues,
   then re-export (`scripts/export_benchmark.py`) and re-check the Music4All query set.
4. **MTG-Jamendo relevance pool**: relaunch `scripts/launch_relevance_pool.sh`
   (v2 grading, `JUDGE_MODEL_ID=Qwen/Qwen3.6-27B-FP8`, `SPLITS=test`, 8 shards,
   `RESUME=1`), then `scripts/relevance_pool_analysis.py --dataset mtg_jamendo`.
5. **Human studies** (Cloudflare rating app, `scripts/run_human_validation_cloudflared.slurm.sh`):
   - instruction rubric on the frozen slice: human–human and human–LLM agreement,
     threshold calibration;
   - relevance grades on a stratified sample of ~300–500 (query, candidate) pairs
     (balanced over grades 0–6 and edit axes, ~100 double-rated): weighted κ / α,
     binary agreement at grade ≥ 3, and Kendall's τ between baseline rankings under
     human vs LLM labels on the sample. Needs a new task type in the app.

### B. Evaluation

6. **Final benchmark runs**, both catalogues, all baselines + ReMIX-C
   (`scripts/run_remix_b.py --dataset … --output-dir results/remix_b/<ds>`;
   add checkpoints with `--ckpt … --as <row>`; tables and figures with
   `scripts/remixb_paper.py`).
7. **Cross-dataset transfer**: ReMIX-C trained on Music4All evaluated on MTG-Jamendo
   and vice versa (needs an MTG ReMIX-C and an MTG `data/` config).
8. **Reranker from a different model family** (e.g. Gemma-4-31B) so the strongest
   baseline does not share a model with the pool judge.
9. **Per-query analysis**: save per-query metrics from `run_remix_b.py`, then report
   results by edit axis, transition difficulty, and candidate provenance, plus
   instruction-shuffle and seed-swap controls.
10. **Conversational track** or an explicit scope statement: ReMIX-B currently scores
    history-unaware queries only, while the paper motivates history-aware ones.

### C. ReMIX-C recipe (in parallel with A/B)

11. Fair data-quality vs data-scale comparison: retrain the filtered model once both
    judges have rated the train split, at matched steps against the all-data run
    (marked `\subjecttoupdate` in the paper).
12. Ablations on all data: `objective=lejepa`; `objective.n_swap=0`; the large model
    with unfrozen towers (`model=large`) with early stopping on val MRR.
13. 2–3 seeds for the reported configuration.

### D. Writing and release

14. Related work (empty) and a verified bibliography (`references.bib` has 5 entries;
    the two added in the cleanup pass are marked "verify").
15. Update the pipeline figure (panel 5 still shows the old 0–4 relevance scale).
16. Page limit, figure placement, NeurIPS checklist (commented out in the source).
17. Release: Hugging Face datasets (push blocked: `hf auth login --force` needed),
    audio licensing statement for Music4All / MTG-Jamendo, datasheet, Croissant metadata.

## Resolved decisions (for reference)

- Relevance grading v2: contiguous 0–6 scale from categorical judge labels, similarity
  gate (floor 0.25 + top-48 judged), failure-mode taxonomy (`benchmark_grading_v2.md`).
- Benchmark metrics via `ir_measures`; relevant = grade ≥ 3; nDCG gains = grade.
- FLOPs in results are query-time work with precomputed catalogue embeddings; ReMIX-C
  reports catalogue encoding separately as `index_tflops`.
- Paper figures share `scripts/paper_style.py` (print size, Inter, grey edges, square
  half-column panels).
