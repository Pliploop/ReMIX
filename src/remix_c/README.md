# ReMIX-C — trainable composed-music-retrieval baseline

Given a **seed clip** and an **edit instruction**, embed a query that retrieves the
**target clip** from an audio corpus. Two objectives share one architecture:

| objective | loss | collapse control | retrieval score |
|---|---|---|---|
| `contrastive` (ReMIX-C) | InfoNCE, negatives = changed target / instruction / both | negatives | cos(q, t) |
| `lejepa` (LeReMIX-C) | MSE ‖h(q) − t‖² (no stop-grad) | SIGReg on q and t | cos(h(q), t) |

Everything below is a hydra key; nothing about the architecture or training is
hard-coded beyond the MuQ-MuLan tower internals.

## Architecture (`model.py`)

```
seed wav (N×10 s) ─► seed_tower (MuQ → proj)      ─► audio tokens (N·250 × 768) ─┐
instruction ──────► instruction_tower (XLM-R → 8L) ─► text tokens (L × 768) ──────┼─► fusion ─► [CLS] ─► q (512)
                                                                                  ┘   (linear in-proj per modality,
                                                                                        pre-norm TransformerEncoder)
target wav (N×10 s) ─► target_tower (separate MuQ copy) ─► mean-pool ─► MuLan audio head ─► t (512)
```

- All three towers come from `OpenMuQ/MuQ-MuLan-large`. `target_tower` is loaded as a
  second pretrained instance (MuQ's `weight_norm` blocks `deepcopy`), so the two audio
  towers can drift apart freely. With the target tower
  frozen, `t` is exactly the MuLan audio embedding that the other baselines use.
- Clips are 30 s long. Each one is cut into `data.n_windows` 10 s windows (MuQ's
  training length), encoded per window, and the token sequences are concatenated
  in time.
- `model.<tower>.freeze=true` turns off gradients **and** keeps the tower in eval
  mode. We bypass MuQ-MuLan's stock `forward`, which wraps XLM-R in `no_grad`
  unconditionally, so `freeze=false` really does fine-tune every backbone.
- Two presets: `model=small` (2-layer fusion, all towers frozen) and `model=large`
  (6-layer fusion, everything trainable).

## Objectives (`objectives.py`)

**Contrastive.** For each anchor *i*, the model builds its own query `q_i = f(seed_i, instr_i)` and K
*instruction-swapped* queries `q_ik = f(seed_i, instr_j)`, where *j* is another
batch item. One InfoNCE runs over all (query, target) pairs, targets gathered
across GPUs. The only positive is (q_i, t_i). Negatives are
(q_i, t_j) with the target changed, (q_ik, t_i) with the instruction changed, and
(q_ik, t_j) with both changed. The instruction-swap negatives stop the model from
collapsing to "retrieve something that sounds like the seed". False negatives are
masked: batch items that share a target clip, and swaps with an identical
(normalised) instruction or the same target. `objective.n_swap=0` gives plain in-batch InfoNCE.

**LeJEPA** (LeVLJEPA, Kuhn et al., arXiv:2607.00784; SIGReg from LeJEPA, Balestriero &
LeCun). `L = (1−λq−λt)·‖h(q) − t‖² + λq·SIGReg(q) + λt·SIGReg(t)`. `h` is the
paper's predictor: a depth-4 MLP of width 2048 with BatchNorm, GELU and dropout 0.1.
There is **no stop-gradient**, so both the fusion and target branches learn from
the prediction loss, and SIGReg (Epps–Pulley on random 1-D projections versus
N(0, 1)) keeps each side from collapsing. Embeddings are left unnormalised during
training and L2-normalised only for retrieval.

## Data (`data.py`)

Build a one-off **unfiltered** index of every instruction variant, with both
judges' scores attached. Nothing is dropped at this step.

```bash
sbatch -p computeshort -c 4 --mem=32G -t 1:00:00 --wrap \
  "PYTHONPATH=src python -m remix_c.data --run-root /gpfs/scratch/acw749/datasets/music4all_instruct/music4all_v1"
# -> <run>/instructions_axis_focused_5/remix_c/{index,clips}.parquet
```

`index.parquet` has one row per (chain, turn, variant), with seed, target, the
`history_unaware_instruction` text, the split, and `{qwen,gemma}_{overall,mean}`.
`overall` is the `overall_validity` question; `mean` is the mean of the 8 rubric
questions. `clips.parquet` maps each clip to its audio file and time span.

**Filtering happens in the datamodule at load time:**

- `data.filter_threshold=null` (default) trains on every train-split variant.
- `data.filter_threshold=4 data.filter_criteria=overall|mean` keeps a variant only
  if both judges rated it and both scored ≥ threshold.
- A training item is a (chain, turn) step. Each time the step is drawn, one of its
  surviving variants is sampled, so the extra variants act as instruction
  augmentation.
- The val set is fixed: `val` split, both judges' `overall_validity` ≥ 4, the first
  passing variant per step, `data.val_max_steps` steps.

Audio is decoded from the original MP3s (`clips.parquet → file_path`) in the
dataloader workers, mixed down to mono, and resampled to 24 kHz.

## Training (`train.py`, `conf/`)

```bash
# 4 runs = {small, large} x {contrastive, lejepa}
PYTHONPATH=src python -m remix_c.train model=small objective=contrastive
PYTHONPATH=src python -m remix_c.train model=large objective=lejepa data.filter_threshold=4
# any key is overridable, e.g.
PYTHONPATH=src python -m remix_c.train model=large model.target_tower.freeze=true \
  objective.n_swap=8 data.batch_size=8 trainer.devices=2 optim.lr_towers=5e-6
```

Config groups: `data/`, `model/`, `objective/`. The trainer, optimiser, logger
(wandb) and callbacks live in `config.yaml`. Interpolation keeps them consistent,
for example `objective.dim: ${model.embed_dim}` and `optim.total_steps: ${trainer.max_steps}`.
The optimiser is AdamW with two learning rates (`optim.lr_towers` for pretrained
towers, `optim.lr_heads` for fusion, objective and new heads), cosine schedule
with warmup, bf16-mixed. SLURM: `scripts/launch_remix_c.sh`.

## Validation (`callbacks.py`)

Runs every `trainer.val_check_interval` steps. The corpus is the val targets, and
each query's seed clip is excluded from its own ranking.

- `RetrievalMetrics` logs `val/R@{1,10,50}`, `val/MRR`, `val/medR`, plus:
  - `val/R@10_shuffled`: the same queries with instructions shuffled inside the
    batch. `val/instruction_gap` = R@10 − R@10_shuffled. If this is about 0, the
    model ignores the instruction.
  - `val/R@10_seed_nn`: the seed clip itself, embedded by the target tower, used
    as the query. This is the seed-NN reference in the current target space.
- `EmbeddingPlots` logs to wandb:
  - cosine-similarity histograms: positive pairs, instruction-shuffled pairs,
    seed→target pairs and random pairs;
  - a t-SNE of queries and targets, showing the modality gap and cluster structure.

## Benchmark (ReMIX-B)

`RemixCBaseline` (`remix_c`) in `model.py` implements the harness contract
(`prepare(corpus)` / `rank_all(queries, k)`): it embeds the test catalogue with the
target tower (reported as index-time `index_tflops`) and ranks by the objective's
retrieval score. `RemixCUntrained` (`remix_c_untrained`) builds the same model from
the checkpoint's config without loading its weights. FLOPs are counted with
`torch.utils.flop_counter`.

```bash
python scripts/run_remix_b.py --dataset music4all --output-dir results/remix_b/music4all \
    --baselines remix_c --ckpt <path.ckpt> --as remix_c_unfiltered
python scripts/remixb_paper.py      # table + figures
```

Checkpoints (hard-linked so a new "best" cannot replace them):
`/gpfs/scratch/acw749/remix_c/ckpts/`. Keep `ModelCheckpoint(save_last=True)` out of
the config: it only mirrors checkpoints saved by another rule, so the `latest`
callback keeps the most recent step instead.

### Results so far (Music4All test, small model, contrastive)

| run | data | best val step | nDCG@10 | R@100 | MRR |
|---|---|---|---|---|---|
| untrained | – | – | 0.002 | 0.019 | 0.005 |
| d2 (2-layer fusion, batch 64) | filtered (both judges ≥ 4, 58k steps) | 8k | 0.099 | 0.567 | 0.173 |
| d4 (4-layer fusion, batch 128) | filtered | 4k | 0.097 | 0.552 | 0.170 |
| d2 | all train variants (163k steps) | 17k (still training) | **0.113** | **0.596** | **0.194** |

On the filtered set validation loss rises after ~2k steps; on all data it stays flat.
Fusion depth does not help; data volume does. Open: retrain the filtered model once the
train split is fully judged, `objective=lejepa`, `objective.n_swap=0`, `model=large`.

## Files

| file | role |
|---|---|
| `data.py` | index builder (CLI), clip loader, `Steps` dataset, `RemixData` datamodule |
| `model.py` | towers, fusion, `RemixC` LightningModule, `RemixCBaseline` |
| `objectives.py` | `Contrastive`, `LeJEPA`, `sigreg` |
| `callbacks.py` | `RetrievalMetrics`, `EmbeddingPlots` |
| `train.py` | hydra entry point |
| `conf/` | `config.yaml` + `data/`, `model/`, `objective/` groups |
