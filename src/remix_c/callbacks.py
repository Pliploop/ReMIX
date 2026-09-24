"""Validation callbacks. Both read `pl_module.val_outputs` (filled by validation_step),
gathered across ranks: q / q_shuffled (retrieval-space queries), t / seed_t (target-tower
embeddings of target / seed audio), seed_id / target_id (clip ids)."""
from __future__ import annotations

from typing import Dict

import lightning as L
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402


def _gathered(pl_module) -> Dict[str, torch.Tensor]:
    out = {k: torch.cat([o[k] for o in pl_module.val_outputs]) for k in pl_module.val_outputs[0]}
    if pl_module.trainer.world_size > 1:
        out = {k: pl_module.all_gather(v).flatten(0, 1) for k, v in out.items()}
    return {k: v.float() if v.is_floating_point() else v for k, v in out.items()}


def _ranks(queries, v) -> torch.Tensor:
    """1-based rank of each query's target among unique val targets; own seed excluded."""
    ids, first = torch.unique(v["target_id"], return_inverse=True)
    corpus = torch.zeros(len(ids), v["t"].shape[1], device=v["t"].device).index_copy_(0, first, v["t"])
    sims = queries @ corpus.T
    seed_col = (v["seed_id"][:, None] == ids[None])
    sims = sims.masked_fill(seed_col, float("-inf"))
    pos = sims.gather(1, first[:, None])
    return (sims > pos).sum(1) + 1


class RetrievalMetrics(L.Callback):
    """R@k / MRR / medR on the val corpus, plus the instruction ablation and a seed-NN reference."""

    def __init__(self, ks=(1, 10, 50)):
        self.ks = ks

    def on_validation_epoch_end(self, trainer, pl_module):
        if not pl_module.val_outputs:
            return
        v = _gathered(pl_module)
        r, r_shuf, r_seed = _ranks(v["q"], v), _ranks(v["q_shuffled"], v), _ranks(v["seed_t"], v)
        logs = {f"val/R@{k}": (r <= k).float().mean() for k in self.ks}
        logs |= {"val/MRR": (1 / r.float()).mean(), "val/medR": r.float().median(),
                 "val/R@10_shuffled": (r_shuf <= 10).float().mean(),
                 "val/R@10_seed_nn": (r_seed <= 10).float().mean(),
                 "val/corpus_size": v["target_id"].unique().numel() * torch.ones((), device=pl_module.device)}
        logs["val/instruction_gap"] = logs["val/R@10"] - logs["val/R@10_shuffled"]
        pl_module.log_dict(logs, sync_dist=True)  # identical on every rank (gathered); silences the DDP warning


class EmbeddingPlots(L.Callback):
    """Cosine-similarity histograms + t-SNE of queries vs targets, logged to wandb (rank 0)."""

    def __init__(self, tsne_points: int = 1000, every_n_val: int = 1):
        self.tsne_points, self.every, self.calls = tsne_points, every_n_val, 0

    def on_validation_epoch_end(self, trainer, pl_module):
        if not pl_module.val_outputs or trainer.sanity_checking:
            return
        self.calls += 1
        v = _gathered(pl_module)                                  # collective: every rank must call it
        if not trainer.is_global_zero or self.calls % self.every or not hasattr(trainer.logger, "log_image"):
            return
        q, t = v["q"].cpu(), v["t"].cpu()
        sims = {"positive  cos(q_i, t_i)": (q * t).sum(1),
                "shuffled instruction": (v["q_shuffled"].cpu() * t).sum(1),
                "seed -> target  cos(s_i, t_i)": (v["seed_t"].cpu() * t).sum(1),
                "random pair  cos(q_i, t_j)": (q * t.roll(1, 0)).sum(1)}
        fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
        for name, s in sims.items():
            ax.hist(s.numpy(), bins=60, alpha=0.5, density=True, label=name)
        ax.set(xlabel="cosine similarity", ylabel="density")
        ax.legend(fontsize=7, frameon=False)
        images = {"val/similarity": fig}

        n = min(self.tsne_points, len(q))
        if n >= 10:
            from sklearn.manifold import TSNE
            xy = TSNE(perplexity=min(30, n // 3), init="pca", random_state=0).fit_transform(
                torch.cat([q[:n], t[:n]]).numpy())
            fig2, ax2 = plt.subplots(figsize=(5, 5), constrained_layout=True)
            ax2.scatter(*xy[:n].T, s=4, alpha=0.6, label="query (seed + instruction)")
            ax2.scatter(*xy[n:].T, s=4, alpha=0.6, label="target audio")
            ax2.set(xticks=[], yticks=[], title=f"t-SNE, step {trainer.global_step}")
            ax2.legend(fontsize=7, frameon=False, markerscale=3)
            images["val/tsne"] = fig2
        for key, f in images.items():
            trainer.logger.log_image(key=key, images=[f], step=trainer.global_step)
            plt.close(f)
