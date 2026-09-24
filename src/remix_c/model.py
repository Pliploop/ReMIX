"""ReMIX-C model: MuQ-MuLan towers + fusion transformer, trained by a pluggable objective.

  q = objective.query(fusion([CLS] + seed_tower(seed) + instruction_tower(instr)))
  t = target_tower(target)                      # separate MuQ copy -> mean-pool -> MuLan audio head
  retrieval score = cos(q, t)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode
from transformers import get_cosine_schedule_with_warmup


class _Tower(nn.Module):
    """freeze=True: no grads, always eval mode, forward under no_grad."""

    def __init__(self, freeze: bool):
        super().__init__()
        self.freeze = freeze

    def finish(self):
        self.requires_grad_(not self.freeze)
        return self

    def train(self, mode: bool = True):
        return super().train(mode and not self.freeze)

    def grad(self):
        return torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze)


class AudioTower(_Tower):
    """MuQ backbone -> proj (1024->768) -> MuLan audio transformer, per 10 s window; windows
    concatenated in time. grad_checkpoint: recompute each window in backward (non-reentrant)."""

    def __init__(self, audio, head: nn.Module | None = None, freeze: bool = False, grad_checkpoint: bool = False):
        super().__init__(freeze)
        self.grad_checkpoint = grad_checkpoint
        self.backbone, self.proj, self.transformer, self.layer = audio.model, audio.proj, audio.transformer, audio.use_layer_idx
        self.head = head                                          # MuLan audio_to_latents, 768 -> 512
        self.finish()

    def _window(self, w):                                        # (B, S) -> (B, T, 768)
        h = self.backbone(w, output_hidden_states=True).hidden_states[self.layer]
        return self.transformer(self.proj(h))

    def tokens(self, wav):                                       # (B, W, S) -> (B, W*T, 768)
        with self.grad():
            ckpt = self.grad_checkpoint and torch.is_grad_enabled()
            return torch.cat([checkpoint(self._window, w, use_reentrant=False) if ckpt else self._window(w)
                              for w in wav.unbind(1)], 1)

    def forward(self, wav):                                      # pooled, unnormalised (B, 512)
        with self.grad():
            return self.head(self.tokens(wav).mean(1))


class TextTower(_Tower):
    """XLM-R -> MuLan text transformer (8L). Returns tokens (B, L, 768) and keep-mask (B, L)."""

    def __init__(self, mulan, freeze: bool = False, max_len: int = 128):
        super().__init__(freeze)
        t = mulan.text
        self.tokenizer, self.backbone, self.proj, self.transformer = t.tokenizer, t.model, t.proj, t.transformer
        self.max_len = max_len
        self.finish()

    def forward(self, texts: List[str]):
        enc = self.tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True,
                             max_length=self.max_len).to(self.backbone.device)
        mask = enc["attention_mask"].bool()
        with self.grad():
            h = self.backbone(**enc).last_hidden_state
            return self.transformer(self.proj(h), mask=mask), mask


class Fusion(nn.Module):
    """[CLS] + audio tokens + text tokens (each linearly projected to `dim`) -> pre-norm
    TransformerEncoder -> CLS -> embed_dim."""

    def __init__(self, in_dim: int = 768, dim: int = 512, depth: int = 6, heads: int = 8,
                 dropout: float = 0.1, embed_dim: int = 512):
        super().__init__()
        self.audio_in, self.text_in = nn.Linear(in_dim, dim), nn.Linear(in_dim, dim)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, embed_dim))
        # ponytail: no positional embedding across windows (MuQ tokens are position-aware
        # within a window); add a learned one if window order turns out to matter.

    def forward(self, audio, text, text_mask):
        B, Ta = audio.shape[:2]
        x = torch.cat([self.cls.expand(B, -1, -1), self.audio_in(audio), self.text_in(text)], 1)
        pad = torch.cat([torch.zeros(B, 1 + Ta, dtype=torch.bool, device=x.device), ~text_mask], 1)
        return self.out(self.encoder(x, src_key_padding_mask=pad)[:, 0])


class RemixC(L.LightningModule):
    def __init__(self, model: DictConfig, objective: DictConfig, optim: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        from muq import MuQMuLan
        mulan = MuQMuLan.from_pretrained(model.mulan_id).mulan_module
        self.seed_tower = AudioTower(mulan.audio, **model.seed_tower)
        # second pretrained instance (MuQ's weight_norm blocks deepcopy): free to drift from the seed tower
        target = MuQMuLan.from_pretrained(model.mulan_id).mulan_module
        self.target_tower = AudioTower(target.audio, target.audio_to_latents, **model.target_tower)
        self.instruction_tower = TextTower(mulan, **model.instruction_tower)
        self.fusion = Fusion(in_dim=mulan.audio.dim, embed_dim=model.embed_dim, **model.fusion)
        self.objective = hydra.utils.instantiate(objective)
        self.val_outputs: List[Dict[str, torch.Tensor]] = []

    # ------------------------------------------------------------ embeddings
    def encode_query(self, seed_wav, instructions, seed_tokens=None):
        a = self.seed_tower.tokens(seed_wav) if seed_tokens is None else seed_tokens
        return self.fusion(a, *self.instruction_tower(instructions))

    def encode_target(self, wav):
        return self.target_tower(wav)

    def score_query(self, q):                                    # retrieval-space query (L2-normalised)
        return self.objective.query(q)

    # ------------------------------------------------------------ training
    def _gather(self, x):
        if self.trainer.world_size == 1:
            return x
        return self.all_gather(x, sync_grads=x.is_floating_point()).flatten(0, 1)

    def _swaps(self, batch, K):
        """K other batch items per anchor; invalid if same target clip or same (normalised) instruction."""
        B = len(batch["instruction"])
        j = (torch.arange(B)[:, None] + torch.randperm(B - 1)[:K][None] + 1) % B
        h = torch.tensor([hash(s.strip().lower()) for s in batch["instruction"]])
        tid = batch["target_id"].cpu()
        valid = (h[j] != h[:, None]) & (tid[j] != tid[:, None])
        return j.to(self.device), valid.to(self.device)

    def training_step(self, batch, _):
        a = self.seed_tower.tokens(batch["seed"])
        text, mask = self.instruction_tower(batch["instruction"])
        q, t = self.fusion(a, text, mask), self.encode_target(batch["target"])
        K = min(self.objective.n_swap, len(q) - 1)
        q_swap = valid = None
        if K > 0:
            j, valid = self._swaps(batch, K)
            q_swap = self.fusion(a.repeat_interleave(K, 0), text[j.flatten()], mask[j.flatten()]).view(len(q), K, -1)
        out = self.objective(q=q, t=t, target_ids=batch["target_id"], gather=self._gather,
                             offset=self.global_rank * len(q), q_swap=q_swap, swap_valid=valid)
        self.log_dict({f"train/{k}": v for k, v in out.items()}, prog_bar=True, batch_size=len(q))
        self.log("train/peak_mem_gb", torch.cuda.max_memory_allocated() / 2**30, batch_size=len(q))
        return out["loss"]

    def on_validation_epoch_start(self):
        self.val_outputs.clear()

    def validation_step(self, batch, _):
        a = self.seed_tower.tokens(batch["seed"])
        instr = list(batch["instruction"])
        shuffled = instr[1:] + instr[:1]                         # instruction ablation: neighbour's instruction
        self.val_outputs.append(dict(
            q=self.score_query(self.encode_query(None, instr, a)),
            q_shuffled=self.score_query(self.encode_query(None, shuffled, a)),
            t=nn.functional.normalize(self.encode_target(batch["target"]), dim=-1),
            seed_t=nn.functional.normalize(self.encode_target(batch["seed"]), dim=-1),
            seed_id=batch["seed_id"], target_id=batch["target_id"]))

    def configure_optimizers(self):
        o = self.hparams.optim
        towers = [p for n, p in self.named_parameters() if p.requires_grad and n.split(".")[0].endswith("_tower")]
        heads = [p for n, p in self.named_parameters() if p.requires_grad and not n.split(".")[0].endswith("_tower")]
        groups = [g for g in ({"params": towers, "lr": o.lr_towers}, {"params": heads, "lr": o.lr_heads}) if g["params"]]
        opt = torch.optim.AdamW(groups, weight_decay=o.weight_decay)
        sched = get_cosine_schedule_with_warmup(opt, o.warmup_steps, o.total_steps)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


# ---------------------------------------------------------------- ReMIX-B harness adapter
class RemixCBaseline:
    """`prepare(corpus)` / `rank_all(queries, k)` contract of jamendo_instruct.benchmark.
    Checkpoint from $REMIX_C_CKPT (run_remix_b.py --ckpt sets it)."""

    name = "remix_c"

    def __init__(self, ckpt: str | None = None, batch_size: int = 32, num_workers: int = 8):
        self.ckpt = ckpt or os.environ["REMIX_C_CKPT"]
        self.bs, self.nw = batch_size, num_workers
        self.flops = 0

    @torch.no_grad()
    def _embed(self, clip_ids, fn):
        """fn(wav batch, slice of clip_ids) -> embeddings; FLOPs counted exactly."""
        from .data import ClipSet
        out, i = [], 0
        for wav in DataLoader(ClipSet(clip_ids, self.clips), batch_size=self.bs, num_workers=self.nw):
            with torch.autocast("cuda", dtype=torch.bfloat16), FlopCounterMode(display=False) as fc:
                out.append(fn(wav.cuda(), slice(i, i + len(wav))).float())
            self.flops += fc.get_total_flops()
            i += len(wav)
        return torch.cat(out)

    def prepare(self, corpus) -> None:
        from .data import Clips
        self.model = RemixC.load_from_checkpoint(self.ckpt, map_location="cuda").eval()
        dm = torch.load(self.ckpt, map_location="cpu", weights_only=False)["datamodule_hyper_parameters"]
        self.clips = Clips(pd.read_parquet(Path(dm["index_dir"]) / "clips.parquet"), dm["n_windows"])
        self.corpus = corpus
        self.T = nn.functional.normalize(self._embed(corpus.ids, lambda wav, _: self.model.encode_target(wav)), dim=-1)

    def rank_all(self, queries, k: int) -> Dict[str, List[str]]:
        from jamendo_instruct.benchmark.baselines.base import batched_topk
        instr = [q.instruction for q in queries]
        Q = self._embed([q.seed_clip_id for q in queries],
                        lambda wav, s: self.model.score_query(self.model.encode_query(wav, instr[s])))
        exclude = [self.corpus.id2row.get(q.seed_clip_id, -1) for q in queries]
        preds = batched_topk(Q.cpu().numpy(), self.T.cpu().numpy(), self.corpus.ids, k, exclude)
        return {q.query_id: p for q, p in zip(queries, preds)}
