"""Training objectives. Both expose the same interface:

  objective(q, t, target_ids, q_swap, swap_valid, gather, offset) -> {"loss": ..., <parts>}
  objective.query(q) -> embedding compared (cosine) against target embeddings at retrieval
  objective.n_swap   -> number of instruction-swapped queries the model must build per anchor
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class Contrastive(nn.Module):
    """InfoNCE over (query, target) pairs. For anchor i the candidates are q_i and its K
    instruction-swapped queries q_ik, each against every (gathered) target; the only
    positive is (q_i, t_i). Negatives: changed target, changed instruction, or both."""

    def __init__(self, temperature: float = 0.07, learn_temperature: bool = True, n_swap: int = 4):
        super().__init__()
        self.n_swap = n_swap
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / temperature)), requires_grad=learn_temperature)

    def query(self, q):
        return F.normalize(q, dim=-1)

    def forward(self, q, t, target_ids, gather, offset, q_swap=None, swap_valid=None):
        B = len(q)
        T, ids = F.normalize(gather(t), dim=-1), gather(target_ids)
        Q = q[:, None] if q_swap is None else torch.cat([q[:, None], q_swap], 1)       # (B, 1+K, D)
        logits = self.logit_scale.exp().clamp(max=100) * F.normalize(Q, dim=-1) @ T.T  # (B, 1+K, N)
        pos = torch.arange(B, device=q.device) + offset
        dup = target_ids[:, None] == ids[None]                                          # other items, same target clip
        dup[torch.arange(B), pos] = False
        logits = logits.masked_fill(dup[:, None], float("-inf"))
        if q_swap is not None:                                                          # identical instruction / target
            logits[:, 1:] = logits[:, 1:].masked_fill(~swap_valid[..., None], float("-inf"))
        loss = F.cross_entropy(logits.flatten(1), pos)                                  # row 0 -> flat index == pos
        return {"loss": loss, "temperature": 1 / self.logit_scale.exp().detach()}


def sigreg(x: torch.Tensor, n_slices: int = 256, knots: int = 17, t_max: float = 5.0) -> torch.Tensor:
    """SIGReg (LeJEPA): Epps-Pulley statistic of random 1-D projections of x vs N(0,1),
    i.e. weighted L2 distance between empirical and Gaussian characteristic functions."""
    x = x.float()
    A = torch.randn(x.shape[1], n_slices, device=x.device)
    A = A / A.norm(dim=0)
    t = torch.linspace(-t_max, t_max, knots, device=x.device)
    xt = (x @ A)[..., None] * t                                  # (N, S, K)
    phi = torch.exp(-0.5 * t ** 2)                               # N(0,1) char. function = weight
    err = (xt.cos().mean(0) - phi) ** 2 + xt.sin().mean(0) ** 2  # |ecf - phi|^2, (S, K)
    return (torch.trapezoid(err * phi, t, dim=-1) * len(x)).mean()


def mlp(dim: int, width: int, depth: int, dropout: float) -> nn.Sequential:
    layers, d = [], dim
    for _ in range(depth - 1):
        layers += [nn.Linear(d, width), nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(dropout)]
        d = width
    return nn.Sequential(*layers, nn.Linear(d, dim))


class LeJEPA(nn.Module):
    """LeVLJEPA-style: (1-lq-lt)*||h(q) - t||^2 + lq*SIGReg(q) + lt*SIGReg(t), no stop-grad."""

    n_swap = 0

    def __init__(self, dim: int = 512, pred_width: int = 2048, pred_depth: int = 4, dropout: float = 0.1,
                 lambda_q: float = 0.01, lambda_t: float = 0.01, n_slices: int = 256):
        super().__init__()
        self.predictor = mlp(dim, pred_width, pred_depth, dropout)
        self.lq, self.lt, self.n_slices = lambda_q, lambda_t, n_slices

    def query(self, q):
        return F.normalize(self.predictor(q), dim=-1)

    def forward(self, q, t, gather, **_):
        pred = (self.predictor(q) - t).pow(2).sum(-1).mean()
        sq, st = sigreg(gather(q), self.n_slices), sigreg(gather(t), self.n_slices)
        loss = (1 - self.lq - self.lt) * pred + self.lq * sq + self.lt * st
        return {"loss": loss, "pred": pred.detach(), "sigreg_q": sq.detach(), "sigreg_t": st.detach()}


def _demo():
    torch.manual_seed(0)
    g = torch.randn(4096, 64)
    assert sigreg(g) < sigreg(g * 0.1) and sigreg(g) < sigreg(g + 3), "SIGReg must favour N(0, I)"
    B, K, D = 8, 3, 16
    q, t, ids = torch.randn(B, D), torch.randn(B, D), torch.arange(B)
    c = Contrastive(n_swap=K)
    ok = c(q, t, ids, gather=lambda x: x, offset=0, q_swap=torch.randn(B, K, D), swap_valid=torch.ones(B, K, dtype=torch.bool))
    perfect = c(t, t, ids, gather=lambda x: x, offset=0)
    assert perfect["loss"] < ok["loss"]
    ids[1] = ids[0]                                              # duplicate target must not be a negative
    assert torch.isfinite(c(q, t, ids, gather=lambda x: x, offset=0)["loss"])
    lj = LeJEPA(dim=D, pred_width=32)(q, t, gather=lambda x: x)
    assert torch.isfinite(lj["loss"])
    print("remix_c.objectives self-check OK")


if __name__ == "__main__":
    _demo()
