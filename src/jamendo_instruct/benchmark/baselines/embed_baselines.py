"""Composed / zero-shot baselines that embed new text at query time."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .base import Baseline, Corpus, Query, batched_topk, matmul_flops
from .embedders import GemmaText, MuLanText, _l2


def _exclude_rows(c: Corpus, queries: List[Query]) -> List[int]:
    return [c.id2row.get(q.seed_clip_id, -1) for q in queries]


def _seed_caption(c: Corpus, q: Query) -> str:
    return (c.meta.get(q.seed_clip_id, {}) or {}).get("caption", "")


class _GemmaTextQuery(Baseline):
    """Embed a per-query text with EmbeddingGemma, search the corpus caption space."""
    name = "gemma_text"

    def prepare(self, corpus: Corpus) -> None:
        self.c = corpus
        self.emb = GemmaText()

    def _texts(self, queries: List[Query]) -> List[str]:
        raise NotImplementedError

    def rank_all(self, queries: List[Query], k: int) -> Dict[str, List[str]]:
        Q = self.emb.encode(self._texts(queries))
        lists = batched_topk(Q, self.c.text, self.c.ids, k, _exclude_rows(self.c, queries))
        self.flops = self.emb.flops + matmul_flops(len(queries), *self.c.text.shape)
        return {q.query_id: lst for q, lst in zip(queries, lists)}


class InstructionText(_GemmaTextQuery):
    """Instruction only (ignores the seed content), text retrieval."""
    name = "instruction_text"

    def _texts(self, queries):
        return [q.instruction for q in queries]


class CaptionPlusInstruction(_GemmaTextQuery):
    """Naive composed: seed caption + instruction, text retrieval ('won't work')."""
    name = "caption_plus_instruction"

    def _texts(self, queries):
        return [f"{_seed_caption(self.c, q)} {q.instruction}".strip() for q in queries]


class MuLanZeroShot(Baseline):
    """Caption+instruction through the MuQ-MuLan text tower, search corpus audio."""
    name = "mulan_zeroshot"

    def prepare(self, corpus: Corpus) -> None:
        self.c = corpus
        self.emb = MuLanText()

    def rank_all(self, queries: List[Query], k: int) -> Dict[str, List[str]]:
        texts = [f"{_seed_caption(self.c, q)} {q.instruction}".strip() for q in queries]
        Q = self.emb.encode(texts)
        lists = batched_topk(Q, self.c.audio, self.c.ids, k, _exclude_rows(self.c, queries))
        self.flops = self.emb.flops + matmul_flops(len(queries), *self.c.audio.shape)
        return {q.query_id: lst for q, lst in zip(queries, lists)}


class LateFusion(Baseline):
    """Composed: seed audio + w * MuLan(instruction), search corpus audio."""
    name = "late_fusion"

    def __init__(self, w: float = 0.5):
        self.w = w

    def prepare(self, corpus: Corpus) -> None:
        self.c = corpus
        self.emb = MuLanText()

    def rank_all(self, queries: List[Query], k: int) -> Dict[str, List[str]]:
        T = self.emb.encode([q.instruction for q in queries])                 # (Nq, 512) joint
        A = np.stack([self.c.audio[self.c.id2row[q.seed_clip_id]]
                      if q.seed_clip_id in self.c.id2row else np.zeros(self.c.audio.shape[1], np.float32)
                      for q in queries])
        Q = _l2(A + self.w * T)
        lists = batched_topk(Q, self.c.audio, self.c.ids, k, _exclude_rows(self.c, queries))
        self.flops = self.emb.flops + matmul_flops(len(queries), *self.c.audio.shape)
        return {q.query_id: lst for q, lst in zip(queries, lists)}


REGISTRY = {b.name: b for b in [InstructionText, CaptionPlusInstruction, MuLanZeroShot, LateFusion]}
