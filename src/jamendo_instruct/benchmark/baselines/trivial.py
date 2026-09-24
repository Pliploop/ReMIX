"""Baselines needing no new embedding (reuse corpus vectors).

Each exposes both the per-query `rank` (the documented API) and a batched
`rank_all` (one matmul over all queries) that the harness prefers for speed.
"""
from __future__ import annotations

import random
from typing import Dict, List

import numpy as np

from .base import Baseline, Corpus, Query, batched_topk, matmul_flops, rank_by_vector


class _EmbBaseline(Baseline):
    """Rank by cosine to a query vector taken from a corpus row (audio or text)."""
    name = "emb"
    matrix_attr = "audio"     # which corpus matrix to search
    key = "seed_clip_id"      # which query clip supplies the query vector

    def prepare(self, corpus: Corpus) -> None:
        self.c = corpus
        self.M = getattr(corpus, self.matrix_attr)

    def _row(self, q: Query):
        return self.c.id2row.get(getattr(q, self.key))

    def rank(self, query: Query, k: int) -> List[str]:
        row = self._row(query)
        if row is None:
            return []
        return rank_by_vector(self.M[row], self.M, self.c.ids, k, {query.seed_clip_id})

    def rank_all(self, queries: List[Query], k: int) -> Dict[str, List[str]]:
        idx = [(i, self._row(q)) for i, q in enumerate(queries)]
        valid = [(i, r) for i, r in idx if r is not None]
        if not valid:
            return {q.query_id: [] for q in queries}
        rows = [r for _, r in valid]
        exclude = [self.c.id2row.get(queries[i].seed_clip_id, -1) for i, _ in valid]
        lists = batched_topk(self.M[rows], self.M, self.c.ids, k, exclude)
        self.flops = matmul_flops(len(valid), self.M.shape[0], self.M.shape[1])
        out = {q.query_id: [] for q in queries}
        for (i, _), lst in zip(valid, lists):
            out[queries[i].query_id] = lst
        return out


class SeedAudioNN(_EmbBaseline):
    """Ignores the instruction: nearest corpus clips to the seed's audio."""
    name = "seed_audio_nn"; matrix_attr = "audio"; key = "seed_clip_id"


class SeedCaptionNN(_EmbBaseline):
    """Ignores the instruction: nearest corpus clips to the seed's caption embedding."""
    name = "seed_caption_nn"; matrix_attr = "text"; key = "seed_clip_id"


class TargetCaptionOracle(_EmbBaseline):
    """Upper bound: rank by text similarity to the *true target*'s caption embedding."""
    name = "target_caption_oracle"; matrix_attr = "text"; key = "target_clip_id"


class Random(Baseline):
    name = "random"

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def prepare(self, corpus: Corpus) -> None:
        self.ids = list(corpus.ids)

    def rank(self, query: Query, k: int) -> List[str]:
        pool = [c for c in self.ids if c != query.seed_clip_id]
        self._rng.shuffle(pool)
        return pool[:k]

    def rank_all(self, queries: List[Query], k: int) -> Dict[str, List[str]]:
        self.flops = 0
        return {q.query_id: self.rank(q, k) for q in queries}


class BM25(Baseline):
    """Lexical sparse retrieval: BM25 over corpus captions, query = seed caption + instruction."""
    name = "bm25"

    @staticmethod
    def _tok(s: str) -> List[str]:
        return "".join(ch.lower() if ch.isalnum() else " " for ch in s).split()

    def prepare(self, corpus: Corpus) -> None:
        from rank_bm25 import BM25Okapi
        self.c = corpus
        self.bm25 = BM25Okapi([self._tok(corpus.meta[cid].get("caption", "")) for cid in corpus.ids])
        self.flops = 0  # lexical; not comparable to the neural FLOPs

    def rank(self, query: Query, k: int) -> List[str]:
        q = self._tok(f"{(self.c.meta.get(query.seed_clip_id, {}) or {}).get('caption','')} {query.instruction}")
        scores = self.bm25.get_scores(q)
        order = np.argsort(-scores)
        out = [self.c.ids[i] for i in order if self.c.ids[i] != query.seed_clip_id]
        return out[:k]

    def rank_all(self, queries: List[Query], k: int) -> Dict[str, List[str]]:
        self.flops = 0
        return {q.query_id: self.rank(q, k) for q in queries}


REGISTRY = {b.name: b for b in [Random, SeedAudioNN, SeedCaptionNN, TargetCaptionOracle, BM25]}
