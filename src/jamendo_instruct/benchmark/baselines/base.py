"""Common baseline API + corpus loader (docs/evaluation.md).

A baseline maps a query (seed + instruction) to a ranked list of corpus clip ids.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Protocol

import numpy as np


@dataclass
class Query:
    query_id: str
    seed_clip_id: str
    target_clip_id: str
    instruction: str
    instruction_history_aware: str = ""


@dataclass
class Corpus:
    ids: List[str]
    audio: np.ndarray                    # (N, Da) L2-normalised
    text: np.ndarray                     # (N, Dt) L2-normalised
    meta: Dict[str, dict]
    id2row: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.id2row = {c: i for i, c in enumerate(self.ids)}

    @classmethod
    def load(cls, bench_dir) -> "Corpus":
        d = Path(bench_dir)
        ids = json.loads((d / "corpus_ids.json").read_text())
        meta = {}
        for line in open(d / "corpus.jsonl"):
            line = line.strip()
            if line:
                r = json.loads(line); meta[r["clip_id"]] = r
        return cls(ids, np.load(d / "corpus_audio.npy"), np.load(d / "corpus_text.npy"), meta)


def load_queries(bench_dir) -> List[Query]:
    out = []
    for line in open(Path(bench_dir) / "queries.jsonl"):
        line = line.strip()
        if line:
            r = json.loads(line)
            out.append(Query(r["query_id"], r["seed_clip_id"], r.get("target_clip_id", ""),
                             r.get("instruction", ""), r.get("instruction_history_aware", "")))
    return out


class Baseline(Protocol):
    name: str
    def prepare(self, corpus: Corpus) -> None: ...
    def rank(self, query: Query, k: int) -> List[str]: ...


def rank_by_vector(qvec: np.ndarray, matrix: np.ndarray, ids: List[str], k: int, exclude: set) -> List[str]:
    """Top-k corpus ids by cosine (both L2-normalised), dropping `exclude`."""
    sims = matrix @ qvec
    k = min(k, len(ids))
    top = np.argpartition(-sims, min(k + len(exclude), len(ids) - 1))[: k + len(exclude)]
    top = top[np.argsort(-sims[top])]
    out = [ids[i] for i in top if ids[i] not in exclude]
    return out[:k]


def batched_topk(qmat: np.ndarray, matrix: np.ndarray, ids: List[str], k: int,
                 exclude_rows: List[int]) -> List[List[str]]:
    """One matmul for all queries: qmat (Nq,D) x matrix (N,D)^T -> top-k ids per row.
    `exclude_rows[i]` (or -1) is a corpus row to drop from query i (its own seed)."""
    sims = qmat @ matrix.T                       # (Nq, N)
    for i, er in enumerate(exclude_rows):
        if er is not None and er >= 0:
            sims[i, er] = -1e30
    k = min(k, sims.shape[1])
    part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    out = []
    for i in range(sims.shape[0]):
        order = part[i][np.argsort(-sims[i, part[i]])]
        out.append([ids[j] for j in order])
    return out
