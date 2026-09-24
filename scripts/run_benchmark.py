#!/usr/bin/env python
"""Run a ReMIX-B baseline and score it against the graded pool (docs/evaluation.md).

  PYTHONPATH=src python scripts/run_benchmark.py --dataset music4all --baseline seed_audio_nn
  PYTHONPATH=src python scripts/run_benchmark.py --dataset music4all --baseline all
"""
from __future__ import annotations

import argparse
from pathlib import Path

from jamendo_instruct.benchmark import evaluate
from jamendo_instruct.benchmark.graded_pool import _read_qrels
from jamendo_instruct.benchmark.baselines.base import Corpus, load_queries
from jamendo_instruct.benchmark.baselines.trivial import REGISTRY

DATASETS = {
    "music4all": "/gpfs/scratch/acw749/datasets/music4all_instruct/music4all_v1",
    "mtg_jamendo": "/gpfs/scratch/acw749/datasets/mtg_jamendo_instruct/v1",
}
FOLDER = "instructions_axis_focused_5"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(DATASETS), default="music4all")
    ap.add_argument("--baseline", default="all", help="baseline name or 'all'")
    ap.add_argument("--k", type=int, default=200)
    args = ap.parse_args()

    root = Path(DATASETS[args.dataset]) / FOLDER
    bench, pool = root / "benchmark", root / "relevance_pool"
    corpus = Corpus.load(bench)
    queries = load_queries(bench)
    qrels = _read_qrels(str(pool), min_grade=3)   # read the pool once, reuse for every baseline
    print(f"corpus {len(corpus.ids):,} clips | {len(queries):,} queries | {len(qrels):,} judged\n", flush=True)

    names = list(REGISTRY) if args.baseline == "all" else [args.baseline]
    rows = []
    for name in names:
        b = REGISTRY[name]()
        b.prepare(corpus)
        if hasattr(b, "rank_all"):
            preds = b.rank_all(queries, args.k)
        else:
            preds = {q.query_id: b.rank(q, args.k) for q in queries}
        rep = evaluate(preds, qrels)
        rows.append((name, rep))
        print(f"[{name}]\n{rep}\n", flush=True)

    if len(rows) > 1:
        metrics = list(rows[0][1])
        w = max(len(n) for n, _ in rows)
        print("  " + " ".join(f"{m:>12}" for m in ["baseline".ljust(w)] + metrics))
        for n, rep in rows:
            print("  " + f"{n:<{w}} " + " ".join(f"{rep[m]:>12.4f}" for m in metrics))


if __name__ == "__main__":
    main()
