#!/usr/bin/env python
"""Re-grade a subsample of the ReMIX-B relevance pool by listening.

The pool's grades come from a text-only judge (captions, tags, semantic deltas).
This grades the SAME candidates with an audio-language model that hears the seed
and the candidate (plus the instruction), to measure how much the text-only
labels bias the benchmark. Output: an alternative qrels file on the same 0-6
scale, plus the text grades for comparison.

  PYTHONPATH=src REMIX_LALM_TP=2 python scripts/audio_judge_pool.py --n 500 \
      --out results/remix_b/audio_judge/music4all_qwen3omni.json

Model/engine settings come from baselines/lalm.py (REMIX_LALM, REMIX_LALM_TP, REMIX_LALM_MEM).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

from jamendo_instruct.benchmark.baselines.base import load_queries
from jamendo_instruct.benchmark.graded_pool import _read_qrels

RUN_ROOT = "/gpfs/scratch/acw749/datasets/music4all_instruct/music4all_v1"
FOLDER = Path(RUN_ROOT) / "instructions_axis_focused_5"

# Same labels and grade mapping as the text judge (stages/relevance_pool.py QUALITY_GRADE),
# plus 0 because this judge also sees the similarity-gated candidates the text judge never graded.
JUDGE_SYS = (
    "You judge how well a candidate track answers a composed music-retrieval query: a seed track plus an "
    "edit instruction describing how the wanted track differs from the seed. The first audio is the seed, "
    "the second is the candidate. Listen to both. Reply with a single integer:\n"
    "6 exact: indistinguishable from the intended edited track.\n"
    "5 strong: satisfies ALL requested changes and keeps what the instruction says to keep.\n"
    "4 good: satisfies the requested change and stays compatible with the seed, but misses a minor preservation or is a looser match.\n"
    "3 partial: satisfies part of the requested change and misses another part.\n"
    "2 soft fail: surface traits fit, but the requested change is not achieved.\n"
    "1 hard fail: fails the main requested change, though still musically on-topic.\n"
    "0 unrelated: clearly unrelated to the query.\n"
    "Integer only.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=500, help="queries to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ["REMIX_RUN_ROOT"] = RUN_ROOT
    from jamendo_instruct.benchmark.baselines import lalm      # after REMIX_RUN_ROOT is set

    queries = load_queries(FOLDER / "benchmark")
    queries = random.Random(a.seed).sample(queries, a.n)
    text = _read_qrels(str(FOLDER / "relevance_pool"), min_grade=0)   # every pooled candidate, gated 0s included
    pairs = [(q, c) for q in queries for c in sorted(text.get(q.query_id, {}))]
    print(f"{len(queries)} queries | {len(pairs):,} (query, candidate) pairs", flush=True)

    aud, eng = lalm._audio(), lalm._engine()
    aud.load([q.seed_clip_id for q, _ in pairs] + [c for _, c in pairs])
    outs = eng.generate(JUDGE_SYS, [[q.seed_clip_id, c] for q, c in pairs],
                        [f"Edit instruction: {q.instruction}\n\nGrade (0-6):" for q, _ in pairs], max_tokens=4)
    audio: dict = {}
    for (q, c), o in zip(pairs, outs):
        m = re.search(r"[0-6]", o or "")
        audio.setdefault(q.query_id, {})[c] = int(m.group()) if m else -1   # -1 = unparsable
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({
        "model": lalm.MODEL_ID, "n_queries": len(queries), "seed": a.seed, "tflops": round(eng.flops / 1e12, 1),
        "audio": audio, "text": {q.query_id: text.get(q.query_id, {}) for q in queries}}))
    bad = sum(g < 0 for d in audio.values() for g in d.values())
    print(f"wrote {a.out} ({bad} unparsable)", flush=True)


if __name__ == "__main__":
    main()
