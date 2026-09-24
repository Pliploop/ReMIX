"""Answer key + scoring for ReMIX-B (docs/evaluation.md).

Two public calls, everything else is plumbing:

    grade_pool(pool)            -> {query_id: [clip_id ranked by grade]}   # answer key
    evaluate(predictions, pool) -> Report                                   # graded metrics

`pool` is either a qrels dict ``{qid: {clip: grade}}`` or a path/glob to the
relevance-pool shards (``chain_step_relevance_pools.shard*.jsonl``). Predictions
are ranked lists of clip IDs (best first): ``{qid: [clip, ...]}`` — or, for a
single query, just ``[clip, ...]``.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Union

import ir_measures
from ir_measures import AP, R, RR, nDCG

Qrels = Dict[str, Dict[str, int]]
Predictions = Union[Mapping[str, Sequence[str]], Sequence[str]]

POSITIVE_MIN = 3  # grade >= 3 (partial or better) is a binary "relevant"
DEFAULT_MEASURES = [nDCG @ 10, nDCG @ 100, AP(rel=POSITIVE_MIN),
                    R(rel=POSITIVE_MIN) @ 10, R(rel=POSITIVE_MIN) @ 100, RR(rel=POSITIVE_MIN)]


class Report(dict):
    """{measure_name: value}; ``print(report)`` shows an aligned table."""

    def __str__(self) -> str:
        w = max((len(k) for k in self), default=0)
        return "\n".join(f"{k:<{w}}  {v:.4f}" for k, v in self.items())


def _qid(rec: Mapping) -> str:
    return f"{rec.get('chain_id')}#{rec.get('turn_index')}"


def _read_qrels(pool: Union[Qrels, str, Path], min_grade: int = 1) -> Qrels:
    """Normalise `pool` to {qid: {clip: grade}}, keeping grades >= min_grade
    (unpooled / gated-0 clips are implicitly non-relevant)."""
    if isinstance(pool, Mapping):
        return {q: {c: int(g) for c, g in d.items() if int(g) >= min_grade} for q, d in pool.items()}
    files = sorted(glob.glob(str(pool))) if any(ch in str(pool) for ch in "*?[") else None
    if files is None:
        p = Path(pool)
        files = (sorted(str(x) for x in p.glob("chain_step_relevance_pools.shard*.jsonl"))
                 if p.is_dir() else [str(p)])
    if not files:
        raise FileNotFoundError(f"no pool shards at {pool}")
    qrels: Qrels = {}
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                d = qrels.setdefault(_qid(rec), {})
                for c in rec.get("candidates", []):
                    g = int(c.get("grade", 0) or 0)
                    if g >= min_grade:
                        d[str(c["clip_id"])] = g
    return qrels


def grade_pool(pool: Union[Qrels, str, Path], *, positive_min: int = POSITIVE_MIN) -> Dict[str, List[str]]:
    """Answer key: {query_id: [clip_id ranked by grade desc]} (ties broken by id)."""
    qrels = _read_qrels(pool, min_grade=positive_min)
    return {q: [c for c, _ in sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))]
            for q, d in qrels.items()}


def _as_run(predictions: Predictions) -> Dict[str, Dict[str, float]]:
    """Ranked ID lists -> ir_measures run (score = descending rank)."""
    if isinstance(predictions, Mapping):
        items = predictions.items()
    else:  # a single ranked list -> single anonymous query
        items = [("_q", list(predictions))]
    return {str(q): {str(c): float(-i) for i, c in enumerate(ranking)} for q, ranking in items}


def evaluate(predictions: Predictions, pool: Union[Qrels, str, Path],
             measures: Sequence = DEFAULT_MEASURES, positive_min: int = POSITIVE_MIN) -> Report:
    """Score ranked ID lists against the pool. Grades below `positive_min`
    (hard/soft-fail = failures) count as non-relevant; grades 3-6 are the graded
    gains, so a grade-ordered ranking is the nDCG=1 ideal."""
    qrels = _read_qrels(pool, min_grade=positive_min)
    run = _as_run(predictions)
    agg = ir_measures.calc_aggregate(list(measures), qrels, run)
    return Report({str(m): float(agg[m]) for m in measures})


def _demo() -> None:
    # one query: clipA grade 6 (best), clipB grade 3, clipC grade 1; clipD non-rel.
    pool = {"q1": {"clipA": 6, "clipB": 3, "clipC": 1, "clipD": 0}}
    assert grade_pool(pool) == {"q1": ["clipA", "clipB"]}, grade_pool(pool)  # grade>=3, ranked
    perfect = evaluate({"q1": ["clipA", "clipB", "clipC", "clipD"]}, pool)
    worst = evaluate({"q1": ["clipD", "clipC", "clipB", "clipA"]}, pool)
    assert perfect["nDCG@10"] == 1.0, perfect
    assert perfect["nDCG@10"] > worst["nDCG@10"], (perfect, worst)
    assert perfect["R(rel=3)@10"] == 1.0, perfect
    assert perfect["RR(rel=3)"] == 1.0, perfect
    print("graded_pool self-check OK")
    print(perfect)


if __name__ == "__main__":
    _demo()
