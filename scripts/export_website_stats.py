#!/usr/bin/env python
"""Export dataset + validation statistics for the website as plain JSON.

The site renders these with Recharts rather than shipping images, so the plots
stay crisp, theme with the page, and are interactive. This script therefore emits
*numbers*, never figures.

Additive and read-only. It reuses the parquet frames that
``scripts/paper_data_stats.py`` already caches under ``paper/cache/<label>/``,
so the site's numbers come from the same computation as the paper's figures.
Build that cache first if it is missing:

    python scripts/paper_data_stats.py --run-root <root> --label <label>

Usage:
    python scripts/export_website_stats.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd

REPO = Path(__file__).resolve().parents[1]

# Keep in sync with scripts/paper_data_stats.py (GENRE_COLORS) and website/src/theme.js.
TOP_GENRES = 8

DATASETS: List[Dict[str, Any]] = [
    {
        "key": "music4all",
        "label": "Music4All",
        "validation": "/gpfs/scratch/acw749/datasets/music4all_instruct/music4all_v1/instructions_axis_focused_5/validation",
        "judges": {"Qwen3.6-27B": "llm_ratings.jsonl", "Gemma-4-31B": "llm_ratings_gemma_full.jsonl"},
    },
    {
        "key": "mtg_jamendo",
        "label": "MTG-Jamendo",
        "validation": "/gpfs/scratch/acw749/datasets/mtg_jamendo_instruct/v1/instructions_axis_focused_5/validation",
        "judges": {"Qwen3.6-27B": "llm_ratings_qwen_full.jsonl", "Gemma-4-31B": "llm_ratings_gemma_full.jsonl"},
    },
]

QUESTION_LABELS = {
    "meaningful_change": "Meaningful change",
    "target_follows": "Target follows",
    "source_support": "Source supported",
    "source_compatible": "Source compatible",
    "conservation_supported": "Keeps what it says",
    "edit_specificity": "Written as an edit",
    "clarity_actionability": "Clear",
    "overall_validity": "Overall valid",
}

ACCEPT_THRESHOLD = 4  # matches the validation gate


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_list(v: Any) -> List[str]:
    """Parquet round-trips list columns as ndarray/list/str depending on writer."""
    if v is None:
        return []
    if isinstance(v, str):
        return [x.strip() for x in v.split(",") if x.strip()]
    try:
        return [str(x) for x in v if str(x).strip()]
    except TypeError:
        return []


def _explode(series: pd.Series) -> Counter:
    c: Counter = Counter()
    for v in series:
        c.update(_as_list(v))
    return c


def _nice(s: str) -> str:
    return s.replace("_", " ").strip().capitalize()


def _hist(values: Sequence[float], bins: int, lo: float, hi: float) -> List[Dict[str, float]]:
    if hi <= lo:
        return []
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        i = int((v - lo) / width)
        if i == bins:
            i -= 1  # right edge belongs to the last bin
        if 0 <= i < bins:
            counts[i] += 1
    return [
        {"x": round(lo + (i + 0.5) * width, 4), "lo": round(lo + i * width, 4), "count": counts[i]}
        for i in range(bins)
    ]


# --------------------------------------------------------------------------- #
# dataset stats (from the paper's parquet cache)
# --------------------------------------------------------------------------- #
def dataset_stats(cache_dir: Path, key: str) -> Dict[str, Any] | None:
    steps_p = cache_dir / key / "steps.parquet"
    corpus_p = cache_dir / key / "corpus.parquet"
    if not steps_p.is_file() or not corpus_p.is_file():
        print(f"  ! no cache for {key} at {cache_dir / key}", file=sys.stderr)
        return None

    steps = pd.read_parquet(steps_p)
    corpus = pd.read_parquet(corpus_p)
    unique = steps.drop_duplicates(subset=["chain_id", "turn_index"])

    # Genre: top N by clip count, everything else folded into "Other".
    gc = Counter(corpus["genre"].fillna("Other"))
    top = gc.most_common(TOP_GENRES)
    other = sum(v for k, v in gc.items() if k not in dict(top))
    genre = [{"name": k, "value": int(v)} for k, v in top]
    if other:
        genre.append({"name": "Other", "value": int(other)})

    lengths = unique.groupby("chain_id").size()
    chain_length = [
        {"steps": int(k), "chains": int(v)} for k, v in sorted(Counter(lengths).items())
    ]

    change = _explode(unique["change_axes"])
    preserve = _explode(unique["preservation_axes"])
    axes_names = [a for a, _ in change.most_common()]
    axes = [
        {"axis": _nice(a), "changed": int(change.get(a, 0)), "preserved": int(preserve.get(a, 0))}
        for a in axes_names
    ]

    num_axes = [
        {"n": int(k), "steps": int(v)}
        for k, v in sorted(Counter(unique["change_axis_count"].dropna().astype(int)).items())
    ]

    instructed = steps[steps["has_instruction"]]

    return {
        "key": key,
        "overview": {
            "clips": int(corpus["clip_id"].nunique()),
            "artists": int(corpus["artist_id"].nunique()),
            "chains": int(unique["chain_id"].nunique()),
            "steps": int(len(unique)),
            "variants": int(len(steps)),
            "median_caption_words": int(corpus["caption_words"].median()),
            "median_instruction_words": int(instructed["history_unaware_words"].median())
            if len(instructed)
            else 0,
        },
        "genre": genre,
        "chain_length": chain_length,
        "axes": axes,
        "num_axes": num_axes,
        "transition_score": _hist(unique["transition_score"].dropna().tolist(), 24, 0.0, 1.0),
        "instruction_words": _hist(
            instructed["history_unaware_words"].dropna().tolist(), 20, 0.0, 40.0
        ),
        "hardness": [
            {"name": _nice(str(k)), "value": int(v)}
            for k, v in Counter(unique["hardness"].fillna("unknown")).most_common()
        ],
    }


# --------------------------------------------------------------------------- #
# validation stats (from the LLM judge ratings)
# --------------------------------------------------------------------------- #
def _iter_ratings(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _key(rec: Dict[str, Any]) -> Tuple[str, int, int, str]:
    return (
        rec.get("chain_id", ""),
        int(rec.get("turn_index", 0)),
        int(rec.get("variant_index", 0)),
        rec.get("instruction_field", ""),
    )


def _gwet_ac1(pairs: Sequence[Tuple[int, int]]) -> float | None:
    """Chance-corrected agreement that does not collapse when one class dominates.

    Cohen's kappa is prevalence-deflated here: judges accept ~80% of items, so a
    high raw agreement still yields a near-zero kappa. AC1 is the standard fix.
    """
    n = len(pairs)
    if n == 0:
        return None
    p_a = sum(1 for a, b in pairs if a == b) / n
    # pi = mean probability of a "yes" across both raters
    pi = sum(a + b for a, b in pairs) / (2 * n)
    p_e = 2 * pi * (1 - pi)
    if p_e >= 1:
        return None
    return (p_a - p_e) / (1 - p_e)


def validation_stats(spec: Dict[str, Any]) -> Dict[str, Any] | None:
    val_dir = Path(spec["validation"])
    per_judge: Dict[str, Dict[Tuple, Dict[str, Any]]] = {}

    for judge, fname in spec["judges"].items():
        path = val_dir / fname
        if not path.is_file():
            print(f"  ! missing {path}", file=sys.stderr)
            continue
        per_judge[judge] = {_key(r): (r.get("answers") or {}) for r in _iter_ratings(path)}
        print(f"  {judge}: {len(per_judge[judge]):,} rated items")

    if not per_judge:
        return None

    # Acceptance rate per question, per judge.
    accept: List[Dict[str, Any]] = []
    for qid, qlabel in QUESTION_LABELS.items():
        row: Dict[str, Any] = {"question": qlabel}
        for judge, items in per_judge.items():
            scored = [
                a[qid]["score"]
                for a in items.values()
                if isinstance(a.get(qid), dict) and a[qid].get("score") is not None
            ]
            row[judge] = round(100 * sum(1 for s in scored if s >= ACCEPT_THRESHOLD) / len(scored), 1) if scored else None
        accept.append(row)

    # Cross-judge agreement on the accept decision, per question.
    judges = list(per_judge)
    agreement: List[Dict[str, Any]] = []
    if len(judges) >= 2:
        a_items, b_items = per_judge[judges[0]], per_judge[judges[1]]
        shared = set(a_items) & set(b_items)
        print(f"  shared items across judges: {len(shared):,}")
        for qid, qlabel in QUESTION_LABELS.items():
            pairs: List[Tuple[int, int]] = []
            for k in shared:
                qa, qb = a_items[k].get(qid), b_items[k].get(qid)
                if not isinstance(qa, dict) or not isinstance(qb, dict):
                    continue
                sa, sb = qa.get("score"), qb.get("score")
                if sa is None or sb is None:
                    continue
                pairs.append((int(sa >= ACCEPT_THRESHOLD), int(sb >= ACCEPT_THRESHOLD)))
            if not pairs:
                continue
            ac1 = _gwet_ac1(pairs)
            agreement.append({
                "question": qlabel,
                "agreement": round(100 * sum(1 for a, b in pairs if a == b) / len(pairs), 1),
                "ac1": round(ac1, 3) if ac1 is not None else None,
                "n": len(pairs),
            })

    # Joint distribution of the overall decision, for a scatter/heat cell view.
    joint: List[Dict[str, Any]] = []
    if len(judges) >= 2:
        a_items, b_items = per_judge[judges[0]], per_judge[judges[1]]
        cells: Counter = Counter()
        for k in set(a_items) & set(b_items):
            qa, qb = a_items[k].get("overall_validity"), b_items[k].get("overall_validity")
            if not isinstance(qa, dict) or not isinstance(qb, dict):
                continue
            sa, sb = qa.get("score"), qb.get("score")
            if sa is None or sb is None:
                continue
            cells[(int(sa), int(sb))] += 1
        joint = [{"x": x, "y": y, "count": n} for (x, y), n in sorted(cells.items())]

    return {
        "key": spec["key"],
        "judges": judges,
        "accept_by_question": accept,
        "agreement": agreement,
        "joint_overall": joint,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="paper/cache", help="Where paper_data_stats.py cached its parquet frames.")
    ap.add_argument("--out", default="website/public/data", help="Output directory (relative to repo root).")
    args = ap.parse_args()

    cache_dir = (REPO / args.cache_dir).resolve()
    out_dir = (REPO / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets: List[Dict[str, Any]] = []
    validation: List[Dict[str, Any]] = []

    for spec in DATASETS:
        print(f"\n{spec['label']}:")
        ds = dataset_stats(cache_dir, spec["key"])
        if ds:
            ds["label"] = spec["label"]
            datasets.append(ds)
            o = ds["overview"]
            print(f"  {o['clips']:,} clips · {o['chains']:,} chains · {o['steps']:,} steps · {o['variants']:,} variants")
        v = validation_stats(spec)
        if v:
            v["label"] = spec["label"]
            validation.append(v)

    if not datasets:
        sys.exit("No cached frames found. Run scripts/paper_data_stats.py first.")

    payload = {
        "generated_by": "scripts/export_website_stats.py",
        "accept_threshold": ACCEPT_THRESHOLD,
        "datasets": datasets,
        "validation": validation,
    }
    path = out_dir / "stats.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {path.relative_to(REPO)} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
