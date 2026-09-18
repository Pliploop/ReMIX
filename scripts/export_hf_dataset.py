#!/usr/bin/env python
"""Export ReMIX to a Hugging Face-ready dataset: parquet by split + a dataset card.

One row per instruction variant of a chain step (the flat unit a retrieval/eval
consumer wants), carrying the source/target track metadata, the instruction, the
semantic delta, and each LLM judge's overall-validity score. Every step is
included; a `validated` flag + the raw judge scores let a consumer keep only the
gated subset or study the rest.

LICENSING: only the *audio* is a redistribution-restricted derivative, and this
script never exports audio -- only text (instructions, captions, tags, metadata)
and reference URLs. So both catalogs are exportable. Pushed repos default to
PRIVATE until the dataset is ready to distribute.

Usage:
  # Export MTG locally:
  python scripts/export_hf_dataset.py --out hf_export

  # Export Music4All:
  python scripts/export_hf_dataset.py --dataset music4all --out hf_export

  # Push to a private HF dataset repo (needs HF_TOKEN):
  python scripts/export_hf_dataset.py --push Pliploop/ReMIX-MTG
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_website_data import (  # noqa: E402  (single source of truth for paths + track refs)
    INSTR_FOLDER,
    JUDGE_FILES,
    M4A_RAW,
    M4A_ROOT,
    MTG_RAW,
    MTG_ROOT,
    build_track,
    load_jamendo_licenses,
    load_m4a_metadata,
    load_manifest,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jamendo_instruct.demo.validation_rubric import RATING_QUESTIONS  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

DATASETS = {
    "mtg_jamendo": {"root": MTG_ROOT, "label": "MTG-Jamendo", "license": "cc-by-nc-sa-4.0"},
    "music4all": {"root": M4A_ROOT, "label": "Music4All", "license": "other"},
}

# Every rubric question, scored per judge (columns like `qwen_overall_validity`).
QUESTION_IDS = [str(q["id"]) for q in RATING_QUESTIONS]
DECISION_Q = "overall_validity"
# JUDGE_FILES[key] is (qwen_file, gemma_file), so these names align to that order.
JUDGE_NAMES = ("qwen", "gemma")
ACCEPT_THRESHOLD = 4.0


def _read_instructions(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in tqdm(f, desc="  read instructions", unit="rec"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") not in (None, "ok"):
                continue
            if not (rec.get("history_unaware_instruction") or "").strip():
                continue
            rows.append(rec)
    return rows


def _delta(rec: Dict[str, Any]) -> Dict[str, List[str]]:
    d = rec.get("semantic_delta_verbalized") or rec.get("semantic_delta_full") or {}
    return {k: list(d.get(k) or []) for k in ("lost", "new", "preserved")}


def _load_judge_scores(path: Path) -> Dict[Tuple[str, int, int], Dict[str, float]]:
    """(chain, turn, variant) -> {question_id: score}, mean if a judge rated it twice."""
    acc: Dict[Tuple[str, int, int], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in tqdm(f, desc=f"  scores {path.name}", unit="rec", leave=False):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (rec.get("chain_id"), int(rec.get("turn_index", 0)), int(rec.get("variant_index", 0)))
            answers = rec.get("answers", {}) or {}
            for qid in QUESTION_IDS:
                a = answers.get(qid, {}) or {}
                if a.get("cannot_judge") or a.get("not_applicable"):
                    continue
                s = a.get("score")
                if isinstance(s, (int, float)):
                    acc[key][qid].append(float(s))
    return {k: {q: round(sum(v) / len(v), 3) for q, v in d.items()} for k, d in acc.items()}


def build_rows(key: str, root: Path, folder: str, jamendo, m4a) -> List[Dict[str, Any]]:
    instr_path = root / folder / "chain_step_instructions.jsonl"
    if not instr_path.is_file():
        sys.exit(f"missing {instr_path}")
    records = _read_instructions(instr_path)
    print(f"  {len(records):,} instruction variants")

    # Per-judge, per-question scores, keyed by (chain, turn, variant).
    val_dir = root / folder / "validation"
    judge_score: List[Dict[Tuple[str, int, int], Dict[str, float]]] = []
    for name in JUDGE_FILES[key]:
        # Prefer the merged .validated.jsonl (full run); fall back to the raw sidecar.
        validated = val_dir / name.replace(".jsonl", ".validated.jsonl")
        judge_score.append(_load_judge_scores(validated if validated.is_file() else val_dir / name))
    rated = len(set().union(*[set(js) for js in judge_score])) if judge_score else 0
    print(f"  {rated:,} steps carry at least one judge score")

    clip_ids = {c for r in records for c in (r["source_clip_id"], r["target_clip_id"])}
    manifest = load_manifest(root / "ingest" / "normalized_track_manifest.csv", clip_ids)
    tracks = {cid: build_track(cid, manifest, dataset=key, jamendo=jamendo, m4a=m4a) for cid in clip_ids}

    rows: List[Dict[str, Any]] = []
    for r in tqdm(records, desc="  build rows", unit="row"):
        k = (r["chain_id"], int(r.get("turn_index", 0)), int(r.get("variant_index", 0)))
        src = tracks.get(r["source_clip_id"], {})
        tgt = tracks.get(r["target_clip_id"], {})
        delta = _delta(r)
        # Every question's score for every judge: <judge>_<question_id>.
        score_cols: Dict[str, Any] = {}
        overalls: List[float] = []
        for jname, js in zip(JUDGE_NAMES, judge_score):
            qmap = js.get(k, {})
            for qid in QUESTION_IDS:
                score_cols[f"{jname}_{qid}"] = qmap.get(qid)
            if qmap.get(DECISION_Q) is not None:
                overalls.append(qmap[DECISION_Q])
        validated = len(overalls) == len(JUDGE_NAMES) and all(v >= ACCEPT_THRESHOLD for v in overalls)
        rows.append({
            "chain_id": r["chain_id"],
            "turn_index": k[1],
            "variant_index": k[2],
            "split": r.get("split", ""),
            "seed_clip_id": r.get("seed_clip_id", ""),
            "instruction": r.get("history_unaware_instruction") or "",
            "instruction_contextual": r.get("history_aware_instruction") or "",
            "hardness": r.get("hardness", ""),
            "verbosity": r.get("verbosity", ""),
            "transition_score": round(float(r.get("transition_score") or 0), 4),
            "change_axes": list(r.get("selected_change_axes") or []),
            "preservation_axes": list(r.get("selected_preservation_axes") or []),
            "delta_lost": delta["lost"],
            "delta_new": delta["new"],
            "delta_preserved": delta["preserved"],
            # source / target tracks
            "source_clip_id": r["source_clip_id"],
            "source_title": src.get("title", ""),
            "source_artist": src.get("artist", ""),
            "source_tags": src.get("tags", []),
            "source_caption": src.get("caption", ""),
            "source_audio_url": (src.get("audio", {}) or {}).get("url", ""),
            "target_clip_id": r["target_clip_id"],
            "target_title": tgt.get("title", ""),
            "target_artist": tgt.get("artist", ""),
            "target_tags": tgt.get("tags", []),
            "target_caption": tgt.get("caption", ""),
            "target_audio_url": (tgt.get("audio", {}) or {}).get("url", ""),
            # per-judge, per-question scores + the gate
            **score_cols,
            "validated": validated,
        })
    return rows


DATASET_CARD = """\
---
license: {license}
task_categories:
- text-retrieval
language:
- en
tags:
- music
- music-information-retrieval
- instruction-following
- multi-turn
- compositional-retrieval
pretty_name: {pretty}
size_categories:
- {size_cat}
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.parquet
  - split: validation
    path: data/validation.parquet
  - split: test
    path: data/test.parquet
---

# {pretty}

**ReMIX** is a large-scale dataset of **multi-turn, compositional, instruction-based
music retrieval**: finding a track is a conversation, not a single query. You start
close to what you want, then steer — *"make it punchier"*, *"keep the vocals but
brighten it"*, *"bring back that piano from before"*. Each turn is an **edit** on the
previous result, and instructions may refer back to earlier turns.

This is the **{label}** split of ReMIX ({total:,} instruction variants over
{n_chains:,} chains).

## How it was built

1. **Enrich** — open music catalogs; every clip is captioned (how it sounds) and its
   lyrics transcribed, so each track carries rich text alongside its tags.
2. **Connect** — every clip is embedded by audio and by description; similar clips are
   linked into a graph of plausible transitions.
3. **Walk** — a stochastic weighted walk over that graph draws multi-turn chains of
   1–6 steps; each hop is a small, plausible musical change.
4. **Instruct** — each hop is diffed into a semantic delta, and an LLM writes the
   instruction that turns one track into the next. Up to five variants per step.
5. **Validate** — two LLM judges score every instruction against an 8-question rubric.
   `validated = true` marks steps both judges scored ≥ {threshold} on overall validity.

Every step is included (validated or not) so the un-gated data is available for study.

## Dataset structure

One row per **instruction variant of a chain step**. Splits: `train`, `validation`,
`test` ({train:,} / {val:,} / {test:,} rows; {validated:,} validated).

### Columns

**Identity** — `chain_id`, `turn_index`, `variant_index`, `split`, `seed_clip_id`.

**Instruction** — `instruction` (standalone), `instruction_contextual` (may refer to
earlier turns), `hardness`, `verbosity`, `transition_score`, `change_axes`,
`preservation_axes`, `delta_lost` / `delta_new` / `delta_preserved` (the semantic diff).

**Tracks** — for `source_` and `target_`: `clip_id`, `title`, `artist`, `tags`,
`caption`, `audio_url`. **No audio is distributed**; `audio_url` references the
{audio_source}.

**Judge scores** (1–5, `null` if not yet judged or the judge abstained) — every rubric
question from each judge ({judges}):

{score_cols_block}

**Gate** — `validated` (bool): both judges scored `overall_validity` ≥ {threshold}.

## Load

```python
from datasets import load_dataset
ds = load_dataset("{repo_hint}")
# validated subset only:
val = ds["train"].filter(lambda r: r["validated"])
```

## Licensing

Only the underlying **audio** is redistribution-restricted, and **this dataset
contains no audio** — only text (instructions, captions, tags, metadata) and
reference URLs. {license_note}

## Citation

_Paper forthcoming._
"""


def _size_category(n: int) -> str:
    for cutoff, label in ((1_000, "n<1K"), (10_000, "1K<n<10K"), (100_000, "10K<n<100K"),
                          (1_000_000, "100K<n<1M"), (10_000_000, "1M<n<10M")):
        if n < cutoff:
            return label
    return "10M<n<100M"


def write_card(out_dir: Path, key: str, meta: Dict[str, Any], repo_hint: str, stats: Dict[str, int]) -> None:
    score_cols_block = "\n".join(
        f"- `{jname}_*`: " + ", ".join(f"`{jname}_{qid}`" for qid in QUESTION_IDS)
        for jname in JUDGE_NAMES
    )
    license_note = (
        "MTG-Jamendo tracks are Creative Commons; `audio_url` points at the Jamendo CDN."
        if key == "mtg_jamendo"
        else "The source audio (Music4All) is under a non-redistribution agreement and is not "
        "included here; only text and identifiers are provided."
    )
    card = DATASET_CARD.format(
        license=meta["license"],
        pretty=f"ReMIX — {meta['label']}",
        label=meta["label"],
        threshold=ACCEPT_THRESHOLD,
        audio_source="Jamendo CDN (Creative Commons)" if key == "mtg_jamendo" else "original catalog (not redistributed)",
        judges=" and ".join(JUDGE_NAMES),
        score_cols_block=score_cols_block,
        total=stats["total"],
        n_chains=stats["chains"],
        train=stats["train"], val=stats["validation"], test=stats["test"],
        validated=stats["validated"],
        size_cat=_size_category(stats["total"]),
        repo_hint=repo_hint,
        license_note=license_note,
    )
    (out_dir / "README.md").write_text(card, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(DATASETS), default="mtg_jamendo")
    ap.add_argument("--folder", default=INSTR_FOLDER,
                    help=f"Instructions folder under the run root (default: {INSTR_FOLDER}, the canonical one).")
    ap.add_argument("--out", default="hf_export", help="Output dir (a repo-shaped folder).")
    ap.add_argument("--push", default=None, metavar="REPO_ID",
                    help="Also push to this HF dataset repo (needs HF_TOKEN). Omit to only write locally.")
    ap.add_argument("--public", action="store_true",
                    help="Push as a PUBLIC repo. Default is private (dataset not ready to distribute).")
    args = ap.parse_args()

    key = args.dataset
    meta = DATASETS[key]

    jamendo = load_jamendo_licenses(MTG_RAW / "audio_licenses.txt") if key == "mtg_jamendo" else {}
    m4a = load_m4a_metadata(M4A_RAW) if key == "music4all" else {}

    print(f"{meta['label']} ({args.folder}):")
    rows = build_rows(key, meta["root"], args.folder, jamendo, m4a)
    df = pd.DataFrame(rows)

    out_dir = (REPO / args.out / key).resolve()
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    split_map = {"train": "train", "val": "validation", "valid": "validation", "validation": "validation", "test": "test"}
    counts: Dict[str, int] = {}
    for split in ("train", "validation", "test"):
        sub = df[df["split"].map(lambda s: split_map.get(str(s), "train")) == split]
        sub.to_parquet(data_dir / f"{split}.parquet", index=False)
        counts[split] = len(sub)
    stats = {
        **counts,
        "total": len(df),
        "chains": int(df["chain_id"].nunique()),
        "validated": int(df["validated"].sum()),
    }
    write_card(out_dir, key, meta, args.push or f"<user>/ReMIX-{key}", stats)
    try:
        shown = out_dir.relative_to(REPO)  # nice short path when --out is under the repo
    except ValueError:
        shown = out_dir  # --out on scratch etc.
    print(f"  wrote {shown}  splits={counts}  validated={stats['validated']:,}/{len(df):,}")

    if args.push:
        from huggingface_hub import HfApi
        # Use HF_TOKEN if set, else the token from `hf auth login` (HfApi(token=None)
        # picks up the cached login automatically).
        token = os.environ.get("HF_TOKEN", "").strip() or None
        api = HfApi(token=token)
        api.create_repo(args.push, repo_type="dataset", private=not args.public, exist_ok=True)
        # upload_folder is content-addressed: unchanged files (same hash) are skipped,
        # so re-running only uploads what actually changed. No delete_patterns, so
        # nothing already on the repo is removed.
        print("  syncing to HF (unchanged files skipped) ...")
        api.upload_folder(folder_path=str(out_dir), repo_id=args.push, repo_type="dataset",
                          commit_message=f"Sync {meta['label']} ({args.folder})")
        vis = "public" if args.public else "private"
        print(f"  pushed ({vis}) -> https://huggingface.co/datasets/{args.push}")


if __name__ == "__main__":
    main()
