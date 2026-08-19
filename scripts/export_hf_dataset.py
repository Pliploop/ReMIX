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
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_website_data import (  # noqa: E402  (single source of truth for paths + track refs)
    DECISION_Q,
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
    load_ratings,
)

REPO = Path(__file__).resolve().parents[1]

DATASETS = {
    "mtg_jamendo": {"root": MTG_ROOT, "label": "MTG-Jamendo", "license": "cc-by-nc-sa-4.0"},
    "music4all": {"root": M4A_ROOT, "label": "Music4All", "license": "other"},
}

# The two judges, in a stable order, keyed to their score column names.
JUDGE_COLS = ("qwen_overall", "gemma_overall")
ACCEPT_THRESHOLD = 4.0


def _read_instructions(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
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


def _mean_or_none(scores: Dict[Tuple[str, int, int], List[float]], key: Tuple[str, int, int]) -> float | None:
    vals = scores.get(key)
    return round(sum(vals) / len(vals), 3) if vals else None


def build_rows(key: str, root: Path, jamendo, m4a) -> List[Dict[str, Any]]:
    instr_path = root / INSTR_FOLDER / "chain_step_instructions.jsonl"
    if not instr_path.is_file():
        sys.exit(f"missing {instr_path}")
    records = _read_instructions(instr_path)
    print(f"  {len(records):,} instruction variants")

    # Per-judge overall-validity score, keyed by (chain, turn, variant).
    val_dir = root / INSTR_FOLDER / "validation"
    judge_score: List[Dict[Tuple[str, int, int], List[float]]] = []
    for name in JUDGE_FILES[key]:
        # Prefer the merged .validated.jsonl (full run); fall back to the raw sidecar.
        validated = val_dir / name.replace(".jsonl", ".validated.jsonl")
        judge_score.append(load_ratings([validated if validated.is_file() else val_dir / name]))
    rated = len(set().union(*[set(js) for js in judge_score])) if judge_score else 0
    print(f"  {rated:,} steps carry at least one judge score")

    clip_ids = {c for r in records for c in (r["source_clip_id"], r["target_clip_id"])}
    manifest = load_manifest(root / "ingest" / "normalized_track_manifest.csv", clip_ids)
    tracks = {cid: build_track(cid, manifest, dataset=key, jamendo=jamendo, m4a=m4a) for cid in clip_ids}

    rows: List[Dict[str, Any]] = []
    for r in records:
        k = (r["chain_id"], int(r.get("turn_index", 0)), int(r.get("variant_index", 0)))
        src = tracks.get(r["source_clip_id"], {})
        tgt = tracks.get(r["target_clip_id"], {})
        delta = _delta(r)
        scores = {col: _mean_or_none(js, k) for col, js in zip(JUDGE_COLS, judge_score)}
        have_all = all(v is not None for v in scores.values())
        validated = have_all and all(v >= ACCEPT_THRESHOLD for v in scores.values())
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
            # judge scores + gate
            **{col: scores[col] for col in JUDGE_COLS},
            "validated": validated,
        })
    return rows


DATASET_CARD = """\
---
license: {license}
task_categories:
- text-retrieval
tags:
- music
- music-information-retrieval
- instruction-following
- multi-turn
pretty_name: {pretty}
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

Multi-turn, compositional, instruction-based music retrieval. Each row is one
instruction variant of a chain step: an instruction that edits a **source** track
toward a **target** track, with both tracks' metadata, the semantic delta, and
each LLM judge's overall-validity score.

`validated = true` marks steps both judges scored >= {threshold} on overall
validity. Every step is included so the un-gated data is available too.

**No audio is distributed.** `*_audio_url` points at the {audio_source}.

## Columns
`chain_id, turn_index, variant_index, split, instruction, instruction_contextual,
change_axes, preservation_axes, delta_{{lost,new,preserved}}, transition_score,
source_*/target_* (title, artist, tags, caption, audio_url), {judge_cols},
validated`.
"""


def write_card(out_dir: Path, key: str, meta: Dict[str, Any]) -> None:
    card = DATASET_CARD.format(
        license=meta["license"],
        pretty=f"ReMIX — {meta['label']}",
        threshold=ACCEPT_THRESHOLD,
        audio_source="Jamendo CDN (Creative Commons)" if key == "mtg_jamendo" else "original catalog (not redistributed)",
        judge_cols=", ".join(JUDGE_COLS),
    )
    (out_dir / "README.md").write_text(card, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(DATASETS), default="mtg_jamendo")
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

    print(f"{meta['label']}:")
    rows = build_rows(key, meta["root"], jamendo, m4a)
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
    write_card(out_dir, key, meta)
    print(f"  wrote {out_dir.relative_to(REPO)}  splits={counts}  validated={int(df['validated'].sum()):,}/{len(df):,}")

    if args.push:
        from huggingface_hub import HfApi
        token = os.environ.get("HF_TOKEN", "").strip() or None
        if not token:
            sys.exit("--push needs HF_TOKEN in the environment.")
        api = HfApi(token=token)
        api.create_repo(args.push, repo_type="dataset", private=not args.public, exist_ok=True)
        api.upload_folder(folder_path=str(out_dir), repo_id=args.push, repo_type="dataset")
        vis = "public" if args.public else "private"
        print(f"  pushed ({vis}) -> https://huggingface.co/datasets/{args.push}")


if __name__ == "__main__":
    main()
