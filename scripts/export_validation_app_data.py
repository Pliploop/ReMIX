#!/usr/bin/env python
"""Export the human-validation sidecars as web-ready JSON for the rating app.

Additive and read-only. The Streamlit app keeps working exactly as it does.

The sidecar (``validation/assignment_*.sidecar.json``) is already a frozen,
self-contained slice: assignments, their chains, and the manifest rows they
reference. That makes it the right input -- the rating app never needs the 8 GB
chains file or the run root at all.

Two things are rewritten on the way out:

  * audio: sidecar rows carry local ``/data/...`` paths, which are useless (and
    for Music4All, unlawful) to serve. They become Jamendo CDN / Spotify refs.
  * rubric: imported from jamendo_instruct.demo.validation_rubric so the web app,
    the Streamlit app and the LLM judge cannot drift apart.

Usage:
    python scripts/export_validation_app_data.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from export_website_data import (  # noqa: E402
    M4A_RAW,
    MTG_RAW,
    _split_tags,
    load_jamendo_licenses,
    load_m4a_metadata,
)
from jamendo_instruct.demo.validation_rubric import (  # noqa: E402
    CANNOT_JUDGE_LABEL,
    DEGREE_OPTIONS,
    ISSUE_TAGS,
    LIKERT_OPTIONS,
    NOT_APPLICABLE_LABEL,
    RATING_QUESTIONS,
)

# Both sidecars per dataset, and they are not interchangeable:
#   axis_focused_5_v1  -> 100 items: 50 sentinel (attention checks) + 50
#                         core_overlap (the shared items every rater sees, which
#                         is what makes inter-rater agreement computable).
#   full_validation_v1 -> ~1.1k items: the bulk rating workload.
# Exporting only the first would cap the study at 100 items per dataset.
SIDECARS = [
    ("MTG-Jamendo", "mtg_jamendo", "/gpfs/scratch/acw749/datasets/mtg_jamendo_instruct/v1/validation"),
    ("Music4All", "music4all", "/gpfs/scratch/acw749/datasets/music4all_instruct/music4all_v1/validation"),
]

SIDECAR_NAMES = ("assignment_axis_focused_5_v1", "assignment_full_validation_v1")


def rubric_payload() -> Dict[str, Any]:
    questions = []
    for q in RATING_QUESTIONS:
        scale = q.get("scale", LIKERT_OPTIONS)
        questions.append({
            "id": q["id"],
            "statement": q["statement"],
            "help": q.get("help", ""),
            "allow_na": bool(q.get("allow_na")),
            "options": [{"label": label, "score": score} for label, score in scale],
        })
    return {
        "questions": questions,
        "issue_tags": list(ISSUE_TAGS),
        "cannot_judge_label": CANNOT_JUDGE_LABEL,
        "not_applicable_label": NOT_APPLICABLE_LABEL,
    }


def track_from_manifest(
    clip_id: str,
    row: Dict[str, Any],
    *,
    dataset: str,
    jamendo: Dict[str, Dict[str, str]],
    m4a: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    """Sidecar manifest row -> a track the web app can render and play."""
    track_id = row.get("track_id") or clip_id.split("::")[0]
    tags = row.get("tags")
    if isinstance(tags, str):
        tags = _split_tags(tags)
    elif not isinstance(tags, list):
        tags = []

    out: Dict[str, Any] = {
        "clip_id": clip_id,
        "tags": tags[:6],
        "caption": (row.get("primary_caption") or row.get("caption") or "").strip(),
        "vocals": row.get("vocals") or "",
        "speed": row.get("speed") or "",
        "lyrics": (row.get("lyrics") or "").strip()[:600],
    }

    if dataset == "mtg_jamendo":
        num = str(track_id).replace("track_", "").lstrip("0") or "0"
        lic = jamendo.get(num, {})
        out["title"] = lic.get("title") or "Untitled"
        out["artist"] = lic.get("artist") or "Unknown artist"
        out["audio"] = {
            "kind": "jamendo",
            "url": f"https://mp3d.jamendo.com/?trackid={num}&format=mp31",
            "page": lic.get("page") or f"https://www.jamendo.com/track/{num}",
            "license": lic.get("license") or "Creative Commons",
            "license_url": lic.get("license_url") or "",
        }
    else:
        meta = m4a.get(str(track_id), {})
        out["title"] = meta.get("title") or row.get("title") or "Untitled"
        out["artist"] = meta.get("artist") or row.get("artist_name") or "Unknown artist"
        sid = meta.get("spotify_id")
        out["audio"] = {"kind": "spotify", "id": sid} if sid else {"kind": "none"}
    return out


def export_sidecar(key: str, path: Path, jamendo, m4a, seen: set) -> List[Dict[str, Any]]:
    if not path.is_file():
        print(f"  ! missing {path.name}", file=sys.stderr)
        return []

    sidecar = json.loads(path.read_text(encoding="utf-8"))
    manifest = sidecar["manifest_by_clip"]
    chains = {c["chain_id"]: c for c in sidecar["chains"]}

    items: List[Dict[str, Any]] = []
    skipped = 0

    for a in sidecar["assignments"]:
        chain = chains.get(a["chain_id"])
        if not chain:
            skipped += 1
            continue
        step = next((s for s in chain["steps"] if int(s["turn_index"]) == int(a["turn_index"])), None)
        if not step:
            skipped += 1
            continue

        field = a["instruction_field"]
        # variant_indices says which drafts this assignment covers.
        wanted = a.get("variant_indices") or [0]
        variants = []
        for vi in wanted:
            recs = step.get("instruction_records") or []
            if vi >= len(recs):
                continue
            rec = recs[vi]
            text = (rec.get(field) or "").strip()
            if text:
                variants.append({"variant_index": vi, "instruction": text})
        if not variants:
            skipped += 1
            continue

        src_row = manifest.get(a["source_clip_id"])
        tgt_row = manifest.get(a["target_clip_id"])
        if not src_row or not tgt_row:
            skipped += 1
            continue

        # The two sidecars overlap on chains; key on the assignment identity so an
        # item is not queued twice.
        ident = (a["chain_id"], a["turn_index"], tuple(wanted), field)
        if ident in seen:
            skipped += 1
            continue
        seen.add(ident)

        delta = step.get("structured_delta") or {}
        items.append({
            # full_validation assignments carry no assignment_id; stable_hash is the
            # identity both sidecars share, and it is what the rating record keys on.
            "assignment_id": a.get("assignment_id") or a["stable_hash"],
            "chain_id": a["chain_id"],
            "turn_index": a["turn_index"],
            "bucket": a["bucket"],
            "is_sentinel": bool(a.get("is_sentinel")),
            "instruction_field": field,
            "hardness": a.get("hardness", ""),
            "change_axes": a.get("change_axes") or [],
            "split": a.get("split", ""),
            "stable_hash": a.get("stable_hash", ""),
            "variants": variants,
            "source": track_from_manifest(a["source_clip_id"], src_row, dataset=key, jamendo=jamendo, m4a=m4a),
            "target": track_from_manifest(a["target_clip_id"], tgt_row, dataset=key, jamendo=jamendo, m4a=m4a),
            "evidence": {
                "tags_added": delta.get("tags_added") or [],
                "tags_removed": delta.get("tags_removed") or [],
                "tags_preserved": delta.get("tags_preserved") or [],
            },
        })

    print(f"    {path.name}: +{len(items)} items ({skipped} skipped/duplicate)")
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Not website/public: everything under public/ is copied into the GitHub
    # Pages build, and the rating workload (7 MB) has no business on the public
    # site. The Space's Dockerfile copies this in before it builds.
    ap.add_argument("--out", default="website/space-data",
                    help="Output dir (relative to repo root).")
    args = ap.parse_args()

    jamendo = load_jamendo_licenses(MTG_RAW / "audio_licenses.txt")
    m4a = load_m4a_metadata(M4A_RAW)

    datasets = []
    for label, key, val_dir in SIDECARS:
        print(f"\n{label}:")
        seen: set = set()
        items: List[Dict[str, Any]] = []
        for name in SIDECAR_NAMES:
            items += export_sidecar(key, Path(val_dir) / f"{name}.sidecar.json", jamendo, m4a, seen)
        if not items:
            continue
        playable = sum(
            1 for it in items for t in (it["source"], it["target"]) if t["audio"]["kind"] != "none"
        )
        buckets: Dict[str, int] = {}
        for it in items:
            buckets[it["bucket"]] = buckets.get(it["bucket"], 0) + 1
        print(f"  total {len(items)} items · buckets {buckets} · {playable}/{len(items) * 2} tracks playable")
        datasets.append({"key": key, "label": label, "items": items})

    if not datasets:
        sys.exit("Nothing exported.")

    payload = {
        "generated_by": "scripts/export_validation_app_data.py",
        "rubric": rubric_payload(),
        "datasets": datasets,
    }
    out = (REPO / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "validation_tasks.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {path.relative_to(REPO)} ({path.stat().st_size / 1024:.0f} KB)")
    print(f"Rubric: {len(payload['rubric']['questions'])} questions, {len(payload['rubric']['issue_tags'])} issue tags")


if __name__ == "__main__":
    main()
