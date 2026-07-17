#!/usr/bin/env python
"""Export showcase chains for the ReMIX companion website as static JSON.

Additive and read-only: nothing here mutates pipeline outputs, and the Streamlit
apps are untouched. Reads the merged instruction JSONL (which already carries
source/target clip ids and the semantic delta, so the 8 GB chains JSONL is not
needed), joins track metadata, and picks chains both LLM judges liked.

Audio is never copied. Each track carries a *reference* instead:

  * MTG-Jamendo -> the Jamendo CDN, which serves the CC-licensed original.
    ``track_0000214`` -> ``14/214.mp3`` -> Jamendo track 214. The mapping and the
    per-track licence come from the dataset's own ``audio_licenses.txt``.
  * Music4All   -> a Spotify embed, keyed by the ``spotify_id`` in
    ``id_metadata.csv``. Music4All audio is under a signed non-redistribution
    agreement and must never be served by us.

Usage:
  python scripts/export_website_data.py
  python scripts/export_website_data.py --per-dataset 8 --out website/public/data
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

csv.field_size_limit(10 ** 9)

REPO = Path(__file__).resolve().parents[1]

MTG_ROOT = Path("/gpfs/scratch/acw749/datasets/mtg_jamendo_instruct/v1")
M4A_ROOT = Path("/gpfs/scratch/acw749/datasets/music4all_instruct/music4all_v1")
MTG_RAW = Path("/data/EECS-Pauwels-C4DM/mtg-jamendo-raw/mtg-jamendo-dataset")
M4A_RAW = Path("/data/EECS-Pauwels-C4DM/music4all")

INSTR_FOLDER = "instructions_axis_focused_5"

# The exact judge files the paper reports on (scripts/paper_validation_stats.py).
# Globbing instead would sweep in smoke/partial runs and silently disagree with the paper.
JUDGE_FILES: Dict[str, Tuple[str, ...]] = {
    "mtg_jamendo": ("llm_ratings_qwen_full.jsonl", "llm_ratings_gemma_full.jsonl"),
    "music4all": ("llm_ratings.jsonl", "llm_ratings_gemma_full.jsonl"),
}

# Rubric question that carries the keep/reject decision.
DECISION_Q = "overall_validity"

# `Intro chiante by David TMX from Jamendo: http://www.jamendo.com/track/214`
_LIC_TITLE_RE = re.compile(r"^(?P<title>.*?) by (?P<artist>.*?) from Jamendo: (?P<url>\S+)\s*$")
_LIC_NAME_RE = re.compile(r"under a (?P<name>.*?) licen[cs]e:\s*(?P<url>\S+)", re.I)


# --------------------------------------------------------------------------- #
# metadata sources
# --------------------------------------------------------------------------- #
def load_jamendo_licenses(path: Path) -> Dict[str, Dict[str, str]]:
    """Parse audio_licenses.txt -> jamendo id -> real title/artist/licence.

    The file repeats 3-line blocks: relative mp3 path, "<title> by <artist> from
    Jamendo: <url>", then the licence line.
    """
    out: Dict[str, Dict[str, str]] = {}
    if not path.is_file():
        return out
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line.endswith(".mp3"):
            continue
        jam_id = Path(line).stem  # "14/214.mp3" -> "214"
        info: Dict[str, str] = {"jamendo_id": jam_id}
        if i + 1 < len(lines):
            m = _LIC_TITLE_RE.match(lines[i + 1].strip())
            if m:
                info["title"] = m.group("title").strip()
                info["artist"] = m.group("artist").strip()
                info["page"] = m.group("url").replace("http://", "https://")
        if i + 2 < len(lines):
            m = _LIC_NAME_RE.search(lines[i + 2].strip())
            if m:
                info["license"] = m.group("name").strip()
                info["license_url"] = m.group("url").strip()
        out[jam_id] = info
    return out


def load_m4a_metadata(root: Path) -> Dict[str, Dict[str, str]]:
    """track_id -> {spotify_id, title, artist, album}. Tab-separated per the readme."""
    out: Dict[str, Dict[str, str]] = defaultdict(dict)
    info = root / "id_information.csv"
    meta = root / "id_metadata.csv"
    if info.is_file():
        with info.open(newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                tid = (row.get("id") or "").strip()
                if tid:
                    out[tid].update(
                        title=(row.get("song") or "").strip(),
                        artist=(row.get("artist") or "").strip(),
                        album=(row.get("album_name") or "").strip(),
                    )
    if meta.is_file():
        with meta.open(newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                tid = (row.get("id") or "").strip()
                sid = (row.get("spotify_id") or "").strip()
                if tid and sid:
                    out[tid]["spotify_id"] = sid
    return dict(out)


def load_manifest(path: Path, clip_ids: set[str]) -> Dict[str, Dict[str, Any]]:
    """clip_id -> the manifest fields the site actually renders."""
    keep = (
        "clip_id", "track_id", "caption", "primary_caption", "tags", "split",
        "start_time", "end_time", "title", "duration", "artist_name", "vocals", "speed",
    )
    out: Dict[str, Dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            cid = row.get("clip_id")
            if cid in clip_ids:
                out[cid] = {k: row.get(k, "") for k in keep}
                if len(out) == len(clip_ids):
                    break
    return out


def load_ratings(paths: Iterable[Path]) -> Dict[Tuple[str, int, int], List[float]]:
    """(chain, turn, variant) -> overall_validity score from each judge."""
    scores: Dict[Tuple[str, int, int], List[float]] = defaultdict(list)
    for p in paths:
        if not p.is_file():
            continue
        with p.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ans = (rec.get("answers") or {}).get(DECISION_Q) or {}
                s = ans.get("score")
                if s is None:
                    continue
                key = (rec.get("chain_id"), int(rec.get("turn_index", 0)), int(rec.get("variant_index", 0)))
                scores[key].append(float(s))
    return dict(scores)


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def _split_tags(raw: str) -> List[str]:
    return [t.strip() for t in (raw or "").split(",") if t.strip()]


_COMPACT_KEYS = (
    "chain_id", "turn_index", "variant_index", "source_clip_id", "target_clip_id",
    "history_unaware_instruction", "history_aware_instruction", "hardness",
    "transition_score", "selected_change_axes", "selected_preservation_axes", "split",
    # Short label distinguishing the variants of one step in the explorer's dropdown.
    "verbosity",
)


def _compact(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only what the site renders, so a whole dataset fits in memory."""
    out = {k: rec.get(k) for k in _COMPACT_KEYS}
    # NB: the instruction record's delta is phrase-based (lost/new/preserved). The
    # tags_added/tags_removed keys belong to the *chain* record's structured_delta.
    delta = rec.get("semantic_delta_full") or {}
    out["_delta"] = {k: delta.get(k) or [] for k in ("lost", "new", "preserved")}
    out["_primary_edit"] = delta.get("primary_edit") or ""
    out["_shape"] = (rec.get("instruction_plan") or {}).get("instruction_shape") or ""
    return out


def pick_chains(
    instr_path: Path,
    ratings: Dict[Tuple[str, int, int], List[float]],
    *,
    min_len: int,
    max_len: int,
    per_dataset: int,
    min_score: float,
) -> List[Dict[str, Any]]:
    """Best-variant-per-turn, keeping only chains every judge liked at every turn."""
    by_chain: Dict[str, Dict[int, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    with instr_path.open(encoding="utf-8", errors="replace") as f:
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
            # Project to just the rendered fields before retaining. The raw records
            # carry full captions and prompt echoes; holding 88k of them OOMs.
            by_chain[rec["chain_id"]][int(rec.get("turn_index", 0))].append(_compact(rec))

    chains: List[Dict[str, Any]] = []
    for chain_id, turns in by_chain.items():
        if not (min_len <= len(turns) <= max_len):
            continue
        steps: List[Dict[str, Any]] = []
        chain_scores: List[float] = []
        ok = True
        for turn in sorted(turns):
            best, best_score = None, -1.0
            for rec in turns[turn]:
                key = (chain_id, turn, int(rec.get("variant_index", 0)))
                got = ratings.get(key)
                if not got:
                    continue
                mean = sum(got) / len(got)
                # Require every judge to like it, not just the average.
                if min(got) < min_score:
                    continue
                if mean > best_score:
                    best, best_score = rec, mean
            if best is None:
                ok = False
                break
            steps.append(best)
            chain_scores.append(best_score)
        if not ok or not steps:
            continue
        chains.append({
            "chain_id": chain_id,
            "steps": steps,
            "score": sum(chain_scores) / len(chain_scores),
        })

    chains.sort(key=lambda c: (-c["score"], c["chain_id"]))
    return chains


def diversify(chains: List[Dict[str, Any]], manifest: Dict[str, Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """Greedily prefer chains whose seed genre we have not shown yet."""
    seen: set[str] = set()
    picked: List[Dict[str, Any]] = []
    for pool in (0, 1):  # first pass: unseen genres only; second: fill remainder
        for c in chains:
            if len(picked) >= limit:
                return picked
            if c in picked:
                continue
            src = manifest.get(c["steps"][0]["source_clip_id"], {})
            tags = _split_tags(src.get("tags", ""))
            genre = tags[0].lower() if tags else "?"
            if pool == 0 and genre in seen:
                continue
            seen.add(genre)
            picked.append(c)
    return picked[:limit]


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def build_track(
    clip_id: str,
    manifest: Dict[str, Dict[str, Any]],
    *,
    dataset: str,
    jamendo: Dict[str, Dict[str, str]],
    m4a: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    row = manifest.get(clip_id, {})
    track_id = row.get("track_id") or clip_id.split("::")[0]

    track: Dict[str, Any] = {
        "clip_id": clip_id,
        "track_id": track_id,
        "tags": _split_tags(row.get("tags", "")),
        "caption": (row.get("primary_caption") or row.get("caption") or "").strip(),
        "vocals": row.get("vocals") or "",
        "start": float(row.get("start_time") or 0),
        "end": float(row.get("end_time") or 30),
    }

    if dataset == "mtg_jamendo":
        # track_0000214 -> 214, matching audio_licenses.txt / the mp3 tree.
        num = track_id.replace("track_", "").lstrip("0") or "0"
        lic = jamendo.get(num, {})
        track["title"] = lic.get("title") or "Untitled"
        track["artist"] = lic.get("artist") or "Unknown artist"
        track["audio"] = {
            "kind": "jamendo",
            # CORS-open and returns audio/mpeg, so wavesurfer can draw a real waveform.
            "url": f"https://mp3d.jamendo.com/?trackid={num}&format=mp31",
            "page": lic.get("page") or f"https://www.jamendo.com/track/{num}",
            "license": lic.get("license") or "Creative Commons",
            "license_url": lic.get("license_url") or "",
        }
    else:
        meta = m4a.get(track_id, {})
        track["title"] = meta.get("title") or row.get("title") or "Untitled"
        track["artist"] = meta.get("artist") or row.get("artist_name") or "Unknown artist"
        sid = meta.get("spotify_id")
        # No spotify_id -> no playable, licence-clean source. Render without audio.
        track["audio"] = {"kind": "spotify", "id": sid} if sid else {"kind": "none"}
    return track


def build_step(rec: Dict[str, Any], tracks: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    delta = rec.get("_delta") or {}
    return {
        "turn": int(rec.get("turn_index", 0)),
        "source": tracks.get(rec["source_clip_id"]),
        "target": tracks.get(rec["target_clip_id"]),
        "instruction": (rec.get("history_unaware_instruction") or "").strip(),
        "instruction_contextual": (rec.get("history_aware_instruction") or "").strip(),
        "hardness": rec.get("hardness") or "",
        "transition_score": round(float(rec.get("transition_score") or 0), 4),
        "change_axes": rec.get("selected_change_axes") or [],
        "preserve_axes": rec.get("selected_preservation_axes") or [],
        "primary_edit": rec.get("_primary_edit") or "",
        "lost": delta.get("lost") or [],
        "new": delta.get("new") or [],
        "preserved": delta.get("preserved") or [],
        "shape": rec.get("_shape") or "",
    }


def export_dataset(
    label: str,
    key: str,
    root: Path,
    *,
    per_dataset: int,
    min_len: int,
    max_len: int,
    min_score: float,
    jamendo: Dict[str, Dict[str, str]],
    m4a: Dict[str, Dict[str, str]],
) -> Optional[Dict[str, Any]]:
    instr_dir = root / INSTR_FOLDER
    instr_path = instr_dir / "chain_step_instructions.jsonl"
    if not instr_path.is_file():
        print(f"  ! {label}: no {instr_path}", file=sys.stderr)
        return None

    val_dir = instr_dir / "validation"
    rating_paths = [val_dir / name for name in JUDGE_FILES[key]]
    missing = [p.name for p in rating_paths if not p.is_file()]
    if missing:
        print(f"  ! {label}: missing judge file(s) {missing}", file=sys.stderr)
    print(f"  ratings: {[p.name for p in rating_paths] or 'none'}")
    ratings = load_ratings(rating_paths)
    print(f"  rated items: {len(ratings):,}")

    chains = pick_chains(
        instr_path, ratings,
        min_len=min_len, max_len=max_len, per_dataset=per_dataset, min_score=min_score,
    )
    print(f"  chains passing gate: {len(chains):,}")
    if not chains:
        return None

    # Only load manifest rows we will actually render.
    head = chains[: per_dataset * 6]
    clip_ids = {c for ch in head for s in ch["steps"] for c in (s["source_clip_id"], s["target_clip_id"])}
    manifest = load_manifest(root / "ingest" / "normalized_track_manifest.csv", clip_ids)
    print(f"  manifest rows: {len(manifest):,} / {len(clip_ids):,}")

    picked = diversify(head, manifest, per_dataset)

    out_chains = []
    for ch in picked:
        tracks = {
            cid: build_track(cid, manifest, dataset=key, jamendo=jamendo, m4a=m4a)
            for s in ch["steps"]
            for cid in (s["source_clip_id"], s["target_clip_id"])
        }
        steps = [build_step(s, tracks) for s in ch["steps"]]
        if any(s["source"] is None or s["target"] is None for s in steps):
            continue
        out_chains.append({
            "chain_id": ch["chain_id"],
            "split": ch["steps"][0].get("split", ""),
            "judge_score": round(ch["score"], 3),
            "steps": steps,
        })

    playable = sum(
        1 for ch in out_chains for s in ch["steps"]
        for t in (s["source"], s["target"]) if t["audio"]["kind"] != "none"
    )
    total = sum(len(ch["steps"]) * 2 for ch in out_chains)
    print(f"  exported: {len(out_chains)} chains, {playable}/{total} tracks playable")

    return {"key": key, "label": label, "chains": out_chains}


def main() -> None:
    global INSTR_FOLDER
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="website/public/data", help="Output directory (relative to repo root).")
    ap.add_argument("--per-dataset", type=int, default=6, help="Showcase chains per dataset.")
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=5)
    ap.add_argument("--min-score", type=float, default=4.0, help="Min overall_validity from *every* judge.")
    ap.add_argument("--folder", default=INSTR_FOLDER, help="Instructions folder under each run root.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    INSTR_FOLDER = args.folder
    print(f"Instructions folder: {INSTR_FOLDER}")

    random.seed(args.seed)
    out_dir = (REPO / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading catalogue metadata ...")
    jamendo = load_jamendo_licenses(MTG_RAW / "audio_licenses.txt")
    print(f"  jamendo licences: {len(jamendo):,}")
    m4a = load_m4a_metadata(M4A_RAW)
    with_sid = sum(1 for v in m4a.values() if v.get("spotify_id"))
    print(f"  music4all tracks: {len(m4a):,} ({with_sid:,} with spotify_id)")

    datasets = []
    for label, key, root in (
        ("MTG-Jamendo", "mtg_jamendo", MTG_ROOT),
        ("Music4All", "music4all", M4A_ROOT),
    ):
        print(f"\n{label}:")
        got = export_dataset(
            label, key, root,
            per_dataset=args.per_dataset, min_len=args.min_len, max_len=args.max_len,
            min_score=args.min_score, jamendo=jamendo, m4a=m4a,
        )
        if got:
            datasets.append(got)

    if not datasets:
        sys.exit("No chains exported.")

    payload = {
        "generated_by": "scripts/export_website_data.py",
        "instructions_folder": INSTR_FOLDER,
        "datasets": datasets,
    }
    out_path = out_dir / "chains.json"
    out_path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    size_kb = out_path.stat().st_size / 1024
    print(f"\nWrote {out_path.relative_to(REPO)} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
