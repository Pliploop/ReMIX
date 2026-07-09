#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence


DEFAULT_MTG_ROOT = Path("/data/EECS-Pauwels-C4DM/mtg-jamendo-raw/mtg-jamendo-dataset")
DEFAULT_OUTPUT_DIR = Path("/gpfs/scratch/acw749/datasets/mtg_jamendo_instruct/metadata")


def _log(message: str) -> None:
    print(f"[mtg-prep] {message}", flush=True)


def _tag_parts(raw: str) -> tuple[list[str], list[str], list[str]]:
    genres: list[str] = []
    instruments: list[str] = []
    vartags: list[str] = []
    for item in str(raw or "").split():
        if "---" not in item:
            continue
        category, value = item.split("---", 1)
        value = value.strip()
        if not value:
            continue
        if category == "genre":
            genres.append(value)
        elif category == "instrument":
            instruments.append(value)
        else:
            vartags.append(value)
    return genres, instruments, vartags


def _speed_label(_duration: str) -> str:
    return ""


def _vocals_label(instruments: Iterable[str]) -> str:
    lowered = {str(item).lower() for item in instruments}
    return "vocal" if "voice" in lowered else ""


def _read_split_rows(split_dir: Path, subset: str) -> Iterable[tuple[str, Dict[str, str]]]:
    names = [("train", "train"), ("validation", "val"), ("test", "test")]
    for suffix, split in names:
        path = split_dir / f"{subset}-{suffix}.tsv"
        if not path.exists():
            raise FileNotFoundError(f"Required MTG split file not found: {path}")
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            if header is None or header[:6] != ["TRACK_ID", "ARTIST_ID", "ALBUM_ID", "PATH", "DURATION", "TAGS"]:
                raise ValueError(f"Unexpected MTG header in {path}: {header}")
            for values in reader:
                padded = list(values[:5]) + [""] * max(0, 5 - len(values[:5]))
                tags = " ".join(str(value or "").strip() for value in values[5:] if str(value or "").strip())
                yield split, {
                    "TRACK_ID": padded[0],
                    "ARTIST_ID": padded[1],
                    "ALBUM_ID": padded[2],
                    "PATH": padded[3],
                    "DURATION": padded[4],
                    "TAGS": tags,
                }


def _record(row: Dict[str, str], *, split: str, audio_root: Path) -> Dict[str, Any]:
    track_id = str(row.get("TRACK_ID", "") or "").strip()
    rel_path = str(row.get("PATH", "") or "").strip()
    audio_path = audio_root / rel_path
    genres, instruments, vartags = _tag_parts(row.get("TAGS", ""))
    return {
        "id": track_id,
        "name": "",
        "duration": row.get("DURATION", ""),
        "artist_id": row.get("ARTIST_ID", ""),
        "artist_name": row.get("ARTIST_ID", ""),
        "releasedate": "",
        "audio": str(audio_path),
        "audiodownload": str(audio_path),
        "license_ccurl": "",
        "split": split,
        "lyrics": "",
        "lyrics_language": "",
        "lyrics_segments_json": "[]",
        "lyrics_status": "missing",
        "lyrics_source": "",
        "lyrics_error": "",
        "musicinfo": {
            "vocalinstrumental": _vocals_label(instruments),
            "speed": _speed_label(row.get("DURATION", "")),
            "album_id": row.get("ALBUM_ID", ""),
            "path": rel_path,
            "tags": {
                "genres": genres,
                "instruments": instruments,
                "vartags": vartags,
            },
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MTG-Jamendo metadata for the Jamendo-Instruct ingest stage.")
    parser.add_argument("--mtg-root", type=Path, default=DEFAULT_MTG_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-index", type=int, default=0)
    parser.add_argument("--subset", default="autotagging")
    parser.add_argument("--output-jsonl", default="mtg_jamendo_tracks.jsonl")
    parser.add_argument("--report-json", default="mtg_jamendo_metadata_report.json")
    parser.add_argument("--require-audio", action="store_true", default=True)
    parser.add_argument("--no-require-audio", dest="require_audio", action="store_false")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    split_dir = args.mtg_root / "data" / "splits" / f"split-{args.split_index}"
    audio_root = args.mtg_root / "mp3"
    if not split_dir.exists():
        raise FileNotFoundError(f"MTG split directory not found: {split_dir}")
    if not audio_root.exists():
        raise FileNotFoundError(f"MTG audio root not found: {audio_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.output_dir / args.output_jsonl
    report_json = args.output_dir / args.report_json
    counts: Counter[str] = Counter()
    seen: set[str] = set()

    _log(f"Reading {args.subset} split-{args.split_index} metadata from {split_dir}")
    with out_jsonl.open("w", encoding="utf-8") as f:
        for split, row in _read_split_rows(split_dir, args.subset):
            track_id = str(row.get("TRACK_ID", "") or "").strip()
            if not track_id or track_id in seen:
                counts["skipped_duplicate_or_missing_track_id"] += 1
                continue
            record = _record(row, split=split, audio_root=audio_root)
            audio_path = Path(record["audiodownload"])
            if args.require_audio and not audio_path.exists():
                counts["skipped_missing_audio"] += 1
                continue
            seen.add(track_id)
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
            counts["written"] += 1
            counts[f"split_{split}"] += 1
            tags = record["musicinfo"]["tags"]
            if tags["genres"]:
                counts["with_genres"] += 1
            if tags["instruments"]:
                counts["with_instruments"] += 1
            if tags["vartags"]:
                counts["with_vartags"] += 1
            if record["musicinfo"]["vocalinstrumental"] == "vocal":
                counts["vocal_by_voice_tag"] += 1

    report = {
        "mtg_root": str(args.mtg_root),
        "split_dir": str(split_dir),
        "subset": args.subset,
        "audio_root": str(audio_root),
        "output_jsonl": str(out_jsonl),
        "caption_jsonl_expected": str(args.output_dir / "final_caption30sec.jsonl"),
        "counts": dict(counts),
        "ingest_overrides": {
            "dataset": "mtgjamendo",
            "stage.tracks.metadata_dir": str(args.output_dir),
            "stage.tracks.shard_glob": args.output_jsonl,
            "stage.tracks.exclude_files": ["final_caption30sec.jsonl"],
            "stage.download.enabled": False,
            "stage.captions.file": str(args.output_dir / "final_caption30sec.jsonl"),
            "stage.audio.root": str(audio_root),
            "stage.audio.path_mode": "audio_download_url",
            "stage.audio.verify_exists": True,
            "stage.split.mode": "source_column",
            "stage.filters.drop_unavailable_audio": True,
        },
    }
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    _log(f"Done. Wrote {counts['written']:,} rows; report={report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
