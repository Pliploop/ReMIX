#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence


DEFAULT_MUSIC4ALL_ROOT = Path("/data/EECS-Pauwels-C4DM/music4all")
DEFAULT_OUTPUT_DIR = Path("/gpfs/scratch/acw749/datasets/music4all_instruct/metadata")


def _log(message: str) -> None:
    print(f"[music4all-prep] {message}", flush=True)


def _read_tsv(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        out: Dict[str, Dict[str, str]] = {}
        for row in reader:
            track_id = str(row.get("id", "") or "").strip()
            if track_id:
                out[track_id] = {str(k): str(v or "") for k, v in row.items()}
    return out


def _split_csv_values(value: str) -> list[str]:
    out: list[str] = []
    for item in str(value or "").split(","):
        text = " ".join(item.strip().split())
        if text and text not in out:
            out.append(text)
    return out


def _duration_seconds(row: Dict[str, str]) -> float | str:
    raw_ms = str(row.get("duration_ms", "") or "").strip()
    if not raw_ms:
        return ""
    try:
        return round(float(raw_ms) / 1000.0, 3)
    except ValueError:
        return ""


def _speed_label(row: Dict[str, str]) -> str:
    raw = str(row.get("tempo", "") or "").strip()
    try:
        tempo = float(raw)
    except ValueError:
        return ""
    if tempo < 90:
        return "slow"
    if tempo > 140:
        return "fast"
    return "medium"


def _vocals_label(track_id: str, tags: Iterable[str], lyrics_dir: Path) -> str:
    lowered = {tag.lower() for tag in tags}
    if any("instrumental" in tag for tag in lowered):
        return "instrumental"
    if (lyrics_dir / f"{track_id}.txt").exists():
        return "vocal"
    return ""


def _record(
    *,
    track_id: str,
    info: Dict[str, str],
    metadata: Dict[str, str],
    genres_row: Dict[str, str],
    tags_row: Dict[str, str],
    lang_row: Dict[str, str],
    audio_path: Path,
    lyrics_dir: Path,
) -> Dict[str, Any]:
    genres = _split_csv_values(genres_row.get("genres", ""))
    tags = _split_csv_values(tags_row.get("tags", ""))
    tag_genres = genres
    tag_vartags = [tag for tag in tags if tag not in set(genres)]
    all_tags = genres + [tag for tag in tags if tag not in set(genres)]
    return {
        "id": track_id,
        "name": info.get("song", ""),
        "duration": _duration_seconds(metadata),
        "artist_id": info.get("artist", ""),
        "artist_name": info.get("artist", ""),
        "releasedate": metadata.get("release", ""),
        "audio": str(audio_path),
        "audiodownload": str(audio_path),
        "license_ccurl": "",
        "split": "",
        "musicinfo": {
            "vocalinstrumental": _vocals_label(track_id, all_tags, lyrics_dir),
            "speed": _speed_label(metadata),
            "language": lang_row.get("lang", ""),
            "spotify_id": metadata.get("spotify_id", ""),
            "popularity": metadata.get("popularity", ""),
            "danceability": metadata.get("danceability", ""),
            "energy": metadata.get("energy", ""),
            "key": metadata.get("key", ""),
            "mode": metadata.get("mode", ""),
            "valence": metadata.get("valence", ""),
            "tempo": metadata.get("tempo", ""),
            "album_name": info.get("album_name", ""),
            "tags": {
                "genres": tag_genres,
                "instruments": [],
                "vartags": tag_vartags,
            },
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Music4All metadata for the Jamendo-Instruct ingest stage.")
    parser.add_argument("--music4all-root", type=Path, default=DEFAULT_MUSIC4ALL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-jsonl", default="music4all_tracks.jsonl")
    parser.add_argument("--report-json", default="music4all_metadata_report.json")
    parser.add_argument("--require-audio", action="store_true", default=True)
    parser.add_argument("--no-require-audio", dest="require_audio", action="store_false")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.music4all_root
    audio_dir = root / "audios"
    lyrics_dir = root / "lyrics"
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / str(args.output_jsonl)
    report_json = out_dir / str(args.report_json)

    required = ["id_information.csv", "id_metadata.csv", "id_tags.csv", "id_genres.csv", "id_lang.csv"]
    for name in required:
        path = root / name
        if not path.exists():
            raise FileNotFoundError(f"Required Music4All metadata file not found: {path}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"Music4All audio directory not found: {audio_dir}")

    _log(f"Reading Music4All metadata from {root}")
    information = _read_tsv(root / "id_information.csv")
    metadata = _read_tsv(root / "id_metadata.csv")
    tags = _read_tsv(root / "id_tags.csv")
    genres = _read_tsv(root / "id_genres.csv")
    langs = _read_tsv(root / "id_lang.csv")
    audio_ids = {path.stem for path in audio_dir.glob("*.mp3")}

    ids = sorted(set(information) | set(metadata) | set(tags) | set(genres) | set(langs))
    counts = {
        "input_unique_ids": len(ids),
        "audio_files": len(audio_ids),
        "written": 0,
        "skipped_missing_audio": 0,
        "with_genres": 0,
        "with_tags": 0,
        "with_language": 0,
        "vocal": 0,
        "instrumental": 0,
    }

    _log(f"Writing pipeline metadata shard: {out_jsonl}")
    with out_jsonl.open("w", encoding="utf-8") as f:
        for track_id in ids:
            audio_path = audio_dir / f"{track_id}.mp3"
            if args.require_audio and track_id not in audio_ids:
                counts["skipped_missing_audio"] += 1
                continue
            rec = _record(
                track_id=track_id,
                info=information.get(track_id, {}),
                metadata=metadata.get(track_id, {}),
                genres_row=genres.get(track_id, {}),
                tags_row=tags.get(track_id, {}),
                lang_row=langs.get(track_id, {}),
                audio_path=audio_path,
                lyrics_dir=lyrics_dir,
            )
            vocal_status = str(rec["musicinfo"]["vocalinstrumental"])
            if vocal_status in counts:
                counts[vocal_status] += 1
            if rec["musicinfo"]["tags"]["genres"]:
                counts["with_genres"] += 1
            if rec["musicinfo"]["tags"]["vartags"]:
                counts["with_tags"] += 1
            if rec["musicinfo"]["language"]:
                counts["with_language"] += 1
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")
            counts["written"] += 1

    report = {
        "music4all_root": str(root),
        "audio_dir": str(audio_dir),
        "lyrics_dir": str(lyrics_dir),
        "output_jsonl": str(out_jsonl),
        "caption_jsonl_expected": str(out_dir / "final_caption30sec.jsonl"),
        "counts": counts,
        "ingest_overrides": {
            "dataset": "jamendomaxcaps",
            "stage.tracks.metadata_dir": str(out_dir),
            "stage.tracks.shard_glob": "music4all_tracks.jsonl",
            "stage.tracks.exclude_files": ["final_caption30sec.jsonl"],
            "stage.download.enabled": False,
            "stage.captions.file": str(out_dir / "final_caption30sec.jsonl"),
            "stage.audio.root": str(audio_dir),
            "stage.audio.path_mode": "track_id_mp3",
            "stage.audio.file_template": "{track_id}.mp3",
            "stage.audio.verify_exists": True,
            "stage.filters.drop_unavailable_audio": True,
        },
    }
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    _log(f"Done. Wrote {counts['written']:,} rows; report={report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
