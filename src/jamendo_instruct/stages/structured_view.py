from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

from jamendo_instruct.progress import StageTracker, rich_tqdm

if TYPE_CHECKING:
    from omegaconf import DictConfig
else:
    DictConfig = Any

CONF_DIR = str(Path(__file__).resolve().parents[3] / "conf")


def _log(cfg: DictConfig, message: str) -> None:
    if bool(cfg.stage.progress.enabled):
        print(f"[structured_view] {message}", flush=True)


def _parse_json_list(raw: str) -> List[str]:
    value = str(raw or "").strip()
    if not value:
        return []
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [str(x).strip() for x in data if str(x).strip()]


def _normalize_caption_text(text: str, cfg: DictConfig) -> str:
    out = str(text or "").strip()
    if bool(cfg.stage.caption_normalization.lowercase):
        out = out.lower()
    if bool(cfg.stage.caption_normalization.collapse_whitespace):
        out = re.sub(r"\s+", " ", out)
    if bool(cfg.stage.caption_normalization.strip_outer_punctuation):
        out = out.strip(" \t\r\n.,;:!?\"'()[]{}")
    return out.strip()


def _caption_word_len(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _parse_tag_list(row: Dict[str, str]) -> List[str]:
    tags_json = _parse_json_list(row.get("normalized_tags_json", ""))
    if tags_json:
        return tags_json
    raw_tags = str(row.get("tags", "") or "")
    parts = [part.strip() for part in raw_tags.split(",")]
    return [part for part in parts if part]


def _build_output_row(row: Dict[str, str], cfg: DictConfig) -> Dict[str, Any]:
    clip_id = str(row.get("clip_id", "") or "").strip() or str(row.get("track_id", "") or "").strip()
    track_id = str(row.get("track_id", "") or "").strip()
    caption = str(row.get("caption", "") or "").strip()
    primary_caption = str(row.get("primary_caption", "") or "").strip() or caption
    track_primary_caption = str(row.get("track_primary_caption", "") or "").strip() or primary_caption
    normalized_tags = _parse_tag_list(row)
    track_captions = _parse_json_list(row.get("track_captions_json", ""))
    clip_captions = _parse_json_list(row.get("captions_json", ""))

    if not track_captions:
        track_captions = clip_captions or ([caption] if caption else [])

    normalized_caption = _normalize_caption_text(caption, cfg)
    normalized_track_primary_caption = _normalize_caption_text(track_primary_caption, cfg)
    lyrics = str(row.get("lyrics", "") or "").strip()
    normalized_lyrics = _normalize_caption_text(lyrics, cfg) if lyrics else ""

    return {
        "clip_id": clip_id,
        "track_id": track_id,
        "file_path": str(row.get("file_path", "") or ""),
        "split": str(row.get("split", "") or ""),
        "caption": caption,
        "primary_caption": primary_caption,
        "track_primary_caption": track_primary_caption,
        "captions_json": json.dumps(clip_captions or ([caption] if caption else []), ensure_ascii=True),
        "track_captions_json": json.dumps(track_captions, ensure_ascii=True),
        "tags": ", ".join(normalized_tags),
        "normalized_tags_json": json.dumps(normalized_tags, ensure_ascii=True),
        "tag_count": len(normalized_tags),
        "normalized_caption": normalized_caption,
        "normalized_track_primary_caption": normalized_track_primary_caption,
        "caption_char_len": len(caption),
        "caption_word_len": _caption_word_len(caption),
        "track_caption_count": len(track_captions),
        "lyrics": lyrics,
        "normalized_lyrics": normalized_lyrics,
        "lyrics_language": str(row.get("lyrics_language", "") or ""),
        "lyrics_segments_json": str(row.get("lyrics_segments_json", "[]") or "[]"),
        "lyrics_status": str(row.get("lyrics_status", "") or ""),
        "lyrics_source": str(row.get("lyrics_source", "") or ""),
        "lyrics_error": str(row.get("lyrics_error", "") or ""),
        "start_time": str(row.get("start_time", "") or ""),
        "end_time": str(row.get("end_time", "") or ""),
        "title": str(row.get("title", "") or ""),
        "duration": str(row.get("duration", "") or ""),
        "artist_id": str(row.get("artist_id", "") or ""),
        "artist_name": str(row.get("artist_name", "") or ""),
        "release_date": str(row.get("release_date", "") or ""),
        "audio_url": str(row.get("audio_url", "") or ""),
        "audio_download_url": str(row.get("audio_download_url", "") or ""),
        "license_url": str(row.get("license_url", "") or ""),
        "vocals": str(row.get("vocals", "") or ""),
        "speed": str(row.get("speed", "") or ""),
    }


def run_structured_view(cfg: DictConfig) -> Dict[str, object]:
    in_csv = Path(str(cfg.stage.io.input_manifest_csv))
    out_dir = Path(str(cfg.stage.io.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / str(cfg.stage.io.output_manifest_csv)
    report_path = out_dir / str(cfg.stage.io.report_file)
    tracker = StageTracker(
        cfg,
        "structured_view",
        title="Build Structured Clip View",
        subtitle=f"input={in_csv}",
        total_steps=3,
    )

    if not in_csv.exists():
        raise FileNotFoundError(f"Input manifest CSV not found: {in_csv}")

    tracker.step("Read normalized manifest", detail=str(in_csv))
    with in_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)
    every_n = int(cfg.stage.progress.every_n_rows)

    fieldnames = [
        "clip_id",
        "track_id",
        "file_path",
        "split",
        "caption",
        "primary_caption",
        "track_primary_caption",
        "captions_json",
        "track_captions_json",
        "tags",
        "normalized_tags_json",
        "tag_count",
        "normalized_caption",
        "normalized_track_primary_caption",
        "caption_char_len",
        "caption_word_len",
        "track_caption_count",
        "lyrics",
        "normalized_lyrics",
        "lyrics_language",
        "lyrics_segments_json",
        "lyrics_status",
        "lyrics_source",
        "lyrics_error",
        "start_time",
        "end_time",
        "title",
        "duration",
        "artist_id",
        "artist_name",
        "release_date",
        "audio_url",
        "audio_download_url",
        "license_url",
        "vocals",
        "speed",
    ]

    counts = {
        "input_rows": total,
        "output_rows": 0,
        "unique_clips": 0,
        "unique_tracks": 0,
        "rows_with_tags": 0,
        "rows_without_tags": 0,
        "rows_with_track_context": 0,
    }
    seen_clips = set()
    seen_tracks = set()

    tracker.step("Normalize captions and tags", detail=f"{total:,} rows")
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        with rich_tqdm(cfg, total=total, desc="Build rows", unit="row") as progress:
            for i, row in enumerate(rows, start=1):
                out_row = _build_output_row(row, cfg)
                writer.writerow(out_row)
                counts["output_rows"] += 1
                seen_clips.add(out_row["clip_id"])
                seen_tracks.add(out_row["track_id"])
                if out_row["tag_count"] > 0:
                    counts["rows_with_tags"] += 1
                else:
                    counts["rows_without_tags"] += 1
                if out_row["track_caption_count"] > 1:
                    counts["rows_with_track_context"] += 1
                if every_n > 0 and i % every_n == 0:
                    _log(cfg, f"Rows processed: {i:,}")
                progress.update(1)

    counts["unique_clips"] = len(seen_clips)
    counts["unique_tracks"] = len(seen_tracks)

    report = {
        "stage": "structured_view",
        "input": {
            "input_manifest_csv": str(in_csv),
        },
        "counts": counts,
        "outputs": {
            "output_manifest_csv": str(out_csv),
            "report": str(report_path),
        },
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)

    tracker.step("Write outputs", detail=f"manifest={out_csv.name}, report={report_path.name}")
    tracker.finish(f"wrote {counts['output_rows']:,} rows")
    _log(
        cfg,
        f"Structured view complete. Wrote {counts['output_rows']:,} rows "
        f"across {counts['unique_tracks']:,} tracks",
    )
    return report


def _main_impl(cfg: DictConfig) -> None:
    report = run_structured_view(cfg)
    print(json.dumps({"status": "ok", "stage": "structured_view", "outputs": report["outputs"]}, indent=2))


def main() -> None:
    import hydra

    @hydra.main(version_base=None, config_path=CONF_DIR, config_name="config")
    def _wrapped(cfg: DictConfig) -> None:
        _main_impl(cfg)

    _wrapped()


if __name__ == "__main__":
    main()
