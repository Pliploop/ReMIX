from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen
from typing import Any, Dict, Iterable, List

import hydra
from jamendo_instruct.progress import StageTracker, rich_tqdm
from omegaconf import DictConfig

CONF_DIR = str(Path(__file__).resolve().parents[3] / "conf")


def _jsonl_iter(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON at {path}:{line_no}") from exc


def _log(cfg: DictConfig, message: str) -> None:
    if bool(cfg.stage.progress.enabled):
        print(f"[ingest] {message}", flush=True)


def _normalize_tag(tag: str, cfg: DictConfig) -> str:
    text = tag.strip()
    if cfg.stage.normalization.lowercase_tags:
        text = text.lower()
    if cfg.stage.normalization.strip_punctuation_tags:
        text = re.sub(r"[^a-z0-9\-\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _track_shards(cfg: DictConfig) -> List[Path]:
    root = Path(cfg.stage.tracks.metadata_dir)
    if not root.exists():
        raise FileNotFoundError(f"metadata_dir does not exist: {root}")

    excluded = set(cfg.stage.tracks.exclude_files)
    shards = [
        p
        for p in sorted(root.glob(cfg.stage.tracks.shard_glob))
        if p.name not in excluded and p.suffix == ".jsonl"
    ]
    if cfg.stage.tracks.max_shards is not None:
        shards = shards[: int(cfg.stage.tracks.max_shards)]
    return shards


def _extract_next_link(link_header: str) -> str:
    if not link_header:
        return ""
    for part in link_header.split(","):
        part = part.strip()
        if 'rel="next"' in part:
            m = re.match(r"<([^>]+)>", part)
            if m:
                return m.group(1)
    return ""


def _hf_list_repo_files(repo_id: str, timeout_sec: int) -> List[str]:
    url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main?recursive=1"
    files: List[str] = []
    seen = set()
    while url and url not in seen:
        seen.add(url)
        req = Request(url, headers={"User-Agent": "jamendo-instruct/0.1"})
        with urlopen(req, timeout=timeout_sec) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            files.extend(str(x.get("path", "")) for x in payload if x.get("path"))
            url = _extract_next_link(resp.headers.get("Link", ""))
    return files


def _download_file(url: str, dst: Path, timeout_sec: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "jamendo-instruct/0.1"})
    with urlopen(req, timeout=timeout_sec) as resp:
        with dst.open("wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)


def _maybe_download_metadata(cfg: DictConfig) -> Dict[str, int]:
    stats = {"downloaded": 0, "skipped_existing": 0}
    if not bool(cfg.stage.download.enabled):
        return stats

    root = Path(cfg.stage.tracks.metadata_dir)
    root.mkdir(parents=True, exist_ok=True)

    shards = _track_shards(cfg)
    captions_path = Path(cfg.stage.captions.file)
    need_shards = len(shards) == 0
    need_captions = not captions_path.exists() and bool(cfg.stage.download.include_captions_file)
    if not need_shards and not need_captions:
        _log(cfg, "Metadata download skipped (required files already present)")
        return stats

    repo_id = str(cfg.stage.download.repo_id)
    timeout_sec = int(cfg.stage.download.timeout_sec)
    all_files = _hf_list_repo_files(repo_id, timeout_sec=timeout_sec)
    jsonl_files = [p for p in all_files if p.endswith(".jsonl")]

    exclude = set(str(x) for x in cfg.stage.tracks.exclude_files)
    shard_files = [p for p in jsonl_files if p not in exclude]
    shard_files.sort()

    max_files = cfg.stage.download.max_files
    if max_files is not None:
        shard_files = shard_files[: int(max_files)]

    to_download: List[str] = []
    if need_shards:
        to_download.extend(shard_files)
    if need_captions and "final_caption30sec.jsonl" in jsonl_files:
        to_download.append("final_caption30sec.jsonl")

    total_download = len(to_download)
    with rich_tqdm(cfg, total=total_download, desc="Download files", unit="file") as progress:
        for rel_path in to_download:
            if bool(cfg.stage.download.allow_only_jsonl) and not rel_path.endswith(".jsonl"):
                raise RuntimeError(f"Refusing to download non-JSONL file: {rel_path}")
            dst = root / rel_path
            if dst.exists() and not bool(cfg.stage.download.overwrite):
                stats["skipped_existing"] += 1
                progress.update(1)
                continue
            url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{quote(rel_path)}"
            _log(cfg, f"Downloading {rel_path}")
            _download_file(url, dst=dst, timeout_sec=timeout_sec)
            stats["downloaded"] += 1
            progress.update(1)

    _log(
        cfg,
        f"Download complete: downloaded {stats['downloaded']}, skipped existing {stats['skipped_existing']}",
    )
    return stats


def _format_timestamp_key(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _make_clip_id(track_id: str, start_time: float, end_time: float, ordinal: int) -> str:
    return (
        f"{track_id}::"
        f"{_format_timestamp_key(start_time)}-"
        f"{_format_timestamp_key(end_time)}::"
        f"{ordinal}"
    )


def _load_captions_map(
    captions_file: Path, min_chars: int, cfg: DictConfig
) -> tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int]]:
    if not captions_file.exists():
        raise FileNotFoundError(f"captions file does not exist: {captions_file}")

    by_track: Dict[str, List[tuple[float, float, str]]] = defaultdict(list)
    total = 0
    kept = 0
    every_n = int(cfg.stage.progress.every_n_rows)
    _log(cfg, f"Loading captions from {captions_file}")
    for row in _jsonl_iter(captions_file):
        total += 1
        track_id = str(row.get("id", "")).strip()
        caption = str(row.get("caption", "")).strip()
        if not track_id or len(caption) < min_chars:
            continue
        start_time = float(row.get("start_time", 0.0) or 0.0)
        end_time = float(row.get("end_time", 0.0) or 0.0)
        by_track[track_id].append((start_time, end_time, caption))
        kept += 1
        if every_n > 0 and total % every_n == 0:
            _log(cfg, f"Caption rows processed: {total:,} (kept {kept:,})")

    captions_map: Dict[str, List[Dict[str, Any]]] = {}
    for track_id, entries in by_track.items():
        entries.sort(key=lambda x: (x[0], x[1]))
        clip_records: List[Dict[str, Any]] = []
        seen_captions = set()
        for ordinal, (start_time, end_time, caption) in enumerate(entries, start=1):
            if cfg.stage.normalization.dedupe_captions and caption in seen_captions:
                continue
            seen_captions.add(caption)
            clip_records.append(
                {
                    "clip_id": _make_clip_id(track_id, start_time, end_time, ordinal),
                    "track_id": track_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "caption": caption,
                }
            )
        captions_map[track_id] = clip_records

    kept_clips = sum(len(v) for v in captions_map.values())
    _log(cfg, f"Captions ready: {kept_clips:,} clip rows across {len(captions_map):,} tracks")
    stats = {
        "caption_rows_total": total,
        "caption_rows_loaded": kept,
        "caption_unique_tracks": len(captions_map),
        "caption_clip_rows_kept": kept_clips,
    }
    return captions_map, stats


def _extract_track_base(row: Dict[str, Any], cfg: DictConfig) -> Dict[str, Any]:
    fields = cfg.dataset.fields
    mfields = cfg.dataset.musicinfo_fields

    track_id = str(row.get(fields.track_id, "")).strip()
    musicinfo = row.get(fields.musicinfo, {}) or {}
    tags_node = musicinfo.get("tags", {}) if isinstance(musicinfo, dict) else {}

    tags: List[str] = []
    for key in [mfields.tags_genres, mfields.tags_instruments, mfields.tags_vartags]:
        vals = tags_node.get(key, []) if isinstance(tags_node, dict) else []
        if not isinstance(vals, list):
            continue
        tags.extend(str(x) for x in vals)

    normalized_tags = [_normalize_tag(t, cfg) for t in tags]
    normalized_tags = [t for t in normalized_tags if t]
    if cfg.stage.normalization.dedupe_tags:
        normalized_tags = _dedupe_preserve_order(normalized_tags)

    return {
        "track_id": track_id,
        "title": str(row.get(fields.title, "") or ""),
        "duration": row.get(fields.duration),
        "artist_id": str(row.get(fields.artist_id, "") or ""),
        "artist_name": str(row.get(fields.artist_name, "") or ""),
        "release_date": str(row.get(fields.release_date, "") or ""),
        "audio_url": str(row.get(fields.audio_url, "") or ""),
        "audio_download_url": str(row.get(fields.audio_download_url, "") or ""),
        "license_url": str(row.get(fields.license_url, "") or ""),
        "vocals": str(musicinfo.get(mfields.vocals, "") if isinstance(musicinfo, dict) else ""),
        "speed": str(musicinfo.get(mfields.speed, "") if isinstance(musicinfo, dict) else ""),
        "lyrics": str(row.get("lyrics", "") or ""),
        "lyrics_language": str(row.get("lyrics_language", "") or ""),
        "lyrics_segments_json": str(row.get("lyrics_segments_json", "[]") or "[]"),
        "lyrics_status": str(row.get("lyrics_status", "") or ""),
        "lyrics_source": str(row.get("lyrics_source", "") or ""),
        "lyrics_error": str(row.get("lyrics_error", "") or ""),
        "normalized_tags": normalized_tags,
        "source_split": str(row.get(fields.source_split, "") or ""),
        "raw_metadata": row,
    }


def _resolve_audio_path(track_id: str, cfg: DictConfig) -> str:
    root = Path(cfg.stage.audio.root)
    mode = str(cfg.stage.audio.path_mode)
    if mode == "track_id_mp3" or mode == "track_id_wav":
        return str(root / cfg.stage.audio.file_template.format(track_id=track_id))
    if mode == "track_id_suffix_dir":
        shard = track_id[-2:].zfill(2)
        return str(root / cfg.stage.audio.file_template.format(track_id=track_id, shard=shard))
    return str(root / cfg.stage.audio.file_template.format(track_id=track_id))


def _pick_primary_caption(captions: List[str], strategy: str) -> str:
    if not captions:
        return ""
    if strategy == "first":
        return captions[0]
    if strategy == "longest":
        return max(captions, key=len)
    return captions[0]


def _get_captions_for_track(captions_map: Dict[str, List[Dict[str, Any]]], track_id: str) -> List[Dict[str, Any]]:
    return captions_map.get(track_id, [])


def _assign_splits(records: List[Dict[str, Any]], cfg: DictConfig) -> None:
    mode = str(cfg.stage.split.mode)

    if mode == "source_column":
        for r in records:
            split = (r.get("source_split") or "").strip().lower()
            if split not in {"train", "val", "test"}:
                split = "train"
            r["split"] = split
        return

    if mode != "custom_track_grouped":
        raise ValueError(f"Unknown split.mode: {mode}")

    group_key = str(cfg.stage.split.group_key)
    rng = random.Random(int(cfg.stage.split.seed))

    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        group_val = str(rec.get(group_key, "") or "").strip() or "__unknown__"
        groups[group_val].append(idx)

    group_ids = list(groups.keys())
    rng.shuffle(group_ids)

    total = len(records)
    target_train = int(total * float(cfg.stage.split.train_ratio))
    target_val = int(total * float(cfg.stage.split.val_ratio))

    assigned_train = 0
    assigned_val = 0
    for g in group_ids:
        idxs = groups[g]
        if assigned_train < target_train:
            split = "train"
            assigned_train += len(idxs)
        elif assigned_val < target_val:
            split = "val"
            assigned_val += len(idxs)
        else:
            split = "test"
        for i in idxs:
            records[i]["split"] = split


def run_ingest(cfg: DictConfig) -> Dict[str, Any]:
    out_dir = Path(cfg.stage.io.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / cfg.stage.io.normalized_manifest_file
    report_path = out_dir / cfg.stage.io.ingest_report_file
    dropped_path = out_dir / cfg.stage.io.dropped_rows_file
    tracker = StageTracker(
        cfg,
        "ingest",
        title="Ingest Metadata And Captions",
        subtitle=f"output_dir={out_dir}",
        total_steps=6,
    )

    _log(cfg, f"Starting Stage 1 ingest. Output dir: {out_dir}")
    tracker.step("Check local metadata", detail=str(cfg.stage.tracks.metadata_dir))
    download_stats = _maybe_download_metadata(cfg)
    tracker.step("Resolve metadata shards", detail=f"captions={cfg.stage.captions.file}")
    shards = _track_shards(cfg)
    if not shards:
        root = Path(cfg.stage.tracks.metadata_dir)
        all_jsonl = sorted(root.glob(cfg.stage.tracks.shard_glob))
        excluded = set(cfg.stage.tracks.exclude_files)
        excluded_hits = [p.name for p in all_jsonl if p.name in excluded]
        raise RuntimeError(
            "No metadata shards found. "
            f"metadata_dir={root}, shard_glob={cfg.stage.tracks.shard_glob}, "
            f"exclude_files={list(excluded)}. "
            f"Matched {len(all_jsonl)} JSONL files, excluded {len(excluded_hits)}."
        )

    _log(cfg, f"Using {len(shards):,} metadata shard files")
    tracker.step("Load captions", detail=str(cfg.stage.captions.file))
    captions_map, caption_stats = _load_captions_map(
        captions_file=Path(cfg.stage.captions.file),
        min_chars=int(cfg.stage.captions.min_caption_chars),
        cfg=cfg,
    )

    track_records_by_id: Dict[str, Dict[str, Any]] = {}
    loaded = 0
    every_n = int(cfg.stage.progress.every_n_rows)
    total_shards = len(shards)
    tracker.step("Read metadata shards", detail=f"{total_shards:,} shard files")
    with rich_tqdm(cfg, total=total_shards, desc="Read shards", unit="shard") as progress:
        for shard_i, shard in enumerate(shards, start=1):
            _log(cfg, f"Reading shard: {shard.name}")
            for row in _jsonl_iter(shard):
                rec = _extract_track_base(row, cfg)
                if rec["track_id"]:
                    track_records_by_id[rec["track_id"]] = rec
                    loaded += 1
                    if every_n > 0 and loaded % every_n == 0:
                        _log(cfg, f"Track rows loaded: {loaded:,} (unique {len(track_records_by_id):,})")
            progress.update(1)

    track_records = list(track_records_by_id.values())
    duplicate_track_rows = loaded - len(track_records)
    _log(
        cfg,
        f"Loaded {loaded:,} track rows across {len(track_records):,} unique tracks "
        f"({duplicate_track_rows:,} duplicate shard rows overwritten by latest record)",
    )

    dropped_rows: List[Dict[str, str]] = []
    counts = Counter()
    kept_records: List[Dict[str, Any]] = []
    total_filter = len(track_records)
    tracker.step("Filter tracks and assign clip metadata", detail=f"{total_filter:,} unique tracks")
    with rich_tqdm(cfg, total=total_filter, desc="Filter tracks", unit="track") as progress:
        for i, rec in enumerate(track_records, start=1):
            track_id = rec["track_id"]
            clip_records = _get_captions_for_track(captions_map, track_id)
            if clip_records:
                counts["tracks_with_captions"] += 1
            else:
                counts["tracks_without_captions"] += 1

            if cfg.stage.filters.drop_missing_caption and not clip_records:
                dropped_rows.append({"track_id": track_id, "reason": "missing_caption"})
                counts["dropped_missing_caption"] += 1
                progress.update(1)
                continue
            if cfg.stage.filters.drop_missing_tags and not rec["normalized_tags"]:
                dropped_rows.append({"track_id": track_id, "reason": "missing_tags"})
                counts["dropped_missing_tags"] += 1
                progress.update(1)
                continue

            file_path = _resolve_audio_path(track_id, cfg)
            if cfg.stage.audio.verify_exists and not Path(file_path).exists():
                if cfg.stage.filters.drop_unavailable_audio:
                    dropped_rows.append({"track_id": track_id, "reason": "audio_not_found"})
                    counts["dropped_audio_missing"] += 1
                    progress.update(1)
                    continue

            rec["file_path"] = file_path
            track_captions = [clip["caption"] for clip in clip_records]
            rec["primary_caption"] = _pick_primary_caption(track_captions, str(cfg.stage.captions.primary_caption))
            rec["track_captions"] = track_captions
            rec["clip_records"] = clip_records
            kept_records.append(rec)

            counts["clips_kept"] += len(clip_records)
            if str(rec.get("lyrics", "") or "").strip():
                counts["tracks_with_lyrics"] += 1
                counts["clips_with_lyrics"] += len(clip_records)
            elif str(rec.get("lyrics_status", "") or "").strip():
                counts[f"tracks_lyrics_status_{str(rec.get('lyrics_status', '') or '').strip()}"] += 1
            if len(clip_records) > 1:
                counts["multi_caption_tracks"] += 1
            if every_n > 0 and i % every_n == 0:
                _log(
                    cfg,
                    f"Tracks filtered: {i:,} (kept {len(kept_records):,} tracks / {counts['clips_kept']:,} clips)",
                )
            progress.update(1)

    _assign_splits(kept_records, cfg)
    _log(cfg, f"Split assignment complete for {len(kept_records):,} tracks")

    split_counts = Counter()
    written = 0
    total_manifest = counts["clips_kept"]
    tracker.step("Write normalized manifest", detail=f"{total_manifest:,} clip rows")
    fieldnames = [
        "clip_id",
        "track_id",
        "file_path",
        "caption",
        "primary_caption",
        "track_primary_caption",
        "tags",
        "split",
        "captions_json",
        "track_captions_json",
        "normalized_tags_json",
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
        "lyrics",
        "lyrics_language",
        "lyrics_segments_json",
        "lyrics_status",
        "lyrics_source",
        "lyrics_error",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_f:
        writer = csv.DictWriter(manifest_f, fieldnames=fieldnames)
        writer.writeheader()
        with rich_tqdm(cfg, total=total_manifest, desc="Write manifest", unit="clip") as progress:
            for rec in kept_records:
                track_id = rec["track_id"]
                for clip in rec["clip_records"]:
                    row = {
                        "clip_id": clip["clip_id"],
                        "track_id": track_id,
                        "file_path": rec["file_path"],
                        "caption": clip["caption"],
                        "primary_caption": clip["caption"],
                        "track_primary_caption": rec["primary_caption"],
                        "tags": ", ".join(rec["normalized_tags"]),
                        "split": rec["split"],
                        "captions_json": json.dumps([clip["caption"]], ensure_ascii=True),
                        "track_captions_json": json.dumps(rec["track_captions"], ensure_ascii=True),
                        "normalized_tags_json": json.dumps(rec["normalized_tags"], ensure_ascii=True),
                        "start_time": clip["start_time"],
                        "end_time": clip["end_time"],
                        "title": rec["title"],
                        "duration": rec["duration"],
                        "artist_id": rec["artist_id"],
                        "artist_name": rec["artist_name"],
                        "release_date": rec["release_date"],
                        "audio_url": rec["audio_url"],
                        "audio_download_url": rec["audio_download_url"],
                        "license_url": rec["license_url"],
                        "vocals": rec["vocals"],
                        "speed": rec["speed"],
                        "lyrics": rec["lyrics"],
                        "lyrics_language": rec["lyrics_language"],
                        "lyrics_segments_json": rec["lyrics_segments_json"],
                        "lyrics_status": rec["lyrics_status"],
                        "lyrics_source": rec["lyrics_source"],
                        "lyrics_error": rec["lyrics_error"],
                    }
                    writer.writerow(row)
                    counts["kept"] += 1
                    split_counts[rec["split"]] += 1
                    written += 1
                    if every_n > 0 and written % every_n == 0:
                        _log(cfg, f"Manifest rows written: {written:,}")
                    progress.update(1)

    if cfg.stage.io.write_dropped_rows and dropped_rows:
        with dropped_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["track_id", "reason"])
            writer.writeheader()
            writer.writerows(dropped_rows)

    report = {
        "stage": "ingest",
        "input": {
            "metadata_dir": str(cfg.stage.tracks.metadata_dir),
            "num_shards": len(shards),
            "captions_file": str(cfg.stage.captions.file),
            "download": download_stats,
        },
        "counts": {
            "tracks_loaded_raw": loaded,
            "unique_tracks_loaded": len(track_records),
            "tracks_loaded": len(kept_records),
            "clips_loaded": counts["clips_kept"],
            "duplicate_track_rows_removed": duplicate_track_rows,
            "tracks_kept": len(kept_records),
            "clips_kept": counts["kept"],
            "tracks_dropped": len(dropped_rows),
            "tracks_with_captions": counts["tracks_with_captions"],
            "tracks_without_captions": counts["tracks_without_captions"],
            "dropped_missing_caption": counts["dropped_missing_caption"],
            "dropped_missing_tags": counts["dropped_missing_tags"],
            "dropped_audio_missing": counts["dropped_audio_missing"],
            "tracks_with_multiple_captions": counts["multi_caption_tracks"],
            "tracks_with_lyrics": counts["tracks_with_lyrics"],
            "clips_with_lyrics": counts["clips_with_lyrics"],
            "tracks_lyrics_status_empty": counts["tracks_lyrics_status_empty"],
            "tracks_lyrics_status_missing": counts["tracks_lyrics_status_missing"],
            "tracks_lyrics_status_failed": counts["tracks_lyrics_status_failed"],
        },
        "split_counts": dict(split_counts),
        "caption_stats": caption_stats,
        "outputs": {
            "normalized_manifest": str(manifest_path),
            "ingest_report": str(report_path),
            "dropped_rows": str(dropped_path) if dropped_rows else None,
        },
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)

    tracker.finish(
        f"kept {len(kept_records):,} tracks / {counts['kept']:,} clips"
    )
    _log(
        cfg,
        f"Ingest complete. Kept {len(kept_records):,} tracks / {counts['kept']:,} clips. Splits: "
        f"train={split_counts.get('train', 0):,}, val={split_counts.get('val', 0):,}, "
        f"test={split_counts.get('test', 0):,}",
    )
    return report


@hydra.main(version_base=None, config_path=CONF_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    report = run_ingest(cfg)
    print(json.dumps({"status": "ok", "stage": "ingest", "kept": report["counts"]["tracks_kept"]}, indent=2))


if __name__ == "__main__":
    main()
