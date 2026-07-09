#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

from music4all_caption_audio import (
    AudioItem,
    DEFAULT_MODEL_ID,
    DEFAULT_PROMPT,
    _caption_batch,
    _claim_path,
    _error_record,
    _is_fatal_cuda_error,
    _load_completed_ids,
    _load_model,
    _progress,
    _record_for_item,
    _release_claim,
    _try_claim_item,
    _write_jsonl_row,
    _write_report,
)


DEFAULT_METADATA_JSONL = Path("/gpfs/scratch/acw749/datasets/mtg_jamendo_instruct/metadata/mtg_jamendo_tracks.jsonl")
DEFAULT_OUTPUT_JSONL = Path("/gpfs/scratch/acw749/datasets/mtg_jamendo_instruct/metadata/final_caption30sec.jsonl")


def _log(message: str) -> None:
    logging.info(message)


def _discover_audio(metadata_jsonl: Path) -> list[AudioItem]:
    if not metadata_jsonl.exists():
        raise FileNotFoundError(f"Metadata JSONL not found: {metadata_jsonl}")
    out: list[AudioItem] = []
    seen: set[str] = set()
    with metadata_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {metadata_jsonl}:{line_no}") from exc
            track_id = str(row.get("id", "") or "").strip()
            audio_path = Path(str(row.get("audiodownload", "") or row.get("audio", "") or "").strip())
            if not track_id or not str(audio_path) or track_id in seen:
                continue
            out.append(AudioItem(track_id=track_id, audio_path=audio_path))
            seen.add(track_id)
    return out


def _batched(items: Sequence[AudioItem], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Caption MTG-Jamendo audio clips with Audio Flamingo Next and write ingest-compatible JSONL.",
    )
    parser.add_argument("--metadata-jsonl", type=Path, default=DEFAULT_METADATA_JSONL)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--error-jsonl", type=Path, default=None)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--clip-seconds", type=float, default=30.0)
    parser.add_argument("--audio-sampling-rate", type=int, default=16000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--claim-dir", type=Path, default=None)
    parser.add_argument("--claim-stale-seconds", type=float, default=86400.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--log-caption-every", type=int, default=100)
    parser.add_argument("--torch-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[mtg-caption] %(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.offset < 0:
        raise ValueError("--offset must be >= 0")

    error_jsonl = args.error_jsonl or args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".errors.jsonl")
    report_json = args.report_json or args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".report.json")
    claim_dir = args.claim_dir or args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".claims")
    stale_seconds = None if args.claim_stale_seconds <= 0 else float(args.claim_stale_seconds)

    items = _discover_audio(args.metadata_jsonl)[args.offset :]
    if args.max_items is not None:
        items = items[: max(0, int(args.max_items))]
    completed = set() if args.overwrite else _load_completed_ids(args.output_jsonl)
    todo = [item for item in items if item.track_id not in completed]
    _log(f"Metadata JSONL: {args.metadata_jsonl}")
    _log(f"Selected tracks: {len(items):,}; completed: {len(completed):,}; remaining: {len(todo):,}")

    report = {
        "status": "planned" if args.dry_run else "running",
        "metadata_jsonl": str(args.metadata_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "error_jsonl": str(error_jsonl),
        "claim_dir": str(claim_dir),
        "model_id": args.model_id,
        "prompt": args.prompt,
        "selected_audio_count": len(items),
        "completed_count_at_start": len(completed),
        "todo_count_at_start": len(todo),
        "written": 0,
        "failed": 0,
        "skipped_already_completed": 0,
        "skipped_claimed": 0,
    }
    _write_report(report_json, report)
    if args.dry_run or not todo:
        report["status"] = "complete"
        _write_report(report_json, report)
        return 0

    processor, model, torch = _load_model(args.model_id, args.torch_dtype, args.device_map, bool(args.trust_remote_code))
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    error_jsonl.parent.mkdir(parents=True, exist_ok=True)
    claim_dir.mkdir(parents=True, exist_ok=True)

    with args.output_jsonl.open("a", encoding="utf-8") as out_f, error_jsonl.open("a", encoding="utf-8") as err_f:
        for batch_items in _progress(_batched(todo, args.batch_size), total=(len(todo) + args.batch_size - 1) // args.batch_size, enabled=not args.no_progress):
            current_completed = set() if args.overwrite else _load_completed_ids(args.output_jsonl)
            claim_candidates = [item for item in batch_items if item.track_id not in current_completed]
            report["skipped_already_completed"] += len(batch_items) - len(claim_candidates)
            claimed = [item for item in claim_candidates if _try_claim_item(item, claim_dir=claim_dir, stale_seconds=stale_seconds)]
            report["skipped_claimed"] += len(claim_candidates) - len(claimed)
            if not claimed:
                continue
            try:
                captions = _caption_batch(
                    claimed,
                    processor=processor,
                    model=model,
                    torch=torch,
                    prompt=args.prompt,
                    clip_seconds=args.clip_seconds,
                    audio_sampling_rate=args.audio_sampling_rate,
                    max_new_tokens=args.max_new_tokens,
                    repetition_penalty=args.repetition_penalty,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                for item, caption in zip(claimed, captions):
                    _write_jsonl_row(out_f, _record_for_item(item, caption, model_id=args.model_id, prompt=args.prompt, clip_seconds=args.clip_seconds))
                    report["written"] += 1
                    if args.log_caption_every > 0 and report["written"] % args.log_caption_every == 0:
                        _log(f"Wrote {report['written']:,} captions")
            except Exception as exc:
                for item in claimed:
                    _write_jsonl_row(err_f, _error_record(item, exc))
                    report["failed"] += 1
                if _is_fatal_cuda_error(exc):
                    raise
            finally:
                for item in claimed:
                    _release_claim(item, claim_dir=claim_dir)
                _write_report(report_json, report)
    report["status"] = "complete"
    _write_report(report_json, report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
