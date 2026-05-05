#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_AUDIO_ROOT = Path("/data/EECS-Pauwels-C4DM/music4all/audios")
DEFAULT_OUTPUT_JSONL = Path("/gpfs/scratch/acw749/datasets/music4all_instruct/metadata/final_caption30sec.jsonl")
DEFAULT_MODEL_ID = "nvidia/audio-flamingo-next-hf"
DEFAULT_PROMPT = (
    "Write one concise music caption in 1-2 sentences for this 30-second audio clip. "
    "Mention genre or style, main instruments, vocals if present, tempo or energy, mood, "
    "production texture, and key or BPM if clear. Do not quote lyrics or invent the artist "
    "name or track title."
)


@dataclass(frozen=True)
class AudioItem:
    track_id: str
    audio_path: Path


def _log(message: str) -> None:
    logging.info(message)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                _log(f"Warning: skipping malformed JSONL row at {path}:{line_no}")
                continue
            if isinstance(value, dict):
                yield value


def _load_completed_ids(output_jsonl: Path) -> set[str]:
    completed: set[str] = set()
    for row in _iter_jsonl(output_jsonl):
        track_id = str(row.get("id", "") or "").strip()
        caption = str(row.get("caption", "") or "").strip()
        if track_id and caption:
            completed.add(track_id)
    return completed


def _claim_path(claim_dir: Path, track_id: str) -> Path:
    return claim_dir / f"{track_id}.claim"


def _try_claim_item(item: AudioItem, *, claim_dir: Path, stale_seconds: float | None) -> bool:
    claim_dir.mkdir(parents=True, exist_ok=True)
    path = _claim_path(claim_dir, item.track_id)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    payload = json.dumps(
        {
            "id": item.track_id,
            "audio_path": str(item.audio_path),
            "pid": os.getpid(),
            "host": os.uname().nodename,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
        ensure_ascii=True,
    )
    try:
        fd = os.open(path, flags, 0o644)
    except FileExistsError:
        if stale_seconds is not None and stale_seconds > 0:
            try:
                age = time.time() - path.stat().st_mtime
            except FileNotFoundError:
                return _try_claim_item(item, claim_dir=claim_dir, stale_seconds=stale_seconds)
            if age > stale_seconds:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    return False
                return _try_claim_item(item, claim_dir=claim_dir, stale_seconds=stale_seconds)
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(payload + "\n")
    return True


def _release_claim(item: AudioItem, *, claim_dir: Path) -> None:
    try:
        _claim_path(claim_dir, item.track_id).unlink()
    except FileNotFoundError:
        pass


def _discover_audio(audio_root: Path, extensions: Sequence[str]) -> list[AudioItem]:
    if not audio_root.exists():
        raise FileNotFoundError(f"Audio root not found: {audio_root}")
    suffixes = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    items = [
        AudioItem(track_id=path.stem, audio_path=path)
        for path in sorted(audio_root.iterdir())
        if path.is_file() and path.suffix.lower() in suffixes
    ]
    if not items:
        raise RuntimeError(f"No audio files found under {audio_root} for extensions {sorted(suffixes)}")
    return items


def _batched(items: Sequence[AudioItem], batch_size: int) -> Iterable[list[AudioItem]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def _progress(iterable: Iterable[list[AudioItem]], *, total: int, enabled: bool) -> Iterable[list[AudioItem]]:
    if not enabled:
        yield from iterable
        return
    try:
        from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Caption batches", total=total)
            for batch in iterable:
                yield batch
                progress.advance(task)
        return
    except Exception:
        pass
    try:
        from tqdm.auto import tqdm
    except Exception:
        yield from iterable
        return
    yield from tqdm(iterable, total=total, desc="Caption batches", unit="batch")


def _resolve_torch_dtype(torch: Any, requested: str) -> Any:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    if requested == "bfloat16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        _log("CUDA device does not report BF16 support; falling back to float16.")
        return torch.float16
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if requested not in dtype_map:
        raise ValueError(f"Unsupported torch dtype: {requested}")
    return dtype_map[requested]


def _load_model(model_id: str, torch_dtype: str, device_map: str, trust_remote_code: bool) -> tuple[Any, Any, Any]:
    import torch

    _patch_torch_custom_op_string_annotations(torch)
    from transformers import AutoModel, AutoProcessor

    dtype = _resolve_torch_dtype(torch, torch_dtype)

    _log(f"Loading processor: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    _log(f"Loading model: {model_id} (torch_dtype={dtype})")
    try:
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        ).eval()
    except ValueError as exc:
        if "audioflamingonext" in str(exc):
            raise RuntimeError(
                "This environment's Transformers build does not register the "
                "`audioflamingonext` architecture required by "
                f"`{model_id}`. The Hugging Face model card currently points "
                "to the pinned demo branch for exact support:\n"
                "  uv pip install --reinstall "
                "'git+https://github.com/lashahub/transformers.git@add_AudioFlamingoNext' accelerate\n"
                "Installing `git+https://github.com/huggingface/transformers` "
                "main may provide `audioflamingo3` but not this checkpoint's "
                "`audioflamingonext` model type."
            ) from exc
        raise
    return processor, model, torch


def _patch_torch_custom_op_string_annotations(torch: Any) -> None:
    """Allow newer Transformers custom-op annotations to import on older torch.

    Transformers main currently registers a grouped-mm fallback custom op during
    import. Some torch 2.4 builds cannot infer schemas from string annotations
    like "torch.Tensor", so normalize those annotations before registration.
    """
    library = getattr(torch, "library", None)
    custom_op = getattr(library, "custom_op", None)
    if custom_op is None or getattr(custom_op, "_jamendo_string_annotation_patch", False):
        return

    def normalize(fn: Any) -> Any:
        annotations = getattr(fn, "__annotations__", None)
        if not annotations:
            return fn
        replacements = {
            "torch.Tensor": torch.Tensor,
            "Tensor": torch.Tensor,
            "torch.dtype": torch.dtype,
            "torch.device": torch.device,
        }
        changed = False
        normalized: dict[str, Any] = {}
        for key, value in annotations.items():
            if isinstance(value, str) and value in replacements:
                normalized[key] = replacements[value]
                changed = True
            else:
                normalized[key] = value
        if changed:
            fn.__annotations__ = normalized
        return fn

    def custom_op_compat(name: str, fn: Any = None, /, **kwargs: Any) -> Any:
        if fn is not None:
            return custom_op(name, normalize(fn), **kwargs)
        decorator = custom_op(name, None, **kwargs)

        def wrapped(inner_fn: Any) -> Any:
            return decorator(normalize(inner_fn))

        return wrapped

    custom_op_compat._jamendo_string_annotation_patch = True  # type: ignore[attr-defined]
    library.custom_op = custom_op_compat


def _model_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return "cpu"


def _load_audio_array(audio_path: Path, *, sampling_rate: int, clip_seconds: float) -> Any:
    import numpy as np
    from transformers.audio_utils import load_audio

    audio = load_audio(str(audio_path), sampling_rate=sampling_rate)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)

    target_samples = max(1, int(round(float(clip_seconds) * int(sampling_rate))))
    if audio.shape[0] > target_samples:
        audio = audio[:target_samples]
    elif audio.shape[0] < target_samples:
        audio = np.pad(audio, (0, target_samples - audio.shape[0]))
    return audio


def _caption_batch(
    items: Sequence[AudioItem],
    *,
    processor: Any,
    model: Any,
    torch: Any,
    prompt: str,
    clip_seconds: float,
    audio_sampling_rate: int,
    max_new_tokens: int,
    repetition_penalty: float,
    temperature: float,
    top_p: float,
) -> list[str]:
    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio"},
                ],
            }
        ]
        for item in items
    ]
    text = processor.apply_chat_template(
        conversations,
        tokenize=False,
        add_generation_prompt=True,
    )
    audio = [
        _load_audio_array(item.audio_path, sampling_rate=audio_sampling_rate, clip_seconds=clip_seconds)
        for item in items
    ]
    batch = processor(
        text=text,
        audio=audio,
        audio_kwargs={
            "sampling_rate": audio_sampling_rate,
            "return_attention_mask": True,
            "padding": "max_length",
        },
        text_kwargs={"padding": True},
        common_kwargs={"return_tensors": "pt", "padding_side": "left"},
    ).to(_model_device(model))

    if "input_features" in batch:
        batch["input_features"] = batch["input_features"].to(model.dtype)

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
    }
    if temperature > 0:
        generation_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
    else:
        generation_kwargs.update({"do_sample": False})

    with torch.inference_mode():
        generated = model.generate(**batch, **generation_kwargs)

    prompt_len = batch["input_ids"].shape[1]
    completions = generated[:, prompt_len:]
    decoded = processor.batch_decode(
        completions,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [str(text).strip() for text in decoded]


def _write_jsonl_row(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    handle.flush()


def _record_for_item(item: AudioItem, caption: str, *, model_id: str, prompt: str, clip_seconds: float) -> dict[str, Any]:
    return {
        "id": item.track_id,
        "start_time": 0.0,
        "end_time": float(clip_seconds),
        "caption": caption,
        "source_audio": str(item.audio_path),
        "caption_model": model_id,
        "caption_prompt": prompt,
    }


def _error_record(item: AudioItem, exc: BaseException) -> dict[str, Any]:
    return {
        "id": item.track_id,
        "audio_path": str(item.audio_path),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _is_fatal_cuda_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    fatal_markers = (
        "CUDA error: device-side assert triggered",
        "CUDA error: no kernel image is available for execution on the device",
    )
    return any(marker in text for marker in fatal_markers)


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Caption Music4all audio clips with Audio Flamingo Next and write ingest-compatible JSONL.",
    )
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--error-jsonl", type=Path, default=None, help="Default: <output-jsonl>.errors.jsonl")
    parser.add_argument("--report-json", type=Path, default=None, help="Default: <output-jsonl>.report.json")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--extensions", nargs="+", default=[".mp3"])
    parser.add_argument("--clip-seconds", type=float, default=30.0)
    parser.add_argument("--audio-sampling-rate", type=int, default=16000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="Ignore completed rows already in output JSONL.")
    parser.add_argument("--claim-dir", type=Path, default=None, help="Default: <output-jsonl>.claims")
    parser.add_argument(
        "--claim-stale-seconds",
        type=float,
        default=86400.0,
        help="Reclaim per-track claim files older than this many seconds. Use <=0 to disable stale reclaim.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List planned work without loading the model.")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--log-caption-every", type=int, default=100, help="Log one generated caption every N written captions.")
    parser.add_argument("--torch-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="[music4all-caption] %(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.offset < 0:
        raise ValueError("--offset must be >= 0")

    error_jsonl = args.error_jsonl or args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".errors.jsonl")
    report_json = args.report_json or args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".report.json")
    claim_dir = args.claim_dir or args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".claims")
    claim_stale_seconds = None if args.claim_stale_seconds <= 0 else float(args.claim_stale_seconds)

    _log(f"Scanning audio root: {args.audio_root}")
    all_items = _discover_audio(args.audio_root, args.extensions)
    all_items = all_items[args.offset :]
    if args.max_items is not None:
        all_items = all_items[: max(0, int(args.max_items))]

    completed = set() if args.overwrite else _load_completed_ids(args.output_jsonl)
    todo = [item for item in all_items if item.track_id not in completed]

    _log(f"Audio root: {args.audio_root}")
    _log(f"Discovered audio clips in selected range: {len(all_items):,}")
    _log(f"Completed captions found: {len(completed):,}")
    _log(f"Remaining captions to generate: {len(todo):,}")
    _log(f"Output JSONL: {args.output_jsonl}")

    report: dict[str, Any] = {
        "status": "planned" if args.dry_run else "running",
        "audio_root": str(args.audio_root),
        "output_jsonl": str(args.output_jsonl),
        "error_jsonl": str(error_jsonl),
        "claim_dir": str(claim_dir),
        "model_id": args.model_id,
        "prompt": args.prompt,
        "selected_audio_count": len(all_items),
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

    processor, model, torch = _load_model(
        args.model_id,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        trust_remote_code=bool(args.trust_remote_code),
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    error_jsonl.parent.mkdir(parents=True, exist_ok=True)
    claim_dir.mkdir(parents=True, exist_ok=True)
    total_batches = (len(todo) + args.batch_size - 1) // args.batch_size
    started = time.time()

    with args.output_jsonl.open("a", encoding="utf-8") as out_f, error_jsonl.open("a", encoding="utf-8") as err_f:
        for batch_index, batch_items in enumerate(
            _progress(_batched(todo, args.batch_size), total=total_batches, enabled=not args.no_progress),
            start=1,
        ):
            current_completed = set() if args.overwrite else _load_completed_ids(args.output_jsonl)
            claim_candidates = [item for item in batch_items if item.track_id not in current_completed]
            report["skipped_already_completed"] += len(batch_items) - len(claim_candidates)

            claimed_items: list[AudioItem] = []
            for item in claim_candidates:
                if _try_claim_item(item, claim_dir=claim_dir, stale_seconds=claim_stale_seconds):
                    claimed_items.append(item)
                else:
                    report["skipped_claimed"] += 1

            if claimed_items and not args.overwrite:
                current_completed = _load_completed_ids(args.output_jsonl)
                ready_items = []
                for item in claimed_items:
                    if item.track_id in current_completed:
                        _release_claim(item, claim_dir=claim_dir)
                        report["skipped_already_completed"] += 1
                    else:
                        ready_items.append(item)
                claimed_items = ready_items

            if not claimed_items:
                if batch_index == 1 or batch_index % 10 == 0:
                    _write_report(report_json, report)
                continue

            try:
                captions = _caption_batch(
                    claimed_items,
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
            except Exception as batch_exc:
                if len(claimed_items) == 1:
                    captions = []
                    _write_jsonl_row(err_f, _error_record(claimed_items[0], batch_exc))
                    report["failed"] += 1
                    _release_claim(claimed_items[0], claim_dir=claim_dir)
                    if _is_fatal_cuda_error(batch_exc):
                        report["status"] = "failed"
                        report["fatal_error"] = str(batch_exc)
                        _write_report(report_json, report)
                        raise RuntimeError(
                            "Fatal CUDA error encountered; stopping because the CUDA context is likely invalid."
                        ) from batch_exc
                else:
                    _log(f"Batch {batch_index} failed; retrying items one at a time: {batch_exc}")
                    captions = []
                    for item in claimed_items:
                        try:
                            captions.extend(
                                _caption_batch(
                                    [item],
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
                            )
                        except Exception as item_exc:
                            _write_jsonl_row(err_f, _error_record(item, item_exc))
                            report["failed"] += 1
                            captions.append("")
                            _release_claim(item, claim_dir=claim_dir)
                            if _is_fatal_cuda_error(item_exc):
                                report["status"] = "failed"
                                report["fatal_error"] = str(item_exc)
                                _write_report(report_json, report)
                                raise RuntimeError(
                                    "Fatal CUDA error encountered; stopping because the CUDA context is likely invalid."
                                ) from item_exc

            for item, caption in zip(claimed_items, captions):
                caption = caption.strip()
                try:
                    if not caption:
                        continue
                    _write_jsonl_row(
                        out_f,
                        _record_for_item(
                            item,
                            caption,
                            model_id=args.model_id,
                            prompt=args.prompt,
                            clip_seconds=args.clip_seconds,
                        ),
                    )
                    report["written"] += 1
                    if args.log_caption_every > 0 and int(report["written"]) % int(args.log_caption_every) == 0:
                        logging.info(
                            "Caption %s/%s id=%s: %s",
                            f"{int(report['written']):,}",
                            f"{len(todo):,}",
                            item.track_id,
                            caption,
                        )
                finally:
                    _release_claim(item, claim_dir=claim_dir)
            for item in claimed_items[len(captions) :]:
                _release_claim(item, claim_dir=claim_dir)

            processed = (
                int(report["written"])
                + int(report["failed"])
                + int(report["skipped_already_completed"])
                + int(report["skipped_claimed"])
            )
            if processed > 0 and args.progress_every > 0 and processed % args.progress_every == 0:
                elapsed = max(time.time() - started, 1e-6)
                rate = processed / elapsed
                _log(
                    f"Processed {processed:,}/{len(todo):,} "
                    f"(written={report['written']:,}, failed={report['failed']:,}, "
                    f"already_done={report['skipped_already_completed']:,}, "
                    f"claimed={report['skipped_claimed']:,}, rate={rate:.2f}/s)"
                )
            if batch_index == 1 or batch_index % 10 == 0:
                _write_report(report_json, report)

    report["status"] = "complete"
    report["elapsed_sec"] = round(time.time() - started, 3)
    _write_report(report_json, report)
    _log(f"Done: written={report['written']:,}, failed={report['failed']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
