from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

from jamendo_instruct.progress import StageTracker, rich_tqdm
from jamendo_instruct.stages.embeddings import _AudioClipDataset, _audio_loading_cfg, _collate_audio_batch

if TYPE_CHECKING:
    from omegaconf import DictConfig
else:
    DictConfig = Any

CONF_DIR = str(Path(__file__).resolve().parents[3] / "conf")

_LYRIC_FIELDS = [
    "lyrics",
    "lyrics_language",
    "lyrics_segments_json",
    "lyrics_status",
    "lyrics_source",
    "lyrics_error",
]
_REUSABLE_STATUSES = {"ok", "empty"}


def _log(cfg: DictConfig, message: str) -> None:
    if bool(cfg.stage.progress.enabled):
        print(f"[lyrics] {message}", flush=True)


def _cfg_section_to_plain(obj: Any) -> Any:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(obj):
            return OmegaConf.to_container(obj, resolve=True)
    except Exception:
        pass
    if hasattr(obj, "items"):
        return {str(k): _cfg_section_to_plain(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {str(k): _cfg_section_to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_cfg_section_to_plain(v) for v in obj]
    return obj


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv_rows(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def _audio_identifier(row: Dict[str, Any]) -> str:
    for key in ("file_path", "audio_path", "audio_download_url", "source_audio", "audio_url"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return str(row.get("clip_id", "") or row.get("track_id", "") or "").strip()


def _detected_lyrics_entries(rows: List[Dict[str, Any]]) -> List[str]:
    entries: List[str] = []
    for row in rows:
        if str(row.get("lyrics_status", "") or "").strip().lower() != "ok":
            continue
        identifier = _audio_identifier(row)
        clip_id = str(row.get("clip_id", "") or "").strip()
        track_id = str(row.get("track_id", "") or "").strip()
        language = str(row.get("lyrics_language", "") or "").strip()
        parts = [identifier]
        metadata = [part for part in (clip_id, track_id, language) if part]
        if metadata:
            parts.append(f"({', '.join(metadata)})")
        entries.append(" ".join(parts).strip())
    return entries


def _write_detected_lyrics_file(path: Path, entries: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(f"{entry}\n")
    tmp_path.replace(path)


def _checkpoint_dir(output_dir: Path) -> Path:
    return output_dir / "lyrics_checkpoints"


def _checkpoint_path(checkpoint_dir: Path, checkpoint_idx: int) -> Path:
    return checkpoint_dir / f"checkpoint_{checkpoint_idx:06d}.csv"


def _load_existing_rows_from_paths(paths: List[Path]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in _read_csv_rows(path):
            clip_id = str(row.get("clip_id", "") or "").strip()
            if clip_id:
                out[clip_id] = row
    return out


def _parse_float(raw: Any) -> float | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _normalize_text(text: str, cfg: DictConfig) -> str:
    out = str(text or "")
    if bool(cfg.stage.transcription.strip_whitespace):
        out = out.strip()
    if bool(cfg.stage.transcription.collapse_whitespace):
        out = re.sub(r"\s+", " ", out)
    return out.strip()


def _normalized_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", str(text or "").lower())


def _normalized_phrase(text: str) -> str:
    return " ".join(_normalized_tokens(text))


def _caption_texts_for_filter(row: Dict[str, Any]) -> List[str]:
    return [
        str(row.get("caption", "") or ""),
        str(row.get("primary_caption", "") or ""),
        str(row.get("track_primary_caption", "") or ""),
    ]


def _normalized_vocals(row: Dict[str, Any]) -> str:
    return str(row.get("vocals", "") or "").strip().lower()


def _allowed_vocals(cfg: DictConfig) -> set[str]:
    raw = getattr(cfg.stage.filters, "allowed_vocals", ["vocal"])
    if isinstance(raw, (list, tuple)) or hasattr(raw, "__iter__") and not isinstance(raw, (str, bytes)):
        return {str(x).strip().lower() for x in list(raw) if str(x).strip()}
    value = str(raw or "").strip().lower()
    return {value} if value else {"vocal"}


def _should_skip_non_vocal(row: Dict[str, Any], cfg: DictConfig) -> bool:
    if not bool(getattr(cfg.stage.filters, "only_vocal_tracks", True)):
        return False
    vocals = _normalized_vocals(row)
    allowed = _allowed_vocals(cfg)
    if vocals:
        return vocals not in allowed
    for text in _caption_texts_for_filter(row):
        if re.search(r"\binstrumental\b", text.lower()):
            return True
    return True


def _suspicious_exact_phrases(cfg: DictConfig) -> set[str]:
    raw = getattr(cfg.stage.filters, "suspicious_exact_phrases", [])
    if isinstance(raw, (list, tuple)) or hasattr(raw, "__iter__") and not isinstance(raw, (str, bytes)):
        return {_normalized_phrase(x) for x in list(raw) if _normalized_phrase(x)}
    value = _normalized_phrase(raw)
    return {value} if value else set()


def _suspicious_suffixes(cfg: DictConfig) -> set[str]:
    raw = getattr(cfg.stage.filters, "suspicious_suffixes", [])
    if isinstance(raw, (list, tuple)) or hasattr(raw, "__iter__") and not isinstance(raw, (str, bytes)):
        return {str(x).strip().lower() for x in list(raw) if str(x).strip()}
    value = str(raw or "").strip().lower()
    return {value} if value else set()


def _hallucination_reason(text: str, cfg: DictConfig) -> str:
    phrase = _normalized_phrase(text)
    if not phrase:
        return ""
    if phrase in _suspicious_exact_phrases(cfg):
        return f"exact_phrase:{phrase}"
    words = phrase.split()
    max_words = max(1, int(getattr(cfg.stage.filters, "max_words_for_suffix_rule", 2) or 2))
    suffixes = _suspicious_suffixes(cfg)
    if words and len(words) <= max_words and words[-1] in suffixes:
        return f"short_suffix:{phrase}"
    return ""


def _resolve_torch_device(cfg: DictConfig) -> Any:
    import torch

    requested = str(cfg.stage.runtime.device)
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("stage.runtime.device requests CUDA, but torch.cuda.is_available() is false.")
    return SimpleNamespace(torch=torch, device=requested)


def _resolve_torch_dtype(cfg: DictConfig, torch_module: Any) -> Any:
    value = str(cfg.stage.runtime.torch_dtype)
    if value == "auto":
        return torch_module.float16 if torch_module.cuda.is_available() else torch_module.float32
    mapping = {
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported stage.runtime.torch_dtype: {value}")
    return mapping[value]


def _primary_temperature(cfg: DictConfig) -> float:
    raw = cfg.stage.transcription.temperature
    if isinstance(raw, (list, tuple)) or hasattr(raw, "__iter__") and not isinstance(raw, (str, bytes)):
        values = [float(x) for x in list(raw)]
        return values[0] if values else 0.0
    return float(raw)


def _build_asr_pipeline(cfg: DictConfig) -> Any:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    runtime = _resolve_torch_device(cfg)
    torch = runtime.torch
    device = runtime.device
    token_env = str(getattr(cfg.stage.auth, "hf_token_env", "HF_TOKEN"))
    token = os.environ.get(token_env, "").strip() or None
    if token is None:
        _log(cfg, f"No Hugging Face token found in ${token_env}; gated model downloads may fail.")

    model_id = str(cfg.stage.models.model_id)
    torch_dtype = _resolve_torch_dtype(cfg, torch)
    _log(cfg, f"Loading ASR model {model_id} on {device}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        token=token,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.generation_config.max_new_tokens = int(cfg.stage.transcription.max_new_tokens)
    model.generation_config.max_length = None
    model.generation_config.num_beams = int(cfg.stage.transcription.num_beams)
    # Avoid Whisper's fallback heuristics here; they are causing shape errors in this environment.
    model.generation_config.temperature = _primary_temperature(cfg)
    model.generation_config.condition_on_prev_tokens = False
    model.generation_config.compression_ratio_threshold = None
    model.generation_config.logprob_threshold = None
    model.generation_config.no_speech_threshold = None
    processor = AutoProcessor.from_pretrained(model_id, token=token)
    model = model.to(device)
    model.eval()
    return SimpleNamespace(
        model=model,
        processor=processor,
        torch=torch,
        device=device,
        torch_dtype=torch_dtype,
        model_id=model_id,
        sampling_rate=int(getattr(processor.feature_extractor, "sampling_rate", int(cfg.stage.audio.sample_rate))),
    )


def _generate_kwargs(cfg: DictConfig) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "max_new_tokens": int(cfg.stage.transcription.max_new_tokens),
        "num_beams": int(cfg.stage.transcription.num_beams),
        "task": str(cfg.stage.transcription.task),
        "temperature": _primary_temperature(cfg),
    }
    language = cfg.stage.transcription.language
    if language not in (None, ""):
        kwargs["language"] = str(language)
    return kwargs


def _default_lyric_fields(row: Dict[str, Any]) -> None:
    row["lyrics"] = str(row.get("lyrics", "") or "")
    row["lyrics_language"] = str(row.get("lyrics_language", "") or "")
    row["lyrics_segments_json"] = str(row.get("lyrics_segments_json", "[]") or "[]")
    row["lyrics_status"] = str(row.get("lyrics_status", "") or "")
    row["lyrics_source"] = str(row.get("lyrics_source", "") or "")
    row["lyrics_error"] = str(row.get("lyrics_error", "") or "")


def _can_reuse_previous(row: Dict[str, Any], cfg: DictConfig, *, overwrite_existing: bool) -> bool:
    if overwrite_existing:
        return False
    status = str(row.get("lyrics_status", "") or "").strip().lower()
    if status not in _REUSABLE_STATUSES:
        return False
    if _should_skip_non_vocal(row, cfg):
        return not str(row.get("lyrics", "") or "").strip()
    return not bool(_hallucination_reason(str(row.get("lyrics", "") or ""), cfg))


def _segments_json(result: Dict[str, Any]) -> str:
    chunks = result.get("chunks", [])
    if not isinstance(chunks, list):
        return "[]"
    return json.dumps(chunks, ensure_ascii=True)


def _apply_failed(row: Dict[str, Any], error: str) -> None:
    row["lyrics"] = ""
    row["lyrics_language"] = ""
    row["lyrics_segments_json"] = "[]"
    row["lyrics_status"] = "failed"
    row["lyrics_source"] = ""
    row["lyrics_error"] = str(error or "").strip()


def _apply_non_vocal_skip(row: Dict[str, Any]) -> None:
    row["lyrics"] = ""
    row["lyrics_language"] = ""
    row["lyrics_segments_json"] = "[]"
    row["lyrics_status"] = "empty"
    row["lyrics_source"] = "metadata_vocals_filter"
    row["lyrics_error"] = "non_vocal_track"


def _apply_filtered_hallucination(row: Dict[str, Any], reason: str, *, model_id: str) -> None:
    row["lyrics"] = ""
    row["lyrics_language"] = ""
    row["lyrics_segments_json"] = "[]"
    row["lyrics_status"] = "empty"
    row["lyrics_source"] = model_id
    row["lyrics_error"] = f"filtered_hallucination:{reason}"


def _apply_result(row: Dict[str, Any], result: Dict[str, Any], cfg: DictConfig, *, model_id: str) -> str:
    text = _normalize_text(str(result.get("text", "") or ""), cfg)
    reason = _hallucination_reason(text, cfg)
    if reason:
        _apply_filtered_hallucination(row, reason, model_id=model_id)
        return "filtered_hallucination"
    row["lyrics"] = text
    row["lyrics_language"] = str(result.get("language", "") or "").strip()
    row["lyrics_segments_json"] = _segments_json(result)
    row["lyrics_status"] = "ok" if text else "empty"
    row["lyrics_source"] = model_id
    row["lyrics_error"] = ""
    return str(row["lyrics_status"])


def _error_context(row: Dict[str, Any]) -> str:
    return (
        f"clip_id={str(row.get('clip_id', '') or '')} "
        f"track_id={str(row.get('track_id', '') or '')} "
        f"file_path={str(row.get('file_path', '') or '')} "
        f"start={str(row.get('start_time', '') or '')} "
        f"end={str(row.get('end_time', '') or '')}"
    ).strip()


def _verbose_error_enabled(cfg: DictConfig) -> bool:
    return bool(getattr(cfg.stage.behavior, "verbose_errors", False))


def _log_verbose_error(
    cfg: DictConfig,
    state: Dict[str, int],
    *,
    row: Dict[str, Any] | None,
    message: str,
    exc: Exception | None = None,
) -> None:
    if not _verbose_error_enabled(cfg):
        return
    limit = max(1, int(getattr(cfg.stage.behavior, "max_logged_errors", 25) or 25))
    if int(state.get("logged_errors", 0)) >= limit:
        if not bool(state.get("limit_notice_emitted", False)):
            _log(cfg, f"Verbose error log limit reached ({limit}); suppressing additional error prints.")
            state["limit_notice_emitted"] = 1
        return
    context = _error_context(row) if row is not None else ""
    prefix = f"{message} :: {context}" if context else message
    _log(cfg, prefix)
    if exc is not None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()
        if tb:
            print(tb, flush=True)
    state["logged_errors"] = int(state.get("logged_errors", 0)) + 1


def _run_pipe_on_inputs(ctx: Any, audio_inputs: List[Dict[str, Any]], cfg: DictConfig) -> List[Dict[str, Any]]:
    if not audio_inputs:
        return []
    arrays = [item["raw"] for item in audio_inputs]
    sample_rates = {int(item["sampling_rate"]) for item in audio_inputs}
    if len(sample_rates) != 1:
        raise ValueError(f"Mixed sampling rates in ASR batch: {sorted(sample_rates)}")

    processor = ctx.processor
    model = ctx.model
    torch = ctx.torch
    model_device = ctx.device
    inputs = processor(
        arrays,
        sampling_rate=ctx.sampling_rate,
        return_tensors="pt",
        truncation=False,
        padding="longest",
        return_attention_mask=True,
    )
    prepared_inputs: Dict[str, Any] = {}
    for key, value in inputs.items():
        tensor = value.to(model_device)
        if hasattr(tensor, "dtype") and str(tensor.dtype).startswith("torch.float"):
            tensor = tensor.to(dtype=ctx.torch_dtype)
        prepared_inputs[key] = tensor

    with torch.no_grad():
        pred_ids = model.generate(**prepared_inputs, **_generate_kwargs(cfg))
    texts = processor.batch_decode(pred_ids, skip_special_tokens=True, decode_with_timestamps=False)
    language = str(cfg.stage.transcription.language or "").strip()
    return [{"text": text, "language": language, "chunks": []} for text in texts]


def _transcribe_pending_rows(
    rows: List[Dict[str, Any]],
    cfg: DictConfig,
    *,
    checkpoint_callback: Any | None = None,
) -> Dict[str, int]:
    counts = {
        "rows_attempted": 0,
        "rows_reused": 0,
        "rows_ok": 0,
        "rows_empty": 0,
        "rows_failed": 0,
        "rows_skipped_non_vocal": 0,
        "rows_filtered_hallucination": 0,
    }
    pending_indices: List[int] = []
    overwrite_existing = bool(cfg.stage.behavior.overwrite_existing)
    for idx, row in enumerate(rows):
        if _should_skip_non_vocal(row, cfg):
            if _can_reuse_previous(row, cfg, overwrite_existing=overwrite_existing):
                counts["rows_reused"] += 1
            else:
                _apply_non_vocal_skip(row)
                counts["rows_skipped_non_vocal"] += 1
            continue
        if _can_reuse_previous(row, cfg, overwrite_existing=overwrite_existing):
            counts["rows_reused"] += 1
            continue
        pending_indices.append(idx)

    if not pending_indices:
        counts["rows_ok"] = sum(str(row.get("lyrics_status", "") or "") == "ok" for row in rows)
        counts["rows_empty"] = sum(str(row.get("lyrics_status", "") or "") == "empty" for row in rows)
        counts["rows_failed"] = sum(str(row.get("lyrics_status", "") or "") == "failed" for row in rows)
        return counts

    runtime = _resolve_torch_device(cfg)
    torch = runtime.torch
    data_loader_kwargs: Dict[str, Any] = {
        "batch_size": int(cfg.stage.audio.batch_size),
        "shuffle": False,
        "num_workers": max(0, int(getattr(cfg.stage.audio, "num_workers", 0) or 0)),
        "pin_memory": str(runtime.device).startswith("cuda"),
        "collate_fn": _collate_audio_batch,
    }
    if data_loader_kwargs["num_workers"] > 0:
        prefetch = getattr(cfg.stage.audio, "prefetch_factor", None)
        if prefetch is not None:
            data_loader_kwargs["prefetch_factor"] = int(prefetch)
        data_loader_kwargs["persistent_workers"] = bool(getattr(cfg.stage.audio, "persistent_workers", False))

    pending_rows = [rows[idx] for idx in pending_indices]
    loader = torch.utils.data.DataLoader(_AudioClipDataset(pending_rows, _audio_loading_cfg(cfg)), **data_loader_kwargs)
    ctx = _build_asr_pipeline(cfg)
    counts["rows_attempted"] = len(pending_rows)
    error_log_state = {"logged_errors": 0, "limit_notice_emitted": 0}
    total_batches = int(math.ceil(len(pending_rows) / max(1, int(cfg.stage.audio.batch_size)))) if pending_rows else 0
    checkpoint_every = max(0, int(getattr(cfg.stage.behavior, "checkpoint_every_n_batches", 0) or 0))
    checkpoint_batch_idx = 0
    checkpoint_buffer: List[Dict[str, Any]] = []

    def flush_checkpoint(force: bool = False) -> None:
        nonlocal checkpoint_batch_idx, checkpoint_buffer
        if checkpoint_callback is None or not checkpoint_buffer:
            return
        if not force and checkpoint_every <= 0:
            return
        checkpoint_batch_idx += 1
        checkpoint_callback(checkpoint_batch_idx, checkpoint_buffer)
        checkpoint_buffer = []

    with rich_tqdm(cfg, total=total_batches, desc="Transcribe lyrics", unit="batch") as progress:
        batch_idx = 0
        for batch in loader:
            batch_idx += 1
            local_indices = [int(x) for x in list(batch.get("indices", []))]
            audio_np = batch.get("audio")
            for error_record in list(batch.get("errors", [])):
                local_idx = int(error_record["idx"])
                row = pending_rows[local_idx]
                _apply_failed(row, str(error_record.get("error", "") or "audio_decode_failed"))
                checkpoint_buffer.append(dict(row))
                _log_verbose_error(
                    cfg,
                    error_log_state,
                    row=row,
                    message="Audio decode failure",
                )
                counts["rows_failed"] += 1

            if local_indices:
                audio_inputs = [{"raw": audio_np[pos], "sampling_rate": ctx.sampling_rate} for pos in range(len(local_indices))]
                try:
                    results = _run_pipe_on_inputs(ctx, audio_inputs, cfg)
                    if len(results) != len(local_indices):
                        raise RuntimeError(f"ASR batch output size mismatch: expected {len(local_indices)}, got {len(results)}")
                    for local_idx, result in zip(local_indices, results):
                        row = pending_rows[local_idx]
                        apply_kind = _apply_result(row, result, cfg, model_id=ctx.model_id)
                        checkpoint_buffer.append(dict(row))
                        if apply_kind == "filtered_hallucination":
                            counts["rows_filtered_hallucination"] += 1
                            counts["rows_empty"] += 1
                        else:
                            counts["rows_ok" if row["lyrics_status"] == "ok" else "rows_empty"] += 1
                except Exception as exc:
                    _log_verbose_error(
                        cfg,
                        error_log_state,
                        row=pending_rows[local_indices[0]] if local_indices else None,
                        message=f"Batch ASR failure for batch_size={len(local_indices)}",
                        exc=exc,
                    )
                    if len(local_indices) > 1:
                        for local_idx, audio_input in zip(local_indices, audio_inputs):
                            row = pending_rows[local_idx]
                            try:
                                single_result = _run_pipe_on_inputs(ctx, [audio_input], cfg)[0]
                                apply_kind = _apply_result(row, single_result, cfg, model_id=ctx.model_id)
                                checkpoint_buffer.append(dict(row))
                                if apply_kind == "filtered_hallucination":
                                    counts["rows_filtered_hallucination"] += 1
                                    counts["rows_empty"] += 1
                                else:
                                    counts["rows_ok" if row["lyrics_status"] == "ok" else "rows_empty"] += 1
                            except Exception as single_exc:
                                _apply_failed(row, f"{exc.__class__.__name__}; {single_exc}")
                                checkpoint_buffer.append(dict(row))
                                _log_verbose_error(
                                    cfg,
                                    error_log_state,
                                    row=row,
                                    message="Single-row ASR retry failure",
                                    exc=single_exc,
                                )
                                counts["rows_failed"] += 1
                    else:
                        row = pending_rows[local_indices[0]]
                        _apply_failed(row, str(exc))
                        checkpoint_buffer.append(dict(row))
                        _log_verbose_error(
                            cfg,
                            error_log_state,
                            row=row,
                            message="Single-row ASR failure",
                            exc=exc,
                        )
                        counts["rows_failed"] += 1

            if checkpoint_every > 0 and batch_idx % checkpoint_every == 0:
                flush_checkpoint(force=True)
            progress.update(1)

    flush_checkpoint(force=True)
    return counts


def run_lyrics(cfg: DictConfig) -> Dict[str, object]:
    input_csv = Path(str(cfg.stage.io.input_manifest_csv))
    out_dir = Path(str(cfg.stage.io.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    output_csv = out_dir / str(cfg.stage.io.output_manifest_csv)
    detected_lyrics_path = out_dir / str(getattr(cfg.stage.io, "output_detected_lyrics_file", "detected_lyrics_files.txt"))
    report_path = out_dir / str(cfg.stage.io.report_file)
    checkpoint_dir = _checkpoint_dir(out_dir)
    tracker = StageTracker(
        cfg,
        "lyrics",
        title="Attach Clip Lyrics",
        subtitle=f"input={input_csv}",
        total_steps=4,
    )

    if not input_csv.exists():
        raise FileNotFoundError(f"Input manifest CSV not found: {input_csv}")

    tracker.step("Read input manifest", detail=str(input_csv))
    input_rows = _read_csv_rows(input_csv)
    base_fieldnames = list(input_rows[0].keys()) if input_rows else []
    fieldnames = base_fieldnames + [field for field in _LYRIC_FIELDS if field not in base_fieldnames]

    existing_by_clip: Dict[str, Dict[str, str]] = {}
    checkpoint_paths: List[Path] = []
    if checkpoint_dir.exists():
        checkpoint_paths = sorted(checkpoint_dir.glob("checkpoint_*.csv"))

    if bool(cfg.stage.behavior.reuse_output_manifest) and not bool(cfg.stage.behavior.overwrite_existing):
        if output_csv.exists():
            tracker.step("Load existing lyrics manifest", detail=str(output_csv))
            existing_by_clip = _load_existing_rows_from_paths([output_csv])
        elif checkpoint_paths:
            tracker.step("Load lyric checkpoints", detail=f"{len(checkpoint_paths):,} files")
            existing_by_clip = _load_existing_rows_from_paths(checkpoint_paths)
        else:
            tracker.step("Initialize lyrics rows", detail="no reusable manifest found")
    else:
        tracker.step("Initialize lyrics rows", detail="reuse disabled or overwrite requested")

    max_rows = cfg.stage.behavior.max_rows
    if max_rows is not None:
        input_rows = input_rows[: max(0, int(max_rows))]

    tracker.step("Prepare transcription rows", detail=f"{len(input_rows):,} clips")
    rows: List[Dict[str, Any]] = []
    reused = 0
    for row in input_rows:
        out_row: Dict[str, Any] = dict(row)
        clip_id = str(out_row.get("clip_id", "") or "").strip()
        previous = existing_by_clip.get(clip_id)
        if previous is not None:
            for key in _LYRIC_FIELDS:
                out_row[key] = str(previous.get(key, "") or "")
        _default_lyric_fields(out_row)
        if _can_reuse_previous(out_row, cfg, overwrite_existing=bool(cfg.stage.behavior.overwrite_existing)):
            reused += 1
        rows.append(out_row)

    def _write_checkpoint(batch_idx: int, checkpoint_rows: List[Dict[str, Any]]) -> None:
        if not checkpoint_rows:
            return
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = _checkpoint_path(checkpoint_dir, batch_idx)
        _write_csv_rows(checkpoint_path, fieldnames, checkpoint_rows)
        _log(cfg, f"Checkpoint write - {checkpoint_path.name} ({len(checkpoint_rows):,} rows)")

    counts = _transcribe_pending_rows(rows, cfg, checkpoint_callback=_write_checkpoint)
    counts["rows_reused"] = reused
    counts["input_rows"] = len(input_rows)
    counts["rows_ok"] = sum(str(row.get("lyrics_status", "") or "") == "ok" for row in rows)
    counts["rows_empty"] = sum(str(row.get("lyrics_status", "") or "") == "empty" for row in rows)
    counts["rows_failed"] = sum(str(row.get("lyrics_status", "") or "") == "failed" for row in rows)

    tracker.step("Write lyrics manifest", detail=output_csv.name)
    _write_csv_rows(output_csv, fieldnames, rows)
    detected_lyrics_entries = _detected_lyrics_entries(rows)
    _write_detected_lyrics_file(detected_lyrics_path, detected_lyrics_entries)
    if bool(getattr(cfg.stage.behavior, "cleanup_checkpoints_after_success", True)) and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    report = {
        "stage": "lyrics",
        "input": {
            "input_manifest_csv": str(input_csv),
        },
        "counts": counts,
        "config": {
            "models": _cfg_section_to_plain(cfg.stage.models),
            "audio": _cfg_section_to_plain(cfg.stage.audio),
            "transcription": _cfg_section_to_plain(cfg.stage.transcription),
            "filters": _cfg_section_to_plain(cfg.stage.filters),
            "behavior": _cfg_section_to_plain(cfg.stage.behavior),
        },
        "outputs": {
            "output_manifest_csv": str(output_csv),
            "detected_lyrics_file": str(detected_lyrics_path),
            "report": str(report_path),
            "checkpoint_dir": str(checkpoint_dir),
        },
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)

    tracker.finish(
        f"ok={counts['rows_ok']:,}, empty={counts['rows_empty']:,}, failed={counts['rows_failed']:,}"
    )
    _log(
        cfg,
        f"Lyrics stage complete. ok={counts['rows_ok']:,}, empty={counts['rows_empty']:,}, failed={counts['rows_failed']:,}",
    )
    preview_limit = max(0, int(getattr(cfg.stage.behavior, "detected_lyrics_preview_limit", 25) or 0))
    _log(cfg, f"Detected lyrics file list: {detected_lyrics_path} ({len(detected_lyrics_entries):,} files)")
    for entry in detected_lyrics_entries[:preview_limit]:
        _log(cfg, f"Detected lyrics: {entry}")
    remaining = len(detected_lyrics_entries) - preview_limit
    if remaining > 0:
        _log(cfg, f"... {remaining:,} more files with detected lyrics listed in {detected_lyrics_path}")
    return report


def _main_impl(cfg: DictConfig) -> None:
    report = run_lyrics(cfg)
    print(json.dumps({"status": "ok", "stage": "lyrics", "outputs": report["outputs"]}, indent=2))


def main() -> None:
    import hydra

    @hydra.main(version_base=None, config_path=CONF_DIR, config_name="config")
    def _wrapped(cfg: DictConfig) -> None:
        _main_impl(cfg)

    _wrapped()


if __name__ == "__main__":
    main()
