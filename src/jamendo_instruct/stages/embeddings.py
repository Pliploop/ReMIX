from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import warnings
import gc
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

from jamendo_instruct.embedding_paths import clip_file_stem
from jamendo_instruct.progress import StageTracker, rich_tqdm_iter

if TYPE_CHECKING:
    import numpy as np
    from omegaconf import DictConfig
else:
    DictConfig = Any

CONF_DIR = str(Path(__file__).resolve().parents[3] / "conf")


def _log(cfg: DictConfig, message: str) -> None:
    if bool(cfg.stage.progress.enabled):
        print(f"[embeddings] {message}", flush=True)


def _read_manifest_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _limit_rows(rows: List[Dict[str, str]], limit: Optional[int]) -> List[Dict[str, str]]:
    if limit is None:
        return rows
    return rows[: max(0, int(limit))]


def _parse_float(raw: str) -> Optional[float]:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _pick_text_input(row: Dict[str, str], cfg: DictConfig) -> str:
    field = str(cfg.stage.text.input_field)
    candidates = [field]
    if field != "normalized_caption":
        candidates.append("normalized_caption")
    if "caption" not in candidates:
        candidates.append("caption")
    if "primary_caption" not in candidates:
        candidates.append("primary_caption")

    for key in candidates:
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _build_lookup_rows(rows: List[Dict[str, str]], text_inputs: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row, text in zip(rows, text_inputs):
        out.append(
            {
                "row_idx": len(out),
                "clip_id": str(row.get("clip_id", "") or ""),
                "track_id": str(row.get("track_id", "") or ""),
                "split": str(row.get("split", "") or ""),
                "file_path": str(row.get("file_path", "") or ""),
                "start_time": str(row.get("start_time", "") or ""),
                "end_time": str(row.get("end_time", "") or ""),
                "caption": str(row.get("caption", "") or ""),
                "text_input": text,
                "audio_embedding_path": "",
                "text_embedding_path": "",
                "audio_embedding_status": "",
                "text_embedding_status": "",
                "audio_error": "",
                "text_error": "",
            }
        )
    return out


def _write_lookup_manifest(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "row_idx",
        "clip_id",
        "track_id",
        "split",
        "file_path",
        "start_time",
        "end_time",
        "caption",
        "text_input",
        "audio_embedding_path",
        "text_embedding_path",
        "audio_embedding_status",
        "text_embedding_status",
        "audio_error",
        "text_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _refresh_lookup_audio_statuses(
    lookup_rows: List[Dict[str, Any]],
    *,
    overwrite_existing: bool,
) -> None:
    for row in lookup_rows:
        audio_path = str(row.get("audio_embedding_path", "") or "")
        if not overwrite_existing and audio_path and _path_exists(audio_path):
            row["audio_embedding_status"] = "ok"
            row["audio_error"] = ""
        elif str(row.get("audio_embedding_status", "") or "").strip() != "failed":
            row["audio_embedding_status"] = "pending"
            row["audio_error"] = ""


def _normalize_embedding_status(raw: Any) -> str:
    status = str(raw or "").strip().lower()
    if status in {"ok", "pending", "failed"}:
        return status
    return ""


def _refresh_lookup_statuses_from_files(
    lookup_rows: List[Dict[str, Any]],
    *,
    overwrite_existing: bool,
) -> None:
    for row in lookup_rows:
        audio_path = str(row.get("audio_embedding_path", "") or "")
        text_path = str(row.get("text_embedding_path", "") or "")
        if not overwrite_existing and audio_path and _path_exists(audio_path):
            row["audio_embedding_status"] = "ok"
            row["audio_error"] = ""
        elif _normalize_embedding_status(row.get("audio_embedding_status")) != "failed":
            row["audio_embedding_status"] = "pending"
            row["audio_error"] = ""

        if not overwrite_existing and text_path and _path_exists(text_path):
            row["text_embedding_status"] = "ok"
            row["text_error"] = ""
        elif _normalize_embedding_status(row.get("text_embedding_status")) != "failed":
            row["text_embedding_status"] = "pending"
            row["text_error"] = ""


def _merge_existing_lookup_rows(
    lookup_rows: List[Dict[str, Any]],
    existing_rows: List[Dict[str, str]],
) -> bool:
    existing_by_clip_id: Dict[str, Dict[str, str]] = {}
    for row in existing_rows:
        clip_id = str(row.get("clip_id", "") or "").strip()
        if clip_id and clip_id not in existing_by_clip_id:
            existing_by_clip_id[clip_id] = row

    upgraded_schema = False
    for row in lookup_rows:
        clip_id = str(row.get("clip_id", "") or "").strip()
        if not clip_id:
            continue
        previous = existing_by_clip_id.get(clip_id)
        if previous is None:
            continue
        for key in ("audio_embedding_path", "text_embedding_path"):
            previous_value = str(previous.get(key, "") or "").strip()
            if previous_value:
                row[key] = previous_value
        row["audio_embedding_status"] = _normalize_embedding_status(previous.get("audio_embedding_status"))
        row["text_embedding_status"] = _normalize_embedding_status(previous.get("text_embedding_status"))
        row["audio_error"] = str(previous.get("audio_error", "") or "")
        row["text_error"] = str(previous.get("text_error", "") or "")
        if "text_embedding_status" not in previous or "text_error" not in previous:
            upgraded_schema = True
    return upgraded_schema


def _count_status(lookup_rows: List[Dict[str, Any]], key: str, value: str) -> int:
    return sum(_normalize_embedding_status(row.get(key)) == value for row in lookup_rows)


def _needs_status_verification(lookup_rows: List[Dict[str, Any]]) -> bool:
    for row in lookup_rows:
        if not _normalize_embedding_status(row.get("audio_embedding_status")):
            return True
        if not _normalize_embedding_status(row.get("text_embedding_status")):
            return True
    return False


def _has_pending_status(lookup_rows: List[Dict[str, Any]]) -> bool:
    for row in lookup_rows:
        if _normalize_embedding_status(row.get("audio_embedding_status")) == "pending":
            return True
        if _normalize_embedding_status(row.get("text_embedding_status")) == "pending":
            return True
    return False


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True, sort_keys=True)
    tmp_path.replace(path)


def _np():
    import numpy as np

    return np


def _save_npy_array(path: Path, array: Any) -> None:
    np = _np()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    np.save(tmp_path, array.astype(np.float32, copy=False))
    tmp_npy = tmp_path if str(tmp_path).endswith(".npy") else Path(str(tmp_path) + ".npy")
    tmp_npy.replace(path)


def _remove_file_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def _attach_embedding_paths(lookup_rows: List[Dict[str, Any]], audio_dir: Path, text_dir: Path) -> None:
    for row in lookup_rows:
        clip_id = str(row.get("clip_id", "") or "").strip() or str(row.get("track_id", "") or "").strip()
        stem = clip_file_stem(clip_id)
        row["audio_embedding_path"] = str(audio_dir / f"{stem}.npy")
        row["text_embedding_path"] = str(text_dir / f"{stem}.npy")


def _batched(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
    size = max(1, int(batch_size))
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _normalize_np(array: Any) -> Any:
    np = _np()
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return array / norms


def _resolve_torch_device(cfg: DictConfig):
    import torch

    requested = str(cfg.stage.runtime.device)
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("stage.runtime.device requests CUDA, but torch.cuda.is_available() is false.")
    device = torch.device(requested)
    if device.type == "cuda" and bool(getattr(cfg.stage.runtime, "cudnn_benchmark", False)):
        torch.backends.cudnn.benchmark = True
    return SimpleNamespace(torch=torch, device=device)


def _build_audio_embedder(cfg: DictConfig):
    from muq import MuQMuLan

    runtime = _resolve_torch_device(cfg)
    torch = runtime.torch
    device = runtime.device

    _log(cfg, f"Loading audio model {cfg.stage.models.audio_model_id} on {device}")
    model = MuQMuLan.from_pretrained(str(cfg.stage.models.audio_model_id))
    model = model.to(device)
    model.eval()

    return SimpleNamespace(model=model, device=device, torch=torch)


class _CorruptAudioError(RuntimeError):
    pass


def _load_audio_clip(row: Dict[str, str], cfg: DictConfig) -> Any:
    audio_cfg = {
        "sample_rate": int(cfg.stage.audio.sample_rate),
        "mono": bool(cfg.stage.audio.mono),
        "min_duration_sec": float(cfg.stage.audio.min_duration_sec),
        "max_duration_sec": (
            None if cfg.stage.audio.max_duration_sec is None else float(cfg.stage.audio.max_duration_sec)
        ),
    }
    return _load_audio_clip_with_params(row, audio_cfg)


def _load_audio_clip_with_params(row: Dict[str, str], audio_cfg: Dict[str, Any]) -> Any:
    np = _np()
    import soundfile as sf
    import torchaudio

    file_path = str(row.get("file_path", "") or "").strip()
    if not file_path:
        raise ValueError("Missing file_path")

    sample_rate = int(audio_cfg["sample_rate"])
    start_time = _parse_float(row.get("start_time", ""))
    end_time = _parse_float(row.get("end_time", ""))
    duration = None
    offset = 0.0

    if start_time is not None and start_time >= 0:
        offset = start_time
    if start_time is not None and end_time is not None and end_time > start_time:
        duration = end_time - start_time

    max_duration = audio_cfg.get("max_duration_sec")
    if max_duration is not None:
        max_duration = float(max_duration)
        duration = min(duration, max_duration) if duration is not None else max_duration

    wav = None
    src_sr = None
    # Fast path: soundfile partial read
    try:
        with sf.SoundFile(file_path) as f:
            src_sr = int(f.samplerate)
            start_frame = max(0, int(round(offset * src_sr))) if offset else 0
            frames = -1
            if duration is not None:
                frames = max(1, int(round(duration * src_sr)))
            f.seek(start_frame)
            wav = f.read(frames=frames, dtype="float32", always_2d=not bool(audio_cfg["mono"]))
            if wav.ndim == 2:
                wav = wav.mean(axis=1)
    except Exception:
        wav = None

    # Fallback: torchaudio partial load
    if wav is None:
        info = torchaudio.info(file_path)
        src_sr = int(info.sample_rate)
        frame_offset = max(0, int(round(offset * src_sr))) if offset else 0
        num_frames = -1
        if duration is not None:
            num_frames = max(1, int(round(duration * src_sr)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio, src_sr = torchaudio.load(
                file_path,
                frame_offset=frame_offset,
                num_frames=num_frames,
            )
        audio = audio.mean(dim=0) if bool(audio_cfg["mono"]) and audio.ndim > 1 else audio.squeeze(0)
        wav = audio.detach().cpu().numpy().astype(np.float32, copy=False)

    if wav is None:
        raise _CorruptAudioError(f"Unable to decode audio: {file_path}")
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 0:
        wav = wav.reshape(1)
    if wav.size == 0 or wav.shape[0] == 0:
        raise _CorruptAudioError(f"Decoded empty audio: {file_path}")

    if src_sr is None:
        src_sr = sample_rate

    if int(src_sr) != sample_rate:
        import torch

        wav_tensor = torch.from_numpy(wav).float()
        wav_tensor = torchaudio.functional.resample(wav_tensor, int(src_sr), sample_rate)
        wav = wav_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        if wav.size == 0 or wav.shape[0] == 0:
            raise _CorruptAudioError(f"Resampling produced empty audio: {file_path}")

    min_duration = float(audio_cfg["min_duration_sec"])
    min_samples = max(1, int(round(min_duration * sample_rate)))
    if wav.shape[0] < min_samples:
        wav = np.pad(wav, (0, min_samples - wav.shape[0]))

    return wav.astype(np.float32, copy=False)


def _audio_loading_cfg(cfg: DictConfig) -> Dict[str, Any]:
    return {
        "sample_rate": int(cfg.stage.audio.sample_rate),
        "mono": bool(cfg.stage.audio.mono),
        "min_duration_sec": float(cfg.stage.audio.min_duration_sec),
        "max_duration_sec": (
            None if cfg.stage.audio.max_duration_sec is None else float(cfg.stage.audio.max_duration_sec)
        ),
    }


class _AudioClipDataset:
    def __init__(self, rows: List[Dict[str, str]], audio_cfg: Dict[str, Any]) -> None:
        self.rows = rows
        self.audio_cfg = audio_cfg

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            wav = _load_audio_clip_with_params(self.rows[idx], self.audio_cfg)
            return {"idx": idx, "wav": wav, "error": None}
        except Exception as exc:
            return {"idx": idx, "wav": None, "error": str(exc) or exc.__class__.__name__}


def _collate_audio_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    np = _np()
    errors = [{"idx": int(item["idx"]), "error": str(item["error"])} for item in batch if item["error"]]
    valid_items = [item for item in batch if item["wav"] is not None]
    indices = [int(item["idx"]) for item in valid_items]
    wavs = [item["wav"] for item in valid_items]
    if not wavs:
        return {"indices": [], "audio": np.zeros((0, 0), dtype=np.float32), "errors": errors}
    max_len = max(w.shape[0] for w in wavs)
    batch_np = np.zeros((len(wavs), max_len), dtype=np.float32)
    for i, wav in enumerate(wavs):
        batch_np[i, : wav.shape[0]] = wav
    return {"indices": indices, "audio": batch_np, "errors": errors}


def _encode_audio_rows(rows: List[Dict[str, str]], cfg: DictConfig, ctx: Optional[Any] = None) -> Any:
    np = _np()
    ctx = ctx or _build_audio_embedder(cfg)
    torch = ctx.torch
    model = ctx.model
    device = ctx.device
    batch_size = int(cfg.stage.audio.batch_size)
    num_workers = max(0, int(getattr(cfg.stage.audio, "num_workers", 0) or 0))
    prefetch_factor = getattr(cfg.stage.audio, "prefetch_factor", None)
    persistent_workers = bool(getattr(cfg.stage.audio, "persistent_workers", False))
    pin_memory = bool(device.type == "cuda")
    data_loader_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": _collate_audio_batch,
    }
    if num_workers > 0:
        if prefetch_factor is not None:
            data_loader_kwargs["prefetch_factor"] = int(prefetch_factor)
        data_loader_kwargs["persistent_workers"] = persistent_workers

    dataset = _AudioClipDataset(rows, _audio_loading_cfg(cfg))
    loader = torch.utils.data.DataLoader(dataset, **data_loader_kwargs)
    embeddings_by_idx: Dict[int, Any] = {}
    failed_by_idx: Dict[int, str] = {}
    total = len(rows)
    total_batches = int(math.ceil(total / max(1, batch_size))) if total else 0

    def _run_audio_forward(batch_indices: List[int], batch_tensor: Any) -> None:
        if not batch_indices:
            return
        try:
            audio_embs = model(wavs=batch_tensor)
        except Exception as exc:
            _cleanup_device_cache(ctx, aggressive=True)
            if len(batch_indices) == 1:
                failed_idx = int(batch_indices[0])
                failed_by_idx[failed_idx] = f"{type(exc).__name__}: {exc}"
                row = rows[failed_idx]
                clip_id = str(row.get("clip_id", "") or row.get("track_id", "") or failed_idx)
                _log(cfg, f"Skipping audio for clip {clip_id} after model error: {failed_by_idx[failed_idx]}")
                return

            split = max(1, len(batch_indices) // 2)
            _log(
                cfg,
                f"Audio batch of {len(batch_indices)} failed on {device}; retrying smaller sub-batches "
                f"to isolate bad sample(s). Root error: {type(exc).__name__}: {exc}",
            )
            left_indices = batch_indices[:split]
            right_indices = batch_indices[split:]
            _run_audio_forward(left_indices, batch_tensor[:split])
            _run_audio_forward(right_indices, batch_tensor[split:])
            return

        if bool(cfg.stage.embedding.normalize):
            audio_embs = torch.nn.functional.normalize(audio_embs, dim=-1)
        batch_embs = audio_embs.detach().cpu().numpy().astype(np.float32)
        for idx, emb in zip(batch_indices, batch_embs):
            embeddings_by_idx[int(idx)] = emb
        del batch_embs
        del audio_embs

    with torch.no_grad():
        processed = 0
        for batch_rows in rich_tqdm_iter(
            cfg,
            loader,
            total=total_batches,
            desc="Audio embeddings",
            unit="batch",
        ):
            for error in batch_rows.get("errors", []):
                failed_idx = int(error["idx"])
                failed_by_idx[failed_idx] = str(error["error"])
                row = rows[failed_idx]
                clip_id = str(row.get("clip_id", "") or row.get("track_id", "") or failed_idx)
                _log(cfg, f"Skipping corrupted audio for clip {clip_id}: {failed_by_idx[failed_idx]}")
            batch_indices = list(batch_rows["indices"])
            if not batch_indices:
                continue
            batch_np = batch_rows["audio"]
            batch_tensor = torch.as_tensor(batch_np)
            batch_tensor = batch_tensor.to(device, non_blocking=(device.type == "cuda"))
            _run_audio_forward(batch_indices, batch_tensor)
            del batch_tensor
            del batch_np
            processed += len(batch_indices)
            if total > 0 and (processed % max(1, int(cfg.stage.progress.every_n_rows)) == 0 or processed == total):
                _log(cfg, f"Audio rows processed: {processed:,}")

    if not embeddings_by_idx:
        embeddings = np.zeros((0, 0), dtype=np.float32)
    else:
        ordered_indices = sorted(embeddings_by_idx)
        embeddings = np.stack([embeddings_by_idx[i] for i in ordered_indices], axis=0).astype(np.float32, copy=False)
    return SimpleNamespace(embeddings=embeddings, ordered_indices=sorted(embeddings_by_idx), failed_by_idx=failed_by_idx)


def _build_text_embedder(cfg: DictConfig):
    from transformers import AutoModel, AutoTokenizer

    runtime = _resolve_torch_device(cfg)
    torch = runtime.torch
    device = runtime.device
    auth_cfg = getattr(cfg.stage, "auth", SimpleNamespace(hf_token_env="HF_TOKEN"))
    token_env = str(getattr(auth_cfg, "hf_token_env", "HF_TOKEN"))
    token = os.environ.get(token_env, "").strip() or None
    _log(cfg, f"Loading text model {cfg.stage.models.text_model_id} on {device}")
    if token is None:
        _log(cfg, f"No Hugging Face token found in ${token_env}; gated model downloads may fail.")
    tokenizer = AutoTokenizer.from_pretrained(str(cfg.stage.models.text_model_id), token=token)
    model = AutoModel.from_pretrained(str(cfg.stage.models.text_model_id), token=token)
    model = model.to(device)
    model.eval()
    return SimpleNamespace(model=model, tokenizer=tokenizer, torch=torch, device=device)


def _encode_text_rows(texts: List[str], cfg: DictConfig, ctx: Optional[Any] = None) -> Any:
    np = _np()
    ctx = ctx or _build_text_embedder(cfg)
    model = ctx.model
    tokenizer = ctx.tokenizer
    torch = ctx.torch
    device = ctx.device
    batch_size = int(cfg.stage.text.batch_size)
    truncate_dim = cfg.stage.text.truncate_dim
    max_length = int(getattr(cfg.stage.text, "max_length", 512))
    total = len(texts)
    total_batches = int(math.ceil(total / max(1, batch_size))) if total else 0
    all_batches: List[Any] = []
    processed = 0

    with torch.no_grad():
        for batch_texts in rich_tqdm_iter(
            cfg,
            _batched(texts, batch_size),
            total=total_batches,
            desc="Text embeddings",
            unit="batch",
        ):
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device, non_blocking=True) for k, v in encoded.items()}
            outputs = model(**encoded)
            token_embeddings = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            masked = token_embeddings * attention_mask
            summed = masked.sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1)
            batch_embs = summed / counts
            if bool(cfg.stage.embedding.normalize):
                batch_embs = torch.nn.functional.normalize(batch_embs, dim=-1)
            batch_embs = batch_embs.detach().cpu().numpy().astype(np.float32)
            if truncate_dim is not None:
                dim = int(truncate_dim)
                batch_embs = batch_embs[:, :dim]
                if bool(cfg.stage.embedding.normalize):
                    batch_embs = _normalize_np(batch_embs)
            all_batches.append(batch_embs)
            del outputs
            del token_embeddings
            del attention_mask
            del masked
            del summed
            del counts
            del encoded
            processed += len(batch_texts)
            if total > 0 and (processed % max(1, int(cfg.stage.progress.every_n_rows)) == 0 or processed == total):
                _log(cfg, f"Text rows processed: {processed:,}")

    return np.concatenate(all_batches, axis=0) if all_batches else np.zeros((0, 0), dtype=np.float32)


def _cfg_section_to_plain(obj: Any) -> Any:
    if hasattr(obj, "items"):
        return {str(k): _cfg_section_to_plain(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {str(k): _cfg_section_to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_cfg_section_to_plain(v) for v in obj]
    if hasattr(obj, "__dict__") and not isinstance(obj, (str, bytes, int, float, bool, type(None))):
        return {str(k): _cfg_section_to_plain(v) for k, v in vars(obj).items()}
    return obj


def _build_run_signature(cfg: DictConfig, input_manifest_csv: str, row_count: int) -> Dict[str, Any]:
    payload = {
        "input_manifest_csv": input_manifest_csv,
        "row_count": row_count,
        "audio_model_id": str(cfg.stage.models.audio_model_id),
        "text_model_id": str(cfg.stage.models.text_model_id),
        "runtime": _cfg_section_to_plain(cfg.stage.runtime),
        "embedding": _cfg_section_to_plain(cfg.stage.embedding),
        "behavior": _cfg_section_to_plain(cfg.stage.behavior),
        "audio": _cfg_section_to_plain(cfg.stage.audio),
        "text": _cfg_section_to_plain(cfg.stage.text),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {
        "payload": payload,
        "hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _chunk_ranges(total_rows: int, chunk_size: int) -> Iterable[tuple[int, int]]:
    size = max(1, int(chunk_size))
    for start in range(0, total_rows, size):
        yield start, min(total_rows, start + size)


def _path_exists(path_str: str) -> bool:
    return Path(path_str).exists()


def _warn_if_legacy_output_layout(
    cfg: DictConfig,
    *,
    output_dir: Path,
    audio_dir: Path,
    text_dir: Path,
    lookup_path: Path,
) -> None:
    run_name = str(cfg.runtime.run_name)
    expected_output_suffix = Path(run_name) / "embeddings"
    expected_audio_suffix = Path("audio") / run_name
    expected_text_suffix = Path("text") / run_name

    if output_dir.parts[-len(expected_output_suffix.parts) :] != expected_output_suffix.parts:
        _log(
            cfg,
            "Warning: stage.io.output_dir does not end with '<run_name>/embeddings'. "
            f"Current value: {output_dir}",
        )
    if audio_dir.parts[-len(expected_audio_suffix.parts) :] != expected_audio_suffix.parts:
        _log(
            cfg,
            "Warning: stage.io.audio_embeddings_dir does not end with 'audio/<run_name>'. "
            f"Current value: {audio_dir}",
        )
    if text_dir.parts[-len(expected_text_suffix.parts) :] != expected_text_suffix.parts:
        _log(
            cfg,
            "Warning: stage.io.text_embeddings_dir does not end with 'text/<run_name>'. "
            f"Current value: {text_dir}",
        )
    _log(
        cfg,
        "Resolved embedding outputs: "
        f"lookup_manifest={lookup_path}, audio_dir={audio_dir}, text_dir={text_dir}",
    )


def _infer_embedding_dim(paths: Iterable[str]) -> int:
    np = _np()
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            continue
        arr = np.load(path)
        if arr.ndim == 0:
            return 1
        if arr.ndim == 1:
            return int(arr.shape[0])
        return int(arr.shape[-1])
    return 0


def _chunk_pending_indices(
    lookup_rows: List[Dict[str, Any]],
    start: int,
    end: int,
    overwrite_existing: bool,
) -> tuple[List[int], List[int]]:
    pending_audio: List[int] = []
    pending_text: List[int] = []
    for global_idx in range(start, end):
        row = lookup_rows[global_idx]
        if overwrite_existing or _normalize_embedding_status(row.get("audio_embedding_status")) != "ok":
            pending_audio.append(global_idx)
        if overwrite_existing or _normalize_embedding_status(row.get("text_embedding_status")) != "ok":
            pending_text.append(global_idx)
    return pending_audio, pending_text


def _cleanup_device_cache(*contexts: Any, aggressive: bool = False) -> None:
    gc.collect()
    for ctx in contexts:
        if ctx is None:
            continue
        torch = getattr(ctx, "torch", None)
        device = getattr(ctx, "device", None)
        if torch is None or device is None or getattr(device, "type", None) != "cuda":
            continue
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
        if aggressive:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def run_embeddings(cfg: DictConfig) -> Dict[str, object]:
    in_csv = Path(str(cfg.stage.io.input_manifest_csv))
    out_dir = Path(str(cfg.stage.io.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = Path(str(cfg.stage.io.audio_embeddings_dir))
    text_dir = Path(str(cfg.stage.io.text_embeddings_dir))
    lookup_path = out_dir / str(cfg.stage.io.lookup_manifest_file)
    report_path = out_dir / str(cfg.stage.io.report_file)
    tracker = StageTracker(
        cfg,
        "embeddings",
        title="Build Audio And Text Embeddings",
        subtitle=f"input={in_csv}",
        total_steps=5,
    )
    _warn_if_legacy_output_layout(
        cfg,
        output_dir=out_dir,
        audio_dir=audio_dir,
        text_dir=text_dir,
        lookup_path=lookup_path,
    )

    if not in_csv.exists():
        raise FileNotFoundError(f"Input manifest CSV not found: {in_csv}")

    tracker.step("Read structured manifest", detail=str(in_csv))
    rows = _read_manifest_rows(in_csv)
    rows = _limit_rows(rows, cfg.stage.behavior.max_rows)
    text_inputs = [_pick_text_input(row, cfg) for row in rows]
    lookup_rows = _build_lookup_rows(rows, text_inputs)
    _attach_embedding_paths(lookup_rows, audio_dir, text_dir)
    overwrite_existing = bool(cfg.stage.behavior.overwrite_existing)
    manifest_only = bool(getattr(cfg.stage.behavior, "manifest_only", False))
    reuse_lookup_manifest = bool(getattr(cfg.stage.behavior, "reuse_lookup_manifest", True))
    verify_existing_files = bool(getattr(cfg.stage.behavior, "verify_existing_files", False))
    lookup_dirty = True
    reused_lookup_manifest = False
    requires_status_upgrade = False

    tracker.step("Prepare resume state", detail=f"lookup={lookup_path.name}")
    if lookup_path.exists() and reuse_lookup_manifest and not overwrite_existing:
        existing_lookup_rows = _read_manifest_rows(lookup_path)
        if len(existing_lookup_rows) == len(lookup_rows):
            requires_status_upgrade = _merge_existing_lookup_rows(lookup_rows, existing_lookup_rows)
            reused_lookup_manifest = True
            lookup_dirty = requires_status_upgrade
            _log(cfg, f"Reused existing lookup manifest: {lookup_path}")
        else:
            _log(
                cfg,
                "Existing lookup manifest row count does not match the input manifest; rebuilding resume metadata.",
            )

    if overwrite_existing:
        _refresh_lookup_audio_statuses(lookup_rows, overwrite_existing=True)
        for row in lookup_rows:
            row["text_embedding_status"] = "pending"
            row["text_error"] = ""
    elif verify_existing_files or requires_status_upgrade or not reused_lookup_manifest:
        if verify_existing_files:
            _log(cfg, "Verifying existing embedding files from disk for resume state.")
        elif requires_status_upgrade:
            _log(cfg, "Upgrading lookup manifest resume statuses from legacy schema.")
        _refresh_lookup_statuses_from_files(lookup_rows, overwrite_existing=False)
        lookup_dirty = True
    elif _has_pending_status(lookup_rows):
        _log(
            cfg,
            "Lookup manifest has pending rows; checking disk for embeddings written by an interrupted run.",
        )
        _refresh_lookup_statuses_from_files(lookup_rows, overwrite_existing=False)
        lookup_dirty = True

    if _needs_status_verification(lookup_rows):
        _log(cfg, "Resume statuses are incomplete; verifying existing embedding files from disk.")
        _refresh_lookup_statuses_from_files(lookup_rows, overwrite_existing=overwrite_existing)
        lookup_dirty = True

    if lookup_dirty:
        _write_lookup_manifest(lookup_path, lookup_rows)
        _log(cfg, f"Lookup manifest written: {lookup_path}")
    else:
        _log(cfg, f"Lookup manifest unchanged: {lookup_path}")
    signature = _build_run_signature(cfg, str(in_csv), len(rows))
    chunk_size = int(cfg.stage.behavior.resume_chunk_size)
    total_chunks = int(math.ceil(len(rows) / max(1, chunk_size))) if rows else 0

    existing_audio_count = _count_status(lookup_rows, "audio_embedding_status", "ok")
    existing_text_count = _count_status(lookup_rows, "text_embedding_status", "ok")
    if manifest_only:
        report = {
            "stage": "embeddings",
            "mode": "manifest_only",
            "input": {
                "input_manifest_csv": str(in_csv),
                "audio_model_id": str(cfg.stage.models.audio_model_id),
                "text_model_id": str(cfg.stage.models.text_model_id),
                "signature_hash": signature["hash"],
            },
            "counts": {
                "rows_embedded": len(rows),
                "lookup_rows_written": len(lookup_rows),
                "existing_audio_files": existing_audio_count,
                "existing_text_files": existing_text_count,
                "chunks_total": total_chunks,
                "chunks_processed": 0,
                "chunks_skipped_existing": 0,
                "audio_rows_written": 0,
                "audio_rows_failed": 0,
                "text_rows_written": 0,
                "audio_rows_skipped_existing": 0,
                "text_rows_skipped_existing": 0,
                "audio_embedding_dim": _infer_embedding_dim(row["audio_embedding_path"] for row in lookup_rows),
                "text_embedding_dim": _infer_embedding_dim(row["text_embedding_path"] for row in lookup_rows),
            },
            "outputs": {
                "audio_embeddings_dir": str(audio_dir),
                "text_embeddings_dir": str(text_dir),
                "lookup_manifest": str(lookup_path),
                "report": str(report_path),
            },
        }
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=True)
        tracker.step("Write manifest-only report", detail=report_path.name)
        tracker.finish(
            f"manifest rows={len(rows):,}, existing audio={existing_audio_count:,}, existing text={existing_text_count:,}"
        )
        _log(
            cfg,
            f"Manifest-only mode complete. Rows={len(rows):,}, "
            f"existing audio files={existing_audio_count:,}, existing text files={existing_text_count:,}",
        )
        return report

    if overwrite_existing:
        _log(cfg, "Overwrite requested. Existing per-clip embedding files will be replaced.")
        for row in lookup_rows:
            _remove_file_if_exists(Path(str(row["audio_embedding_path"])))
            _remove_file_if_exists(Path(str(row["text_embedding_path"])))
        _refresh_lookup_audio_statuses(lookup_rows, overwrite_existing=overwrite_existing)
        for row in lookup_rows:
            row["text_embedding_status"] = "pending"
            row["text_error"] = ""
        _write_lookup_manifest(lookup_path, lookup_rows)

    _log(cfg, f"Preparing embeddings for {len(rows):,} rows")
    if overwrite_existing:
        _log(
            cfg,
            f"Resume state before overwrite: existing audio files={existing_audio_count:,}, "
            f"existing text files={existing_text_count:,}",
        )
    else:
        _log(
            cfg,
            f"Resume state: existing audio files={existing_audio_count:,}, "
            f"existing text files={existing_text_count:,}, "
            f"pending audio rows={len(rows) - existing_audio_count:,}, "
            f"pending text rows={len(rows) - existing_text_count:,}",
        )
    pending_audio_any = any(
        overwrite_existing or _normalize_embedding_status(row.get("audio_embedding_status")) != "ok"
        for row in lookup_rows
    )
    pending_text_any = any(
        overwrite_existing or _normalize_embedding_status(row.get("text_embedding_status")) != "ok"
        for row in lookup_rows
    )

    tracker.step("Resolve embedding models", detail=f"rows={len(rows):,}")
    audio_ctx = _build_audio_embedder(cfg) if pending_audio_any else None
    text_ctx = _build_text_embedder(cfg) if pending_text_any else None
    if audio_ctx is not None:
        _log(cfg, f"Resolved audio device: {audio_ctx.device}")
    else:
        _log(cfg, "All audio embedding files already exist; audio encoding will be skipped.")
    if text_ctx is not None:
        _log(cfg, f"Resolved text device: {text_ctx.device}")
    else:
        _log(cfg, "All text embedding files already exist; text encoding will be skipped.")

    chunks_skipped = 0
    chunks_processed = 0
    audio_rows_written = 0
    audio_rows_failed = 0
    text_rows_written = 0
    audio_rows_skipped_existing = 0
    text_rows_skipped_existing = 0
    audio_embedding_dim = _infer_embedding_dim(row["audio_embedding_path"] for row in lookup_rows)
    text_embedding_dim = _infer_embedding_dim(row["text_embedding_path"] for row in lookup_rows)

    tracker.step("Encode pending chunks", detail=f"chunk_size={chunk_size:,}, total_chunks={total_chunks:,}")
    for chunk_idx, (start, end) in enumerate(_chunk_ranges(len(rows), chunk_size), start=1):
        pending_audio_idx, pending_text_idx = _chunk_pending_indices(lookup_rows, start, end, overwrite_existing)
        audio_rows_skipped_existing += (end - start) - len(pending_audio_idx)
        text_rows_skipped_existing += (end - start) - len(pending_text_idx)
        if not pending_audio_idx and not pending_text_idx:
            chunks_skipped += 1
            continue

        if pending_audio_idx:
            chunk_rows_audio = [rows[i] for i in pending_audio_idx]
            audio_result = _encode_audio_rows(chunk_rows_audio, cfg, ctx=audio_ctx)
            audio_embs = audio_result.embeddings
            if audio_embs.size:
                audio_embedding_dim = int(audio_embs.shape[1])
            for failed_local_idx, error_message in audio_result.failed_by_idx.items():
                failed_global_idx = pending_audio_idx[failed_local_idx]
                lookup_rows[failed_global_idx]["audio_embedding_status"] = "failed"
                lookup_rows[failed_global_idx]["audio_error"] = str(error_message)
                audio_rows_failed += 1
            for emb_local_idx, local_idx in enumerate(audio_result.ordered_indices):
                global_idx = pending_audio_idx[local_idx]
                audio_path = Path(str(lookup_rows[global_idx]["audio_embedding_path"]))
                _save_npy_array(audio_path, audio_embs[emb_local_idx])
                lookup_rows[global_idx]["audio_embedding_status"] = "ok"
                lookup_rows[global_idx]["audio_error"] = ""
            audio_rows_written += len(audio_result.ordered_indices)
            del audio_embs
            del audio_result

        if pending_text_idx:
            chunk_texts = [text_inputs[i] for i in pending_text_idx]
            text_embs = _encode_text_rows(chunk_texts, cfg, ctx=text_ctx)
            if text_embs.size:
                text_embedding_dim = int(text_embs.shape[1])
            for local_idx, global_idx in enumerate(pending_text_idx):
                text_path = Path(str(lookup_rows[global_idx]["text_embedding_path"]))
                _save_npy_array(text_path, text_embs[local_idx])
                lookup_rows[global_idx]["text_embedding_status"] = "ok"
                lookup_rows[global_idx]["text_error"] = ""
            text_rows_written += len(pending_text_idx)
            del text_embs

        chunks_processed += 1
        _cleanup_device_cache(
            audio_ctx,
            text_ctx,
            aggressive=bool(getattr(cfg.stage.runtime, "empty_cache_between_chunks", False)),
        )
        _log(cfg, f"Completed chunk {chunk_idx}/{total_chunks}")

    for row in lookup_rows:
        if _normalize_embedding_status(row.get("audio_embedding_status")) != "failed":
            if _normalize_embedding_status(row.get("audio_embedding_status")) != "ok":
                row["audio_embedding_status"] = "pending"
                row["audio_error"] = ""
        if _normalize_embedding_status(row.get("text_embedding_status")) != "failed":
            if _normalize_embedding_status(row.get("text_embedding_status")) != "ok":
                row["text_embedding_status"] = "pending"
                row["text_error"] = ""
    _write_lookup_manifest(lookup_path, lookup_rows)

    report = {
        "stage": "embeddings",
        "mode": "full",
        "input": {
            "input_manifest_csv": str(in_csv),
            "audio_model_id": str(cfg.stage.models.audio_model_id),
            "text_model_id": str(cfg.stage.models.text_model_id),
            "signature_hash": signature["hash"],
        },
        "counts": {
            "rows_embedded": len(rows),
            "lookup_rows_written": len(lookup_rows),
            "existing_audio_files": existing_audio_count,
            "existing_text_files": existing_text_count,
            "chunks_total": total_chunks,
            "chunks_processed": chunks_processed,
            "chunks_skipped_existing": chunks_skipped,
            "audio_rows_written": audio_rows_written,
            "audio_rows_failed": audio_rows_failed,
            "text_rows_written": text_rows_written,
            "audio_rows_skipped_existing": audio_rows_skipped_existing,
            "text_rows_skipped_existing": text_rows_skipped_existing,
            "audio_embedding_dim": audio_embedding_dim,
            "text_embedding_dim": text_embedding_dim,
        },
        "outputs": {
            "audio_embeddings_dir": str(audio_dir),
            "text_embeddings_dir": str(text_dir),
            "lookup_manifest": str(lookup_path),
            "report": str(report_path),
        },
    }

    tracker.step("Write final manifests and report", detail=f"lookup={lookup_path.name}, report={report_path.name}")
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)

    tracker.finish(
        f"audio_written={audio_rows_written:,}, text_written={text_rows_written:,}, failed_audio={audio_rows_failed:,}"
    )
    _log(
        cfg,
        f"Embedding extraction complete. Rows={len(rows):,}, "
        f"audio_dim={report['counts']['audio_embedding_dim']}, "
        f"text_dim={report['counts']['text_embedding_dim']}",
    )
    return report


def _main_impl(cfg: DictConfig) -> None:
    report = run_embeddings(cfg)
    print(json.dumps({"status": "ok", "stage": "embeddings", "outputs": report["outputs"]}, indent=2))


def main() -> None:
    import hydra

    @hydra.main(version_base=None, config_path=CONF_DIR, config_name="config")
    def _wrapped(cfg: DictConfig) -> None:
        _main_impl(cfg)

    _wrapped()


if __name__ == "__main__":
    main()
