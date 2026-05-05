from __future__ import annotations

import csv
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Sequence

from jamendo_instruct.embedding_paths import embedding_paths_for_clip
from jamendo_instruct.progress import StageTracker, rich_tqdm

if TYPE_CHECKING:
    from omegaconf import DictConfig
else:
    DictConfig = Any

CONF_DIR = str(Path(__file__).resolve().parents[3] / "conf")


def _np():
    import numpy as np

    return np


def _faiss():
    try:
        import faiss
    except Exception as exc:  # pragma: no cover - import depends on local env
        raise RuntimeError(
            "FAISS audio backend requested, but the 'faiss' module is not installed. "
            "Install faiss-cpu/faiss-gpu or set stage.retrieval.audio_backend=brute_force."
        ) from exc
    return faiss


def _log(cfg: DictConfig, message: str) -> None:
    if bool(cfg.stage.progress.enabled):
        print(f"[neighborhood] {message}", flush=True)


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


def _read_csv_rows(path: Path, *, limit: int | None = None) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _load_structured_index(path: Path, *, required_clip_ids: set[str] | None = None) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip_id = str(row.get("clip_id", "") or "").strip()
            if not clip_id:
                continue
            if required_clip_ids is not None and clip_id not in required_clip_ids:
                continue
            out[clip_id] = row
            if required_clip_ids is not None and len(out) >= len(required_clip_ids):
                break
    return out


def _limit_rows(rows: List[Dict[str, str]], max_rows: Any) -> List[Dict[str, str]]:
    if max_rows in (None, "", 0):
        return rows
    return rows[: int(max_rows)]


def _scan_embedding_filenames(directory: Path) -> set[str]:
    if not directory.exists():
        return set()
    out: set[str] = set()
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(".npy"):
                out.add(entry.name)
    return out


def _prepare_lookup_rows_from_dirs(cfg: DictConfig) -> List[Dict[str, str]]:
    structured_path = Path(str(cfg.stage.io.input_manifest_csv))
    if not structured_path.exists():
        raise FileNotFoundError(f"Structured manifest CSV not found: {structured_path}")
    audio_dir = Path(str(cfg.stage.io.audio_embeddings_dir))
    text_dir = Path(str(cfg.stage.io.text_embeddings_dir))
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio embeddings directory not found: {audio_dir}")
    if not text_dir.exists():
        raise FileNotFoundError(f"Text embeddings directory not found: {text_dir}")

    max_rows = cfg.stage.behavior.max_rows
    structured_rows = _read_csv_rows(
        structured_path,
        limit=None if max_rows in (None, "", 0) else int(max_rows),
    )
    use_text_rerank = bool(getattr(cfg.stage.retrieval, "use_text_rerank", True))
    require_text = bool(cfg.stage.behavior.require_text_embeddings) or use_text_rerank
    audio_files = _scan_embedding_filenames(audio_dir)
    text_files = _scan_embedding_filenames(text_dir) if require_text else set()
    out: List[Dict[str, str]] = []
    missing_clip_id = 0
    missing_audio = 0
    missing_text = 0

    for row in structured_rows:
        clip_id = str(row.get("clip_id", "") or "").strip()
        if not clip_id:
            missing_clip_id += 1
            continue

        derived_paths = embedding_paths_for_clip(clip_id, audio_dir=audio_dir, text_dir=text_dir)
        audio_name = Path(derived_paths["audio_embedding_path"]).name
        text_name = Path(derived_paths["text_embedding_path"]).name
        if audio_name not in audio_files:
            missing_audio += 1
            continue
        if require_text and text_name not in text_files:
            missing_text += 1
            continue

        merged = dict(row)
        merged.update(derived_paths)
        merged["node_idx"] = len(out)
        out.append(merged)

    _log(
        cfg,
        f"Prepared {len(out):,} indexed rows "
        f"(missing clip_id={missing_clip_id:,}, missing audio={missing_audio:,}, missing text={missing_text:,})",
    )
    return out


def _prepare_lookup_rows_from_manifest(cfg: DictConfig) -> List[Dict[str, str]]:
    structured_path = Path(str(cfg.stage.io.input_manifest_csv))
    lookup_path = Path(str(cfg.stage.io.input_lookup_manifest_csv))
    if not structured_path.exists():
        raise FileNotFoundError(f"Structured manifest CSV not found: {structured_path}")
    if not lookup_path.exists():
        raise FileNotFoundError(f"Embedding lookup manifest CSV not found: {lookup_path}")

    max_rows = cfg.stage.behavior.max_rows
    lookup_limit = None if max_rows in (None, "", 0) else int(max_rows)
    lookup_rows = _read_csv_rows(lookup_path, limit=lookup_limit)
    required_clip_ids = {
        str(row.get("clip_id", "") or "").strip()
        for row in lookup_rows
        if str(row.get("clip_id", "") or "").strip()
    }
    structured_by_clip = _load_structured_index(structured_path, required_clip_ids=required_clip_ids)

    use_text_rerank = bool(getattr(cfg.stage.retrieval, "use_text_rerank", True))
    require_text = bool(cfg.stage.behavior.require_text_embeddings) or use_text_rerank
    out: List[Dict[str, str]] = []
    missing_structured = 0
    missing_audio = 0
    missing_text = 0

    for row in lookup_rows:
        clip_id = str(row.get("clip_id", "") or "").strip()
        structured_row = structured_by_clip.get(clip_id)
        if structured_row is None:
            missing_structured += 1
            continue

        audio_path = Path(str(row.get("audio_embedding_path", "") or ""))
        text_path = Path(str(row.get("text_embedding_path", "") or ""))
        if not audio_path.exists():
            missing_audio += 1
            continue
        if require_text and not text_path.exists():
            missing_text += 1
            continue

        merged = dict(structured_row)
        merged["audio_embedding_path"] = str(audio_path)
        merged["text_embedding_path"] = str(text_path)
        merged["node_idx"] = len(out)
        out.append(merged)

    _log(
        cfg,
        f"Prepared {len(out):,} indexed rows from lookup manifest "
        f"(missing structured={missing_structured:,}, missing audio={missing_audio:,}, missing text={missing_text:,})",
    )
    return out


def _prepare_lookup_rows(cfg: DictConfig) -> List[Dict[str, str]]:
    io_cfg = getattr(cfg.stage, "io", None)
    audio_dir_raw = getattr(io_cfg, "audio_embeddings_dir", None)
    text_dir_raw = getattr(io_cfg, "text_embeddings_dir", None)
    if audio_dir_raw and text_dir_raw:
        audio_dir = Path(str(audio_dir_raw))
        text_dir = Path(str(text_dir_raw))
        if audio_dir.exists() and text_dir.exists():
            _log(cfg, f"Resolving embeddings from directories: audio={audio_dir}, text={text_dir}")
            return _prepare_lookup_rows_from_dirs(cfg)
        _log(
            cfg,
            "Embedding directories were configured but not found; falling back to lookup manifest if available.",
        )
    return _prepare_lookup_rows_from_manifest(cfg)


def _normalize_np(matrix: Any) -> Any:
    np = _np()
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def _load_embedding_matrix(
    rows: Sequence[Dict[str, str]],
    field: str,
    cfg: DictConfig,
    *,
    desc: str,
) -> Any:
    np = _np()
    arrays: List[Any] = []
    worker_count = max(1, int(getattr(cfg.stage.runtime, "embedding_load_workers", 8) or 8))
    paths = [Path(str(row.get(field, "") or "")) for row in rows]

    def _load_one(path: Path) -> Any:
        return np.load(path).astype(np.float32, copy=False).reshape(-1)

    with rich_tqdm(cfg, total=len(rows), desc=desc, unit="vec") as progress:
        if worker_count == 1 or len(paths) <= 1:
            for path in paths:
                arrays.append(_load_one(path))
                progress.update(1)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                for array in executor.map(_load_one, paths):
                    arrays.append(array)
                    progress.update(1)
    if not arrays:
        return np.zeros((0, 0), dtype=np.float32)
    matrix = np.stack(arrays, axis=0).astype(np.float32, copy=False)
    if bool(cfg.stage.retrieval.normalize_loaded_embeddings):
        matrix = _normalize_np(matrix)
    return matrix


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


def _build_tag_sets(rows: Sequence[Dict[str, str]]) -> List[set[str]]:
    return [set(_parse_json_list(row.get("normalized_tags_json", ""))) for row in rows]


def _select_seed_indices(rows: Sequence[Dict[str, str]], cfg: DictConfig) -> List[int]:
    allowed_splits = {str(x) for x in cfg.stage.behavior.seed_splits if str(x)}
    seed_indices = [
        idx
        for idx, row in enumerate(rows)
        if not allowed_splits or str(row.get("split", "") or "") in allowed_splits
    ]
    seed_limit = cfg.stage.behavior.seed_row_limit
    if seed_limit not in (None, "", 0):
        seed_indices = seed_indices[: int(seed_limit)]
    return seed_indices


def _iter_query_blocks(seed_indices: Sequence[int], chunk_size: int) -> Iterable[List[int]]:
    size = max(1, int(chunk_size))
    for start in range(0, len(seed_indices), size):
        yield list(seed_indices[start : start + size])


def _resolve_audio_backend(cfg: DictConfig) -> str:
    backend = str(getattr(cfg.stage.retrieval, "audio_backend", "auto") or "auto").strip().lower()
    if backend in {"", "auto"}:
        try:
            _faiss()
            return "faiss_flat_ip"
        except RuntimeError:
            return "brute_force"
    if backend in {"brute_force", "faiss_flat_ip"}:
        if backend == "faiss_flat_ip":
            _faiss()
        return backend
    raise ValueError(f"Unsupported neighborhood audio backend: {backend}")


def _topk_audio_neighbors_brute_force(
    query_indices: Sequence[int],
    audio_matrix: Any,
    track_ids: Sequence[str],
    splits: Sequence[str],
    file_paths: Sequence[str],
    candidate_pools_by_split: Dict[str, Any] | None,
    cfg: DictConfig,
) -> tuple[Any, Any]:
    np = _np()
    total_rows = int(audio_matrix.shape[0])
    keep = min(int(cfg.stage.retrieval.audio_top_k), max(0, total_rows - 1))
    batch_size = len(query_indices)
    if keep <= 0:
        return (
            np.full((batch_size, 0), -1, dtype=np.int64),
            np.full((batch_size, 0), -np.inf, dtype=np.float32),
        )

    query_audio = audio_matrix[list(query_indices)]
    query_tracks = np.asarray([track_ids[i] for i in query_indices], dtype=object)
    query_splits = np.asarray([splits[i] for i in query_indices], dtype=object)
    query_files = np.asarray([file_paths[i] for i in query_indices], dtype=object)
    global_query_idx = np.asarray(list(query_indices), dtype=np.int64)
    best_indices = np.full((batch_size, keep), -1, dtype=np.int64)
    best_scores = np.full((batch_size, keep), -np.inf, dtype=np.float32)
    candidate_chunk_size = max(1, int(cfg.stage.runtime.candidate_chunk_size))
    min_audio_similarity = cfg.stage.behavior.min_audio_similarity
    min_audio_threshold = None if min_audio_similarity in (None, "") else float(min_audio_similarity)

    def _merge_best(local_rows: Any, cand_idx: Any, scores: Any) -> None:
        nonlocal best_indices, best_scores
        cand_idx_full = np.broadcast_to(cand_idx, scores.shape)
        merged_scores = np.concatenate([best_scores[local_rows], scores.astype(np.float32, copy=False)], axis=1)
        merged_indices = np.concatenate([best_indices[local_rows], cand_idx_full], axis=1)
        kth = max(0, merged_scores.shape[1] - keep)
        part = np.argpartition(merged_scores, kth=kth, axis=1)[:, -keep:]
        row_ids = np.arange(merged_scores.shape[0])[:, None]
        local_best_scores = merged_scores[row_ids, part]
        local_best_indices = merged_indices[row_ids, part]
        order = np.argsort(local_best_scores, axis=1)[:, ::-1]
        best_scores[local_rows] = local_best_scores[row_ids, order]
        best_indices[local_rows] = local_best_indices[row_ids, order]

    if bool(getattr(cfg.stage.behavior, "enforce_same_split", False)) and candidate_pools_by_split:
        unique_splits = sorted({str(split) for split in query_splits.tolist()})
        for split_value in unique_splits:
            local_rows = np.where(query_splits == split_value)[0]
            split_pool = candidate_pools_by_split.get(split_value)
            if split_pool is None or int(split_pool.size) == 0:
                continue
            split_query_audio = query_audio[local_rows]
            split_query_tracks = query_tracks[local_rows]
            split_query_files = query_files[local_rows]
            split_query_idx = global_query_idx[local_rows]
            for cand_start in range(0, int(split_pool.size), candidate_chunk_size):
                cand_idx = split_pool[cand_start : cand_start + candidate_chunk_size]
                cand_audio = audio_matrix[cand_idx]
                scores = split_query_audio @ cand_audio.T
                if bool(cfg.stage.behavior.exclude_self):
                    scores = np.where(split_query_idx[:, None] == cand_idx[None, :], -np.inf, scores)
                if bool(cfg.stage.behavior.exclude_same_track):
                    cand_tracks = np.asarray([track_ids[int(i)] for i in cand_idx.tolist()], dtype=object)
                    scores = np.where(split_query_tracks[:, None] == cand_tracks[None, :], -np.inf, scores)
                if bool(cfg.stage.behavior.exclude_duplicate_file_path):
                    cand_files = np.asarray([file_paths[int(i)] for i in cand_idx.tolist()], dtype=object)
                    scores = np.where(split_query_files[:, None] == cand_files[None, :], -np.inf, scores)
                if min_audio_threshold is not None:
                    scores = np.where(scores >= min_audio_threshold, scores, -np.inf)
                _merge_best(local_rows, cand_idx, scores)
    else:
        for cand_start in range(0, total_rows, candidate_chunk_size):
            cand_end = min(total_rows, cand_start + candidate_chunk_size)
            cand_audio = audio_matrix[cand_start:cand_end]
            scores = query_audio @ cand_audio.T
            cand_idx = np.arange(cand_start, cand_end, dtype=np.int64)

            if bool(cfg.stage.behavior.exclude_self):
                scores = np.where(global_query_idx[:, None] == cand_idx[None, :], -np.inf, scores)
            if bool(cfg.stage.behavior.exclude_same_track):
                cand_tracks = np.asarray(track_ids[cand_start:cand_end], dtype=object)
                scores = np.where(query_tracks[:, None] == cand_tracks[None, :], -np.inf, scores)
            if bool(cfg.stage.behavior.exclude_duplicate_file_path):
                cand_files = np.asarray(file_paths[cand_start:cand_end], dtype=object)
                scores = np.where(query_files[:, None] == cand_files[None, :], -np.inf, scores)
            if min_audio_threshold is not None:
                scores = np.where(scores >= min_audio_threshold, scores, -np.inf)
            _merge_best(np.arange(batch_size), cand_idx, scores)

    return best_indices, best_scores


def _filter_retrieved_candidates(
    *,
    query_indices: Sequence[int],
    candidate_indices: Any,
    candidate_scores: Any,
    keep: int,
    track_ids: Sequence[str],
    splits: Sequence[str],
    file_paths: Sequence[str],
    cfg: DictConfig,
) -> tuple[Any, Any]:
    np = _np()
    batch_size = len(query_indices)
    filtered_indices = np.full((batch_size, keep), -1, dtype=np.int64)
    filtered_scores = np.full((batch_size, keep), -np.inf, dtype=np.float32)
    min_audio_similarity = cfg.stage.behavior.min_audio_similarity
    min_audio_threshold = None if min_audio_similarity in (None, "") else float(min_audio_similarity)
    enforce_same_split = bool(getattr(cfg.stage.behavior, "enforce_same_split", False))
    exclude_self = bool(cfg.stage.behavior.exclude_self)
    exclude_same_track = bool(cfg.stage.behavior.exclude_same_track)
    exclude_duplicate_file_path = bool(cfg.stage.behavior.exclude_duplicate_file_path)

    for local_idx, query_idx in enumerate(query_indices):
        query_track = track_ids[query_idx]
        query_split = splits[query_idx]
        query_file = file_paths[query_idx]
        write_pos = 0
        for pos, candidate_idx in enumerate(candidate_indices[local_idx].tolist()):
            if candidate_idx < 0:
                continue
            score = float(candidate_scores[local_idx, pos])
            if not math.isfinite(score):
                continue
            candidate_idx = int(candidate_idx)
            if exclude_self and candidate_idx == query_idx:
                continue
            if enforce_same_split and splits[candidate_idx] != query_split:
                continue
            if exclude_same_track and track_ids[candidate_idx] == query_track:
                continue
            if exclude_duplicate_file_path and file_paths[candidate_idx] == query_file:
                continue
            if min_audio_threshold is not None and score < min_audio_threshold:
                continue
            filtered_indices[local_idx, write_pos] = candidate_idx
            filtered_scores[local_idx, write_pos] = score
            write_pos += 1
            if write_pos >= keep:
                break

    return filtered_indices, filtered_scores


def _faiss_search_k(pool_size: int, keep: int, cfg: DictConfig) -> int:
    multiplier = max(1, int(getattr(cfg.stage.retrieval, "faiss_candidate_multiplier", 4)))
    minimum_extra = max(32, keep)
    return min(pool_size, max(keep * multiplier, keep + minimum_extra))


def _search_faiss_index(
    *,
    index: Any,
    row_indices: Any,
    query_audio: Any,
    query_indices: Sequence[int],
    keep: int,
    track_ids: Sequence[str],
    splits: Sequence[str],
    file_paths: Sequence[str],
    cfg: DictConfig,
) -> tuple[Any, Any]:
    np = _np()
    batch_size = len(query_indices)
    best_indices = np.full((batch_size, keep), -1, dtype=np.int64)
    best_scores = np.full((batch_size, keep), -np.inf, dtype=np.float32)
    if keep <= 0 or batch_size == 0:
        return best_indices, best_scores

    pool_size = int(row_indices.shape[0])
    if pool_size <= 0:
        return best_indices, best_scores

    pending_rows = np.arange(batch_size, dtype=np.int64)
    search_k = _faiss_search_k(pool_size, keep, cfg)

    while pending_rows.size > 0:
        local_scores, local_indices = index.search(query_audio[pending_rows], search_k)
        global_indices = np.where(
            local_indices >= 0,
            row_indices[local_indices.clip(min=0)],
            -1,
        ).astype(np.int64, copy=False)
        filtered_indices, filtered_scores = _filter_retrieved_candidates(
            query_indices=[query_indices[int(i)] for i in pending_rows.tolist()],
            candidate_indices=global_indices,
            candidate_scores=local_scores,
            keep=keep,
            track_ids=track_ids,
            splits=splits,
            file_paths=file_paths,
            cfg=cfg,
        )
        best_indices[pending_rows] = filtered_indices
        best_scores[pending_rows] = filtered_scores
        filled_counts = np.sum(filtered_indices >= 0, axis=1)
        needs_more = filled_counts < keep
        if not bool(needs_more.any()) or search_k >= pool_size:
            break
        pending_rows = pending_rows[needs_more]
        next_search_k = min(pool_size, max(search_k + 1, search_k * 2))
        if next_search_k == search_k:
            break
        search_k = next_search_k

    return best_indices, best_scores


def _build_audio_search_state(
    audio_matrix: Any,
    candidate_pools_by_split: Dict[str, Any] | None,
    cfg: DictConfig,
    *,
    audio_backend: str,
) -> Dict[str, Any] | None:
    if audio_backend != "faiss_flat_ip":
        return None

    faiss = _faiss()
    if bool(getattr(cfg.stage.behavior, "enforce_same_split", False)) and candidate_pools_by_split:
        by_split: Dict[str, Dict[str, Any]] = {}
        for split, row_indices in candidate_pools_by_split.items():
            split_audio = audio_matrix[row_indices].astype(_np().float32, copy=False)
            index = faiss.IndexFlatIP(int(split_audio.shape[1]))
            index.add(split_audio)
            by_split[str(split)] = {"index": index, "row_indices": row_indices}
        return {"by_split": by_split}

    row_indices = _np().arange(int(audio_matrix.shape[0]), dtype=_np().int64)
    index = faiss.IndexFlatIP(int(audio_matrix.shape[1]))
    index.add(audio_matrix.astype(_np().float32, copy=False))
    return {"global": {"index": index, "row_indices": row_indices}}


def _topk_audio_neighbors_faiss(
    query_indices: Sequence[int],
    audio_matrix: Any,
    track_ids: Sequence[str],
    splits: Sequence[str],
    file_paths: Sequence[str],
    candidate_pools_by_split: Dict[str, Any] | None,
    audio_search_state: Dict[str, Any] | None,
    cfg: DictConfig,
) -> tuple[Any, Any]:
    np = _np()
    total_rows = int(audio_matrix.shape[0])
    keep = min(int(cfg.stage.retrieval.audio_top_k), max(0, total_rows - 1))
    batch_size = len(query_indices)
    if keep <= 0:
        return (
            np.full((batch_size, 0), -1, dtype=np.int64),
            np.full((batch_size, 0), -np.inf, dtype=np.float32),
        )

    query_audio = audio_matrix[list(query_indices)].astype(np.float32, copy=False)
    best_indices = np.full((batch_size, keep), -1, dtype=np.int64)
    best_scores = np.full((batch_size, keep), -np.inf, dtype=np.float32)

    if bool(getattr(cfg.stage.behavior, "enforce_same_split", False)) and candidate_pools_by_split:
        query_splits = np.asarray([splits[i] for i in query_indices], dtype=object)
        by_split = {} if audio_search_state is None else dict(audio_search_state.get("by_split", {}))
        for split_value in sorted({str(split) for split in query_splits.tolist()}):
            local_rows = np.where(query_splits == split_value)[0]
            split_state = by_split.get(split_value)
            if split_state is None:
                continue
            local_best_indices, local_best_scores = _search_faiss_index(
                index=split_state["index"],
                row_indices=split_state["row_indices"],
                query_audio=query_audio,
                query_indices=[query_indices[int(i)] for i in local_rows.tolist()],
                keep=keep,
                track_ids=track_ids,
                splits=splits,
                file_paths=file_paths,
                cfg=cfg,
            )
            best_indices[local_rows] = local_best_indices
            best_scores[local_rows] = local_best_scores
    else:
        if audio_search_state is None or "global" not in audio_search_state:
            raise ValueError("FAISS audio backend is enabled, but the global search index was not initialized.")
        global_state = audio_search_state["global"]
        return _search_faiss_index(
            index=global_state["index"],
            row_indices=global_state["row_indices"],
            query_audio=query_audio,
            query_indices=query_indices,
            keep=keep,
            track_ids=track_ids,
            splits=splits,
            file_paths=file_paths,
            cfg=cfg,
        )

    return best_indices, best_scores


def _topk_audio_neighbors(
    query_indices: Sequence[int],
    audio_matrix: Any,
    track_ids: Sequence[str],
    splits: Sequence[str],
    file_paths: Sequence[str],
    candidate_pools_by_split: Dict[str, Any] | None,
    audio_search_state: Dict[str, Any] | None,
    audio_backend: str,
    cfg: DictConfig,
) -> tuple[Any, Any]:
    if audio_backend == "faiss_flat_ip":
        return _topk_audio_neighbors_faiss(
            query_indices,
            audio_matrix,
            track_ids,
            splits,
            file_paths,
            candidate_pools_by_split,
            audio_search_state,
            cfg,
        )
    return _topk_audio_neighbors_brute_force(
        query_indices,
        audio_matrix,
        track_ids,
        splits,
        file_paths,
        candidate_pools_by_split,
        cfg,
    )


def _write_query_neighbors(
    writer: csv.DictWriter,
    query_indices: Sequence[int],
    candidate_indices: Any,
    candidate_audio_scores: Any,
    rows: Sequence[Dict[str, str]],
    text_matrix: Any | None,
    tag_sets: Sequence[set[str]],
    cfg: DictConfig,
) -> tuple[int, int]:
    np = _np()
    retained_rows = 0
    filtered_tag_delta_rows = 0
    retain_top_k = min(int(cfg.stage.retrieval.retain_top_k), int(candidate_indices.shape[1]))
    audio_weight = float(cfg.stage.retrieval.rerank_audio_weight)
    text_weight = float(cfg.stage.retrieval.rerank_text_weight)
    use_text_rerank = bool(getattr(cfg.stage.retrieval, "use_text_rerank", True))
    max_changed_tags = getattr(cfg.stage.behavior, "max_changed_tags_pre_rerank", None)
    max_changed_tags = None if max_changed_tags in (None, "", 0) else int(max_changed_tags)

    for local_idx, query_idx in enumerate(query_indices):
        raw_candidate_idx = candidate_indices[local_idx]
        raw_audio_scores = candidate_audio_scores[local_idx]
        valid_mask = (raw_candidate_idx >= 0) & np.isfinite(raw_audio_scores)
        cand_idx = raw_candidate_idx[valid_mask]
        audio_scores = raw_audio_scores[valid_mask].astype(np.float32, copy=False)
        if cand_idx.size == 0:
            continue

        if max_changed_tags is not None:
            query_tags = tag_sets[query_idx]
            kept_positions: List[int] = []
            for pos, neighbor_idx in enumerate(cand_idx.tolist()):
                neighbor_tags = tag_sets[int(neighbor_idx)]
                changed = len(query_tags - neighbor_tags) + len(neighbor_tags - query_tags)
                if changed <= max_changed_tags:
                    kept_positions.append(pos)
                else:
                    filtered_tag_delta_rows += 1
            if not kept_positions:
                continue
            kept_pos_np = np.asarray(kept_positions, dtype=np.int64)
            cand_idx = cand_idx[kept_pos_np]
            audio_scores = audio_scores[kept_pos_np]

        if use_text_rerank:
            if text_matrix is None:
                raise ValueError("Text reranking is enabled, but no text embedding matrix was loaded.")
            text_scores = text_matrix[cand_idx] @ text_matrix[query_idx]
            rerank_scores = (audio_weight * audio_scores) + (text_weight * text_scores.astype(np.float32, copy=False))
            rerank_order = np.argsort(rerank_scores)[::-1]
        else:
            text_scores = np.zeros_like(audio_scores, dtype=np.float32)
            rerank_scores = audio_scores.astype(np.float32, copy=False)
            rerank_order = np.arange(cand_idx.size, dtype=np.int64)
        keep = min(retain_top_k, int(cand_idx.size))
        audio_rank_lookup = {int(idx): rank for rank, idx in enumerate(cand_idx.tolist(), start=1)}
        query_row = rows[query_idx]

        for rerank_rank, order_pos in enumerate(rerank_order[:keep], start=1):
            neighbor_idx = int(cand_idx[order_pos])
            neighbor_row = rows[neighbor_idx]
            writer.writerow(
                {
                    "source_node_idx": int(query_row.get("node_idx", query_idx)),
                    "target_node_idx": int(neighbor_row.get("node_idx", neighbor_idx)),
                    "audio_rank": audio_rank_lookup[neighbor_idx],
                    "rerank_rank": rerank_rank,
                    "audio_similarity": f"{float(audio_scores[order_pos]):.8f}",
                    "text_similarity": f"{float(text_scores[order_pos]):.8f}",
                    "rerank_score": f"{float(rerank_scores[order_pos]):.8f}",
                }
            )
            retained_rows += 1

    return retained_rows, filtered_tag_delta_rows


def _chunk_meta_path(chunks_dir: Path, chunk_idx: int) -> Path:
    return chunks_dir / f"edges_chunk_{chunk_idx:06d}.json"


def _load_chunk_meta(path: Path) -> Dict[str, int] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "chunk_idx": int(data.get("chunk_idx", 0)),
        "queries_processed": int(data.get("queries_processed", 0)),
        "output_rows": int(data.get("output_rows", 0)),
        "filtered_tag_delta_rows": int(data.get("filtered_tag_delta_rows", 0)),
    }


def _write_chunk_neighbors(
    *,
    writer: csv.DictWriter,
    chunk_meta: Path,
    chunk_idx: int,
    query_block: Sequence[int],
    candidate_indices: Any,
    candidate_audio_scores: Any,
    rows: Sequence[Dict[str, str]],
    text_matrix: Any | None,
    tag_sets: Sequence[set[str]],
    cfg: DictConfig,
) -> Dict[str, int]:
    output_rows, filtered_tag_delta_rows = _write_query_neighbors(
        writer,
        query_block,
        candidate_indices,
        candidate_audio_scores,
        rows,
        text_matrix,
        tag_sets,
        cfg,
    )
    meta = {
        "chunk_idx": int(chunk_idx),
        "queries_processed": int(len(query_block)),
        "output_rows": int(output_rows),
        "filtered_tag_delta_rows": int(filtered_tag_delta_rows),
    }
    chunk_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _collect_resumable_chunk_metas(
    *,
    chunks_dir: Path,
    total_chunks: int,
) -> List[Dict[str, int]]:
    completed: List[Dict[str, int]] = []
    for chunk_idx in range(total_chunks):
        chunk_meta = _chunk_meta_path(chunks_dir, chunk_idx)
        meta = _load_chunk_meta(chunk_meta)
        if meta is None:
            break
        completed.append(meta)
    return completed


def _clear_chunk_metas(chunks_dir: Path) -> None:
    for path in chunks_dir.glob("edges_chunk_*.json"):
        path.unlink()


def run_neighborhood(cfg: DictConfig) -> Dict[str, object]:
    tracker = StageTracker(
        cfg,
        "neighborhood",
        title="Build Retrieval Neighborhoods",
        subtitle=f"manifest={cfg.stage.io.input_manifest_csv}",
        total_steps=6,
    )
    tracker.step("Resolve embedding lookup rows", detail=str(cfg.stage.io.input_lookup_manifest_csv))
    rows = _prepare_lookup_rows(cfg)
    if not rows:
        raise ValueError("No embedding lookup rows are available for neighborhood retrieval.")

    out_dir = Path(str(cfg.stage.io.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes_csv = out_dir / str(cfg.stage.io.output_nodes_csv)
    edges_csv = out_dir / str(cfg.stage.io.output_edges_csv)
    report_path = out_dir / str(cfg.stage.io.report_file)

    seed_indices = _select_seed_indices(rows, cfg)
    if not seed_indices:
        raise ValueError("No seed rows selected for neighborhood retrieval.")

    audio_backend = _resolve_audio_backend(cfg)
    use_text_rerank = bool(getattr(cfg.stage.retrieval, "use_text_rerank", True))
    tracker.step("Load embedding caches", detail=f"audio_rows={len(rows):,}, text_rerank={use_text_rerank}")
    _log(cfg, f"Loading {len(rows):,} audio embeddings")
    audio_matrix = _load_embedding_matrix(rows, "audio_embedding_path", cfg, desc="Audio cache")
    text_matrix = None
    if use_text_rerank:
        _log(cfg, f"Loading {len(rows):,} text embeddings")
        text_matrix = _load_embedding_matrix(rows, "text_embedding_path", cfg, desc="Text cache")
    else:
        _log(cfg, "Skipping text embedding cache because stage.retrieval.use_text_rerank=false")
    tag_sets = _build_tag_sets(rows)

    track_ids = [str(row.get("track_id", "") or "") for row in rows]
    splits = [str(row.get("split", "") or "") for row in rows]
    file_paths = [str(row.get("file_path", "") or "") for row in rows]
    candidate_pools_by_split = None
    if bool(getattr(cfg.stage.behavior, "enforce_same_split", False)):
        np = _np()
        split_to_indices: Dict[str, List[int]] = {}
        for idx, split in enumerate(splits):
            split_to_indices.setdefault(split, []).append(idx)
        candidate_pools_by_split = {
            split: np.asarray(indices, dtype=np.int64) for split, indices in split_to_indices.items()
        }
    audio_search_state = _build_audio_search_state(
        audio_matrix,
        candidate_pools_by_split,
        cfg,
        audio_backend=audio_backend,
    )
    tracker.step("Prepare search state", detail=f"backend={audio_backend}, seeds={len(seed_indices):,}")
    every_n = max(1, int(cfg.stage.progress.every_n_rows))
    query_chunk_size = max(1, int(cfg.stage.runtime.query_chunk_size))
    total_seed_rows = len(seed_indices)
    total_chunks = int(math.ceil(total_seed_rows / query_chunk_size))
    queries_processed = 0
    output_rows = 0
    filtered_tag_delta_rows = 0
    chunks_processed = 0
    chunks_skipped_existing = 0
    chunks_dir = out_dir / "edge_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    overwrite_existing = bool(getattr(cfg.stage.behavior, "overwrite_existing", False))
    resume_existing_chunks = bool(getattr(cfg.stage.behavior, "resume_existing_chunks", True))

    node_fieldnames = [
        "node_idx",
        "clip_id",
        "track_id",
    ]
    edge_fieldnames = [
        "source_node_idx",
        "target_node_idx",
        "audio_rank",
        "rerank_rank",
        "audio_similarity",
        "text_similarity",
        "rerank_score",
    ]

    can_resume_existing = resume_existing_chunks and not overwrite_existing
    completed_chunk_metas: List[Dict[str, int]] = []
    if can_resume_existing and edges_csv.exists():
        completed_chunk_metas = _collect_resumable_chunk_metas(chunks_dir=chunks_dir, total_chunks=total_chunks)
    elif can_resume_existing and not edges_csv.exists():
        _log(cfg, "Ignoring neighborhood chunk resume metadata because output_edges_csv is missing.")

    append_existing_output = bool(completed_chunk_metas) and edges_csv.exists()
    if append_existing_output:
        resumed_queries = sum(int(meta["queries_processed"]) for meta in completed_chunk_metas)
        _log(
            cfg,
            f"Resuming neighborhood edges append from chunk {len(completed_chunk_metas):,} "
            f"({resumed_queries:,} seed rows already written)",
        )
    else:
        _clear_chunk_metas(chunks_dir)

    tracker.step("Write node index", detail=f"{len(rows):,} nodes")
    with nodes_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=node_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "node_idx": int(row.get("node_idx", 0)),
                    "clip_id": str(row.get("clip_id", "")),
                    "track_id": str(row.get("track_id", "")),
                }
            )

    edge_file_mode = "a" if append_existing_output else "w"
    tracker.step("Score and write neighborhood edges", detail=f"{total_seed_rows:,} seed rows across {total_chunks:,} chunks")
    with edges_csv.open(edge_file_mode, encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=edge_fieldnames)
        if not append_existing_output:
            writer.writeheader()
        with rich_tqdm(cfg, total=total_seed_rows, desc="Neighborhood rows", unit="seed") as progress:
            for chunk_idx, query_block in enumerate(_iter_query_blocks(seed_indices, query_chunk_size)):
                chunk_meta = _chunk_meta_path(chunks_dir, chunk_idx)
                if chunk_idx < len(completed_chunk_metas):
                    meta = completed_chunk_metas[chunk_idx]
                    queries_processed += int(meta["queries_processed"])
                    output_rows += int(meta["output_rows"])
                    filtered_tag_delta_rows += int(meta["filtered_tag_delta_rows"])
                    chunks_skipped_existing += 1
                    progress.update(len(query_block))
                    if queries_processed % every_n == 0:
                        _log(cfg, f"Seed rows processed: {queries_processed:,}")
                    continue

                candidate_indices, candidate_audio_scores = _topk_audio_neighbors(
                    query_block,
                    audio_matrix,
                    track_ids,
                    splits,
                    file_paths,
                    candidate_pools_by_split,
                    audio_search_state,
                    audio_backend,
                    cfg,
                )
                meta = _write_chunk_neighbors(
                    writer=writer,
                    chunk_meta=chunk_meta,
                    chunk_idx=chunk_idx,
                    query_block=query_block,
                    candidate_indices=candidate_indices,
                    candidate_audio_scores=candidate_audio_scores,
                    rows=rows,
                    text_matrix=text_matrix,
                    tag_sets=tag_sets,
                    cfg=cfg,
                )
                out_f.flush()
                queries_processed += int(meta["queries_processed"])
                output_rows += int(meta["output_rows"])
                filtered_tag_delta_rows += int(meta["filtered_tag_delta_rows"])
                chunks_processed += 1
                progress.update(len(query_block))
                if queries_processed % every_n == 0:
                    _log(cfg, f"Seed rows processed: {queries_processed:,}")

    report = {
        "stage": "neighborhood",
        "input": {
            "input_manifest_csv": str(cfg.stage.io.input_manifest_csv),
            "input_lookup_manifest_csv": str(cfg.stage.io.input_lookup_manifest_csv),
        },
        "counts": {
            "indexed_rows": len(rows),
            "seed_rows": total_seed_rows,
            "queries_processed": queries_processed,
            "output_rows": output_rows,
            "filtered_tag_delta_rows": filtered_tag_delta_rows,
            "audio_embedding_dim": int(audio_matrix.shape[1]) if audio_matrix.ndim == 2 else 0,
            "text_embedding_dim": int(text_matrix.shape[1]) if text_matrix is not None and text_matrix.ndim == 2 else 0,
            "audio_top_k": int(cfg.stage.retrieval.audio_top_k),
            "retain_top_k": int(cfg.stage.retrieval.retain_top_k),
            "chunks_total": total_chunks,
            "chunks_processed": chunks_processed,
            "chunks_skipped_existing": chunks_skipped_existing,
            "audio_backend": audio_backend,
            "use_text_rerank": use_text_rerank,
        },
        "config": {
            "retrieval": _cfg_section_to_plain(cfg.stage.retrieval),
            "runtime": _cfg_section_to_plain(cfg.stage.runtime),
            "behavior": _cfg_section_to_plain(cfg.stage.behavior),
        },
        "outputs": {
            "output_nodes_csv": str(nodes_csv),
            "output_edges_csv": str(edges_csv),
            "edge_chunks_dir": str(chunks_dir),
            "report": str(report_path),
        },
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)

    tracker.step("Write report", detail=report_path.name)
    tracker.finish(
        f"rows={output_rows:,}, chunks_processed={chunks_processed:,}, chunks_reused={chunks_skipped_existing:,}"
    )
    _log(
        cfg,
        f"Neighborhood retrieval complete. Seeds={total_seed_rows:,}, "
        f"rows={output_rows:,}, chunks_processed={chunks_processed:,}, "
        f"chunks_skipped_existing={chunks_skipped_existing:,}, "
        f"audio_top_k={int(cfg.stage.retrieval.audio_top_k)}",
    )
    return report


def _main_impl(cfg: DictConfig) -> None:
    report = run_neighborhood(cfg)
    print(json.dumps({"status": "ok", "stage": "neighborhood", "outputs": report["outputs"]}, indent=2))


def main() -> None:
    import hydra

    @hydra.main(version_base=None, config_path=CONF_DIR, config_name="config")
    def _wrapped(cfg: DictConfig) -> None:
        _main_impl(cfg)

    _wrapped()


if __name__ == "__main__":
    main()
