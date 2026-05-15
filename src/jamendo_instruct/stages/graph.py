from __future__ import annotations

import csv
import json
import math
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Sequence, TextIO

import numpy as np
from jamendo_instruct.progress import StageTracker, rich_tqdm

if TYPE_CHECKING:
    from omegaconf import DictConfig
else:
    DictConfig = Any

CONF_DIR = str(Path(__file__).resolve().parents[3] / "conf")


def _log(cfg: DictConfig, message: str) -> None:
    if bool(cfg.stage.progress.enabled):
        print(f"[graph] {message}", flush=True)


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


def _iter_csv_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


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


def _structured_index(path: Path, *, include_captions: bool, include_lyrics: bool) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in _iter_csv_rows(path):
        clip_id = str(row.get("clip_id", "") or "").strip()
        if not clip_id:
            continue
        tags = sorted(set(_parse_json_list(row.get("normalized_tags_json", ""))))
        cached: Dict[str, Any] = {
            "clip_id": clip_id,
            "split": str(row.get("split", "") or ""),
            "tags": tags,
            "vocals": str(row.get("vocals", "") or ""),
            "speed": str(row.get("speed", "") or ""),
        }
        if include_captions:
            cached["caption"] = str(row.get("normalized_caption", "") or row.get("caption", "") or "")
        if include_lyrics:
            cached["lyrics"] = str(row.get("normalized_lyrics", "") or row.get("lyrics", "") or "")
        out[clip_id] = cached
    return out


def _node_index(path: Path) -> Dict[int, Dict[str, str]]:
    out: Dict[int, Dict[str, str]] = {}
    for row in _iter_csv_rows(path):
        out[int(row["node_idx"])] = row
    return out


def _structured_delta(
    source_row: Dict[str, Any],
    target_row: Dict[str, Any],
    *,
    include_captions: bool,
    include_lyrics: bool,
) -> Dict[str, Any]:
    source_tags = set(source_row["tags"])
    target_tags = set(target_row["tags"])
    delta = {
        "tags_added": sorted(target_tags - source_tags),
        "tags_removed": sorted(source_tags - target_tags),
        "tags_preserved": sorted(source_tags & target_tags),
        "source_vocals": str(source_row.get("vocals", "") or ""),
        "target_vocals": str(target_row.get("vocals", "") or ""),
        "source_speed": str(source_row.get("speed", "") or ""),
        "target_speed": str(target_row.get("speed", "") or ""),
    }
    if include_captions:
        delta["source_caption"] = str(source_row.get("caption", "") or "")
        delta["target_caption"] = str(target_row.get("caption", "") or "")
    if include_lyrics:
        delta["source_lyrics"] = str(source_row.get("lyrics", "") or "")
        delta["target_lyrics"] = str(target_row.get("lyrics", "") or "")
    return delta


def _tag_delta_size(delta: Dict[str, Any]) -> int:
    return len(delta["tags_added"]) + len(delta["tags_removed"])


def _hardness(tag_delta_size: int) -> str:
    if tag_delta_size <= 1:
        return "easy"
    if tag_delta_size <= 3:
        return "medium"
    return "hard"


def _clip01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _transition_score(cfg: DictConfig, *, audio_similarity: float, text_similarity: float, tag_delta_size: int) -> float:
    max_changed = max(1, int(cfg.stage.filters.max_changed_tags))
    audio_component = _clip01(audio_similarity)
    text_component = _clip01(text_similarity)
    tag_component = _clip01(1.0 - (float(tag_delta_size) / float(max_changed)))
    audio_weight = float(cfg.stage.scoring.audio_weight)
    text_weight = float(cfg.stage.scoring.text_weight)
    tag_weight = float(cfg.stage.scoring.tag_weight)
    total = audio_weight + text_weight + tag_weight
    if total <= 0:
        raise ValueError("Stage graph scoring weights must sum to a positive value.")
    score = (
        (audio_weight * audio_component)
        + (text_weight * text_component)
        + (tag_weight * tag_component)
    ) / total
    return _clip01(score)


def _fieldnames(cfg: DictConfig) -> List[str]:
    compact = bool(getattr(cfg.stage.output, "compact", True))
    include_delta = bool(getattr(cfg.stage.output, "include_structured_delta_json", False))
    base = [
        "source_node_idx",
        "target_node_idx",
        "audio_rank",
        "rerank_rank",
        "audio_similarity",
        "text_similarity",
        "tag_delta_size",
        "tags_preserved_count",
        "transition_score",
        "transition_cost",
        "hardness",
    ]
    if not compact:
        base[2:2] = [
            "source_clip_id",
            "target_clip_id",
            "source_track_id",
            "target_track_id",
            "source_split",
            "target_split",
            "source_tags_json",
            "target_tags_json",
        ]
    if include_delta:
        base.append("structured_delta_json")
    return base


def _chunk_csv_path(chunks_dir: Path, chunk_idx: int) -> Path:
    return chunks_dir / f"graph_chunk_{chunk_idx:06d}.csv"


def _chunk_meta_path(chunks_dir: Path, chunk_idx: int) -> Path:
    return chunks_dir / f"graph_chunk_{chunk_idx:06d}.json"


def _load_chunk_meta(path: Path) -> Dict[str, int] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): int(v) for k, v in data.items()}


def _iter_edge_chunks(path: Path, chunk_size: int) -> Iterable[List[Dict[str, str]]]:
    chunk: List[Dict[str, str]] = []
    for row in _iter_csv_rows(path):
        chunk.append(row)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _process_edge_chunk(
    *,
    cfg: DictConfig,
    chunk_rows: Sequence[Dict[str, str]],
    chunk_csv: Path,
    chunk_meta: Path,
    structured_by_clip: Dict[str, Dict[str, Any]],
    nodes_by_idx: Dict[int, Dict[str, str]],
    fieldnames: Sequence[str],
) -> Dict[str, int]:
    include_delta = bool(getattr(cfg.stage.output, "include_structured_delta_json", False))
    include_captions = bool(getattr(cfg.stage.output, "include_caption_text_in_structured_delta", False))
    include_lyrics = bool(getattr(cfg.stage.output, "include_lyrics_text_in_structured_delta", True))
    compact = bool(getattr(cfg.stage.output, "compact", True))
    max_changed = int(cfg.stage.filters.max_changed_tags)

    counts = {
        "input_edges": len(chunk_rows),
        "output_edges": 0,
        "edges_filtered_max_changed_tags": 0,
        "edges_filtered_cross_split": 0,
        "missing_node_refs": 0,
        "missing_structured_refs": 0,
        "edges_with_source_lyrics": 0,
        "edges_with_target_lyrics": 0,
        "edges_with_lyric_change": 0,
    }

    tmp_csv = chunk_csv.with_suffix(".tmp")
    with tmp_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for edge in chunk_rows:
            source_idx = int(edge["source_node_idx"])
            target_idx = int(edge["target_node_idx"])
            source_node = nodes_by_idx.get(source_idx)
            target_node = nodes_by_idx.get(target_idx)
            if source_node is None or target_node is None:
                counts["missing_node_refs"] += 1
                continue

            source_clip_id = str(source_node.get("clip_id", "") or "")
            target_clip_id = str(target_node.get("clip_id", "") or "")
            source_struct = structured_by_clip.get(source_clip_id)
            target_struct = structured_by_clip.get(target_clip_id)
            if source_struct is None or target_struct is None:
                counts["missing_structured_refs"] += 1
                continue

            source_split = str(source_struct.get("split", "") or "")
            target_split = str(target_struct.get("split", "") or "")
            if source_split != target_split:
                counts["edges_filtered_cross_split"] += 1
                continue

            delta = _structured_delta(
                source_struct,
                target_struct,
                include_captions=include_captions,
                include_lyrics=include_lyrics,
            )
            if include_lyrics:
                source_lyrics = str(delta.get("source_lyrics", "") or "").strip()
                target_lyrics = str(delta.get("target_lyrics", "") or "").strip()
                if source_lyrics:
                    counts["edges_with_source_lyrics"] += 1
                if target_lyrics:
                    counts["edges_with_target_lyrics"] += 1
                if source_lyrics != target_lyrics:
                    counts["edges_with_lyric_change"] += 1
            tag_delta_size = _tag_delta_size(delta)
            if tag_delta_size > max_changed:
                counts["edges_filtered_max_changed_tags"] += 1
                continue

            audio_similarity = float(edge["audio_similarity"])
            text_similarity = float(edge["text_similarity"])
            transition_score = _transition_score(
                cfg,
                audio_similarity=audio_similarity,
                text_similarity=text_similarity,
                tag_delta_size=tag_delta_size,
            )
            row = {
                "source_node_idx": source_idx,
                "target_node_idx": target_idx,
                "audio_rank": str(edge["audio_rank"]),
                "rerank_rank": str(edge["rerank_rank"]),
                "audio_similarity": f"{audio_similarity:.8f}",
                "text_similarity": f"{text_similarity:.8f}",
                "tag_delta_size": tag_delta_size,
                "tags_preserved_count": len(delta["tags_preserved"]),
                "transition_score": f"{transition_score:.8f}",
                "transition_cost": f"{1.0 - transition_score:.8f}",
                "hardness": _hardness(tag_delta_size),
            }
            if not compact:
                row.update(
                    {
                        "source_clip_id": source_clip_id,
                        "target_clip_id": target_clip_id,
                        "source_track_id": str(source_node.get("track_id", "") or ""),
                        "target_track_id": str(target_node.get("track_id", "") or ""),
                        "source_split": source_split,
                        "target_split": target_split,
                        "source_tags_json": json.dumps(source_struct["tags"], ensure_ascii=True),
                        "target_tags_json": json.dumps(target_struct["tags"], ensure_ascii=True),
                    }
                )
            if include_delta:
                row["structured_delta_json"] = json.dumps(delta, ensure_ascii=True)
            writer.writerow(row)
            counts["output_edges"] += 1

    tmp_csv.replace(chunk_csv)
    chunk_meta.write_text(json.dumps(counts, indent=2), encoding="utf-8")
    return counts


def _iter_chunk_rows(chunks_dir: Path, total_chunks: int) -> Iterable[Dict[str, str]]:
    for chunk_idx in range(total_chunks):
        chunk_csv = _chunk_csv_path(chunks_dir, chunk_idx)
        if not chunk_csv.exists():
            raise FileNotFoundError(f"Missing graph chunk CSV: {chunk_csv}")
        with chunk_csv.open("r", encoding="utf-8", newline="") as in_f:
            reader = csv.DictReader(in_f)
            for row in reader:
                yield row


class _AdjacencyShardWriter:
    def __init__(
        self,
        *,
        shards_dir: Path,
        metadata_path: Path,
        fieldnames: Sequence[str],
        shard_count: int,
        max_open_writers: int = 32,
    ) -> None:
        self.shards_dir = shards_dir
        self.metadata_path = metadata_path
        self.fieldnames = list(fieldnames)
        self.shard_count = max(1, int(shard_count))
        self.max_open_writers = max(1, int(max_open_writers))
        self._writers: "OrderedDict[int, tuple[TextIO, csv.DictWriter]]" = OrderedDict()
        self.shard_counts: Dict[int, int] = {}

    def _shard_path(self, shard_idx: int) -> Path:
        return self.shards_dir / f"adjacency_shard_{shard_idx:04d}.csv"

    def _open_writer(self, shard_idx: int) -> csv.DictWriter:
        existing = self._writers.get(shard_idx)
        if existing is not None:
            handle, writer = existing
            self._writers.move_to_end(shard_idx)
            return writer

        path = self._shard_path(shard_idx)
        is_new = not path.exists()
        handle = path.open("a", encoding="utf-8", newline="")
        writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
        if is_new:
            writer.writeheader()
        self._writers[shard_idx] = (handle, writer)
        self._writers.move_to_end(shard_idx)

        while len(self._writers) > self.max_open_writers:
            _, (old_handle, _) = self._writers.popitem(last=False)
            old_handle.close()
        return writer

    def write_row(self, row: Dict[str, Any]) -> None:
        source_idx = int(row["source_node_idx"])
        shard_idx = source_idx % self.shard_count
        writer = self._open_writer(shard_idx)
        writer.writerow(row)
        self.shard_counts[shard_idx] = self.shard_counts.get(shard_idx, 0) + 1

    def close(self) -> None:
        while self._writers:
            _, (handle, _) = self._writers.popitem(last=False)
            handle.close()

    def write_metadata(self) -> None:
        payload = {
            "format": "modulo_source_node_idx_csv_shards_v1",
            "fieldnames": list(self.fieldnames),
            "shard_count": self.shard_count,
            "path_pattern": "adjacency_shard_{shard_idx:04d}.csv",
            "row_counts_by_shard": {
                str(idx): int(count) for idx, count in sorted(self.shard_counts.items())
            },
        }
        self.metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_adjacency_memmap_from_chunks(
    *,
    cfg: DictConfig,
    chunks_dir: Path,
    total_chunks: int,
    memmap_dir: Path,
    node_count: int,
    total_output_edges: int,
) -> Dict[str, Any]:
    memmap_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = memmap_dir / "metadata.json"
    offsets_path = memmap_dir / "offsets.npy"
    targets_path = memmap_dir / "targets.npy"
    scores_path = memmap_dir / "scores.npy"
    costs_path = memmap_dir / "costs.npy"
    valid_seed_nodes_path = memmap_dir / "valid_seed_nodes.npy"

    degrees = np.zeros(node_count, dtype=np.int64)
    scan_every = 5_000_000
    scanned_edges = 0
    _log(cfg, f"Adjacency pass 1/2: counting degrees across {total_output_edges:,} retained edges")
    for row in _iter_chunk_rows(chunks_dir, total_chunks):
        source_idx = int(row["source_node_idx"])
        degrees[source_idx] += 1
        scanned_edges += 1
        if scanned_edges % scan_every == 0:
            _log(cfg, f"Adjacency pass 1/2: counted {scanned_edges:,}/{total_output_edges:,} edges")

    offsets = np.empty(node_count + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(degrees, out=offsets[1:])
    if int(offsets[-1]) != int(total_output_edges):
        raise ValueError(
            f"Adjacency offset total mismatch: offsets[-1]={int(offsets[-1])} "
            f"but expected total_output_edges={int(total_output_edges)}"
        )

    valid_seed_nodes = np.flatnonzero(degrees > 0).astype(np.int32, copy=False)
    np.save(offsets_path, offsets)
    np.save(valid_seed_nodes_path, valid_seed_nodes)

    targets = np.lib.format.open_memmap(
        targets_path,
        mode="w+",
        dtype=np.int32,
        shape=(total_output_edges,),
    )
    scores = np.lib.format.open_memmap(
        scores_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_output_edges,),
    )
    costs = np.lib.format.open_memmap(
        costs_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_output_edges,),
    )
    cursors = offsets[:-1].copy()

    written_edges = 0
    _log(cfg, f"Adjacency pass 2/2: writing memmaps for {total_output_edges:,} retained edges")
    for row in _iter_chunk_rows(chunks_dir, total_chunks):
        source_idx = int(row["source_node_idx"])
        write_idx = int(cursors[source_idx])
        targets[write_idx] = int(row["target_node_idx"])
        scores[write_idx] = np.float32(float(row["transition_score"]))
        costs[write_idx] = np.float32(float(row["transition_cost"]))
        cursors[source_idx] += 1
        written_edges += 1
        if written_edges % scan_every == 0:
            _log(cfg, f"Adjacency pass 2/2: wrote {written_edges:,}/{total_output_edges:,} edges")

    targets.flush()
    scores.flush()
    costs.flush()

    metadata = {
        "format": "csr_memmap_v1",
        "node_count": int(node_count),
        "edge_count": int(total_output_edges),
        "offsets_file": offsets_path.name,
        "targets_file": targets_path.name,
        "scores_file": scores_path.name,
        "costs_file": costs_path.name,
        "valid_seed_nodes_file": valid_seed_nodes_path.name,
        "offsets_dtype": "int64",
        "targets_dtype": "int32",
        "scores_dtype": "float32",
        "costs_dtype": "float32",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "memmap_dir": str(memmap_dir),
        "metadata_path": str(metadata_path),
        "valid_seed_nodes": int(valid_seed_nodes.shape[0]),
    }


def run_graph(cfg: DictConfig) -> Dict[str, object]:
    structured_path = Path(str(cfg.stage.io.input_manifest_csv))
    nodes_path = Path(str(cfg.stage.io.input_nodes_csv))
    edges_path = Path(str(cfg.stage.io.input_edges_csv))
    out_dir = Path(str(cfg.stage.io.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    adjacency_memmap_dir = out_dir / str(cfg.stage.io.output_adjacency_memmap_dir)
    adjacency_shards_dir = out_dir / str(cfg.stage.io.output_adjacency_shards_dir)
    adjacency_metadata_path = out_dir / str(cfg.stage.io.output_adjacency_metadata_json)
    report_path = out_dir / str(cfg.stage.io.report_file)
    chunks_dir = out_dir / "graph_chunks"
    tracker = StageTracker(
        cfg,
        "graph",
        title="Build Transition Graph",
        subtitle=f"edges={edges_path}",
        total_steps=5,
    )

    for path in (structured_path, nodes_path, edges_path):
        if not path.exists():
            raise FileNotFoundError(f"Required graph input not found: {path}")

    include_delta = bool(getattr(cfg.stage.output, "include_structured_delta_json", False))
    include_captions = bool(getattr(cfg.stage.output, "include_caption_text_in_structured_delta", False))
    include_lyrics = bool(getattr(cfg.stage.output, "include_lyrics_text_in_structured_delta", True))
    tracker.step("Load node and clip indexes", detail=f"manifest={structured_path.name}, nodes={nodes_path.name}")
    structured_by_clip = _structured_index(
        structured_path,
        include_captions=include_delta and include_captions,
        include_lyrics=include_delta and include_lyrics,
    )
    nodes_by_idx = _node_index(nodes_path)
    fieldnames = _fieldnames(cfg)
    input_edge_chunk_size = max(1, int(cfg.stage.runtime.input_edge_chunk_size))
    overwrite_existing = bool(getattr(cfg.stage.behavior, "overwrite_existing", False))
    cleanup_chunks_after_merge = bool(getattr(cfg.stage.behavior, "cleanup_chunks_after_merge", True))
    adjacency_shard_count = max(1, int(getattr(cfg.stage.output, "adjacency_shard_count", 4096)))
    every_n = max(1, int(cfg.stage.progress.every_n_rows))
    memmap_metadata_path = adjacency_memmap_dir / "metadata.json"

    if memmap_metadata_path.exists() and adjacency_memmap_dir.exists() and not overwrite_existing:
        _log(cfg, f"Skipping graph stage because adjacency memmap metadata already exists: {memmap_metadata_path}")
        tracker.finish(f"skipped existing output {memmap_metadata_path}")
        if report_path.exists():
            with report_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "stage": "graph",
            "input": {
                "input_manifest_csv": str(structured_path),
                "input_nodes_csv": str(nodes_path),
                "input_edges_csv": str(edges_path),
            },
            "counts": {},
            "config": {
                "filters": _cfg_section_to_plain(cfg.stage.filters),
                "scoring": _cfg_section_to_plain(cfg.stage.scoring),
                "runtime": _cfg_section_to_plain(cfg.stage.runtime),
                "behavior": _cfg_section_to_plain(cfg.stage.behavior),
                "output": _cfg_section_to_plain(cfg.stage.output),
            },
            "outputs": {
                "output_adjacency_memmap_dir": str(adjacency_memmap_dir),
                "output_adjacency_memmap_metadata_json": str(memmap_metadata_path),
                "output_adjacency_shards_dir": str(adjacency_shards_dir) if adjacency_shards_dir.exists() else None,
                "output_adjacency_metadata_json": str(adjacency_metadata_path) if adjacency_metadata_path.exists() else None,
                "graph_chunks_dir": None,
                "report": str(report_path),
            },
        }

    tracker.step("Prepare chunk workspace", detail=str(chunks_dir))
    if chunks_dir.exists():
        shutil.rmtree(chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    if adjacency_memmap_dir.exists():
        shutil.rmtree(adjacency_memmap_dir)
    if adjacency_shards_dir.exists():
        shutil.rmtree(adjacency_shards_dir)
    if adjacency_metadata_path.exists():
        adjacency_metadata_path.unlink()
    total_input_edges = sum(1 for _ in _iter_csv_rows(edges_path))
    total_chunks = int(math.ceil(total_input_edges / input_edge_chunk_size))

    counts = {
        "input_edges": 0,
        "output_edges": 0,
        "edges_filtered_max_changed_tags": 0,
        "edges_filtered_cross_split": 0,
        "missing_node_refs": 0,
        "missing_structured_refs": 0,
        "edges_with_source_lyrics": 0,
        "edges_with_target_lyrics": 0,
        "edges_with_lyric_change": 0,
        "chunks_total": total_chunks,
        "chunks_processed": 0,
    }

    tracker.step("Process neighborhood edges", detail=f"{total_input_edges:,} edges across {total_chunks:,} chunks")
    with rich_tqdm(cfg, total=total_input_edges, desc="Graph edges", unit="edge") as progress:
        for chunk_idx, chunk_rows in enumerate(_iter_edge_chunks(edges_path, input_edge_chunk_size)):
            chunk_csv = _chunk_csv_path(chunks_dir, chunk_idx)
            chunk_meta = _chunk_meta_path(chunks_dir, chunk_idx)

            meta = _process_edge_chunk(
                cfg=cfg,
                chunk_rows=chunk_rows,
                chunk_csv=chunk_csv,
                chunk_meta=chunk_meta,
                structured_by_clip=structured_by_clip,
                nodes_by_idx=nodes_by_idx,
                fieldnames=fieldnames,
            )
            for key in (
                "input_edges",
                "output_edges",
                "edges_filtered_max_changed_tags",
                "edges_filtered_cross_split",
                "missing_node_refs",
                "missing_structured_refs",
                "edges_with_source_lyrics",
                "edges_with_target_lyrics",
                "edges_with_lyric_change",
            ):
                counts[key] += int(meta.get(key, 0))
            counts["chunks_processed"] += 1
            progress.update(len(chunk_rows))
            if counts["input_edges"] % every_n == 0:
                _log(cfg, f"Edges processed: {counts['input_edges']:,}")

    tracker.step("Build adjacency memmap", detail=adjacency_memmap_dir.name)
    adjacency_memmap_info = _build_adjacency_memmap_from_chunks(
        cfg=cfg,
        chunks_dir=chunks_dir,
        total_chunks=total_chunks,
        memmap_dir=adjacency_memmap_dir,
        node_count=(max(nodes_by_idx.keys()) + 1) if nodes_by_idx else 0,
        total_output_edges=int(counts["output_edges"]),
    )

    graph_chunks_dir_output: str | None = str(chunks_dir)
    if cleanup_chunks_after_merge and chunks_dir.exists():
        shutil.rmtree(chunks_dir)
        graph_chunks_dir_output = None
        _log(cfg, f"Removed temporary graph chunk directory: {chunks_dir}")

    report = {
        "stage": "graph",
        "input": {
            "input_manifest_csv": str(structured_path),
            "input_nodes_csv": str(nodes_path),
            "input_edges_csv": str(edges_path),
        },
        "counts": counts,
        "config": {
            "filters": _cfg_section_to_plain(cfg.stage.filters),
            "scoring": _cfg_section_to_plain(cfg.stage.scoring),
            "runtime": _cfg_section_to_plain(cfg.stage.runtime),
            "behavior": _cfg_section_to_plain(cfg.stage.behavior),
            "output": _cfg_section_to_plain(cfg.stage.output),
        },
        "outputs": {
            "output_adjacency_memmap_dir": str(adjacency_memmap_dir),
            "output_adjacency_memmap_metadata_json": str(memmap_metadata_path),
            "output_adjacency_shards_dir": None,
            "output_adjacency_metadata_json": None,
            "graph_chunks_dir": graph_chunks_dir_output,
            "report": str(report_path),
        },
    }
    report["counts"]["valid_seed_nodes"] = int(adjacency_memmap_info["valid_seed_nodes"])

    tracker.step("Write report", detail=report_path.name)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)

    tracker.finish(f"retained {counts['output_edges']:,}/{counts['input_edges']:,} edges")
    _log(
        cfg,
        f"Graph construction complete. Retained {counts['output_edges']:,} / {counts['input_edges']:,} edges "
        f"(chunks_processed={counts['chunks_processed']:,})",
    )
    return report


def _main_impl(cfg: DictConfig) -> None:
    report = run_graph(cfg)
    print(json.dumps({"status": "ok", "stage": "graph", "outputs": report["outputs"]}, indent=2))


def main() -> None:
    import hydra

    @hydra.main(version_base=None, config_path=CONF_DIR, config_name="config")
    def _wrapped(cfg: DictConfig) -> None:
        _main_impl(cfg)

    _wrapped()


if __name__ == "__main__":
    main()
