from __future__ import annotations

import csv
import json
import math
import random
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Sequence

import numpy as np
from jamendo_instruct.progress import StageTracker, rich_tqdm

if TYPE_CHECKING:
    from omegaconf import DictConfig
else:
    DictConfig = Any

CONF_DIR = str(Path(__file__).resolve().parents[3] / "conf")


def _log(cfg: DictConfig, message: str) -> None:
    if bool(cfg.stage.progress.enabled):
        print(f"[chains] {message}", flush=True)


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


def _parse_json_obj(raw: str) -> Dict[str, Any]:
    value = str(raw or "").strip()
    if not value:
        return {}
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _structured_index(path: Path) -> Dict[str, Dict[str, str]]:
    rows = _read_csv_rows(path)
    out: Dict[str, Dict[str, str]] = {}
    for row in rows:
        clip_id = str(row.get("clip_id", "") or "").strip()
        if clip_id:
            out[clip_id] = row
    return out


def _nodes_index(path: Path) -> Dict[int, Dict[str, str]]:
    rows = _read_csv_rows(path)
    out: Dict[int, Dict[str, str]] = {}
    for row in rows:
        out[int(row["node_idx"])] = row
    return out


def _parse_edge_row(row: Dict[str, Any]) -> Dict[str, Any]:
    parsed = dict(row)
    parsed["source_node_idx"] = int(row["source_node_idx"])
    parsed["target_node_idx"] = int(row["target_node_idx"])
    parsed["transition_score"] = float(row["transition_score"])
    parsed["transition_cost"] = float(row["transition_cost"])
    parsed["audio_similarity"] = float(row["audio_similarity"])
    parsed["text_similarity"] = float(row["text_similarity"])
    parsed["tag_delta_size"] = int(row["tag_delta_size"])
    parsed["tags_preserved_count"] = int(row["tags_preserved_count"])
    parsed["structured_delta"] = _parse_json_obj(row.get("structured_delta_json", ""))
    return parsed


def _hardness(tag_delta_size: int) -> str:
    if tag_delta_size <= 1:
        return "easy"
    if tag_delta_size <= 3:
        return "medium"
    return "hard"


class _TransitionShardStore:
    def __init__(
        self,
        *,
        cfg: DictConfig,
        shards_dir: Path,
        metadata_path: Path,
        cache_size: int,
    ) -> None:
        self.cfg = cfg
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        self.shards_dir = shards_dir
        self.metadata_path = metadata_path
        self.metadata = metadata
        self.shard_count = max(1, int(metadata["shard_count"]))
        self.path_pattern = str(metadata.get("path_pattern", "adjacency_shard_{shard_idx:04d}.csv"))
        self.cache_size = max(1, int(cache_size))
        self._cache: "OrderedDict[int, Dict[int, List[Dict[str, Any]]]]" = OrderedDict()
        self._shard_load_count = 0

    def _shard_path(self, shard_idx: int) -> Path:
        return self.shards_dir / self.path_pattern.format(shard_idx=shard_idx)

    def _load_shard(self, shard_idx: int) -> Dict[int, List[Dict[str, Any]]]:
        cached = self._cache.get(shard_idx)
        if cached is not None:
            self._cache.move_to_end(shard_idx)
            return cached

        shard_path = self._shard_path(shard_idx)
        outgoing: Dict[int, List[Dict[str, Any]]] = {}
        if shard_path.exists():
            for row in _iter_csv_rows(shard_path):
                parsed = _parse_edge_row(row)
                source = int(parsed["source_node_idx"])
                outgoing.setdefault(source, []).append(parsed)
        self._shard_load_count += 1
        if self._shard_load_count <= 5 or self._shard_load_count % 100 == 0:
            edge_count = sum(len(edges) for edges in outgoing.values())
            _log(
                self.cfg,
                f"Loaded shard {shard_idx:,}/{self.shard_count - 1:,} "
                f"(load_count={self._shard_load_count:,}, sources={len(outgoing):,}, edges={edge_count:,})",
            )
        self._cache[shard_idx] = outgoing
        self._cache.move_to_end(shard_idx)

        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return outgoing

    def get_edges(self, source_node_idx: int) -> List[Dict[str, Any]]:
        shard_idx = int(source_node_idx) % self.shard_count
        outgoing = self._load_shard(shard_idx)
        return outgoing.get(int(source_node_idx), [])

    def valid_seed_indices(self) -> List[int]:
        seeds = set()
        log_every = max(1, self.shard_count // 20)
        for shard_idx in range(self.shard_count):
            shard_path = self._shard_path(shard_idx)
            if not shard_path.exists():
                if shard_idx == 0 or (shard_idx + 1) % log_every == 0 or shard_idx + 1 == self.shard_count:
                    _log(
                        self.cfg,
                        f"Scanning shards for valid seeds: {shard_idx + 1:,}/{self.shard_count:,} "
                        f"(current_valid_seeds={len(seeds):,})",
                    )
                continue
            for row in _iter_csv_rows(shard_path):
                source = int(row["source_node_idx"])
                seeds.add(source)
            if shard_idx == 0 or (shard_idx + 1) % log_every == 0 or shard_idx + 1 == self.shard_count:
                _log(
                    self.cfg,
                    f"Scanning shards for valid seeds: {shard_idx + 1:,}/{self.shard_count:,} "
                    f"(current_valid_seeds={len(seeds):,})",
                )
        return sorted(seeds)


class _TransitionMemmapStore:
    def __init__(
        self,
        *,
        cfg: DictConfig,
        memmap_dir: Path,
    ) -> None:
        self.cfg = cfg
        self.memmap_dir = memmap_dir
        metadata_path = memmap_dir / "metadata.json"
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        self.metadata = metadata
        self.node_count = int(metadata["node_count"])
        self.edge_count = int(metadata["edge_count"])
        self.offsets = np.load(memmap_dir / str(metadata["offsets_file"]), mmap_mode="r")
        self.targets = np.load(memmap_dir / str(metadata["targets_file"]), mmap_mode="r")
        self.scores = np.load(memmap_dir / str(metadata["scores_file"]), mmap_mode="r")
        self.costs = np.load(memmap_dir / str(metadata["costs_file"]), mmap_mode="r")
        self.valid_seed_nodes = np.load(memmap_dir / str(metadata["valid_seed_nodes_file"]), mmap_mode="r")
        self._slice_log_count = 0

    def valid_seed_indices(self) -> List[int]:
        return [int(x) for x in self.valid_seed_nodes.tolist()]

    def get_edge_block(self, source_node_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        source_node_idx = int(source_node_idx)
        if source_node_idx < 0 or source_node_idx + 1 >= self.offsets.shape[0]:
            return (
                self.targets[0:0],
                self.scores[0:0],
                self.costs[0:0],
            )
        start = int(self.offsets[source_node_idx])
        end = int(self.offsets[source_node_idx + 1])
        self._slice_log_count += 1
        if self._slice_log_count <= 5 or self._slice_log_count % 500_000 == 0:
            _log(
                self.cfg,
                f"Memmap slice fetches: {self._slice_log_count:,} "
                f"(source_node_idx={source_node_idx:,}, out_degree={end - start:,})",
            )
        return (
            self.targets[start:end],
            self.scores[start:end],
            self.costs[start:end],
        )


def _tag_set(row: Dict[str, str]) -> List[str]:
    tags = _parse_json_list(row.get("normalized_tags_json", ""))
    if tags:
        return sorted(set(tags))
    raw = str(row.get("tags", "") or "")
    return sorted({part.strip() for part in raw.split(",") if part.strip()})


def _metadata_snapshot(row: Dict[str, str]) -> Dict[str, str]:
    keep_keys = [
        "track_id",
        "title",
        "artist_id",
        "artist_name",
        "release_date",
        "vocals",
        "speed",
        "lyrics_status",
        "start_time",
        "end_time",
        "split",
    ]
    return {key: str(row.get(key, "") or "") for key in keep_keys}


def _structured_delta(source_row: Dict[str, str], target_row: Dict[str, str]) -> Dict[str, Any]:
    source_tags = set(_tag_set(source_row))
    target_tags = set(_tag_set(target_row))
    return {
        "tags_added": sorted(target_tags - source_tags),
        "tags_removed": sorted(source_tags - target_tags),
        "tags_preserved": sorted(source_tags & target_tags),
        "source_vocals": str(source_row.get("vocals", "") or ""),
        "target_vocals": str(target_row.get("vocals", "") or ""),
        "source_speed": str(source_row.get("speed", "") or ""),
        "target_speed": str(target_row.get("speed", "") or ""),
        "source_caption": str(source_row.get("normalized_caption", "") or source_row.get("caption", "") or ""),
        "target_caption": str(target_row.get("normalized_caption", "") or target_row.get("caption", "") or ""),
        "source_lyrics": str(source_row.get("normalized_lyrics", "") or source_row.get("lyrics", "") or ""),
        "target_lyrics": str(target_row.get("normalized_lyrics", "") or target_row.get("lyrics", "") or ""),
    }


def _sample_target_chain_length(cfg: DictConfig, rng: random.Random) -> int:
    min_len = max(1, int(cfg.stage.behavior.min_chain_length))
    max_len = max(min_len, int(cfg.stage.behavior.max_chain_length))
    mean = float(cfg.stage.sampling.length_lognormal_mean)
    sigma = float(cfg.stage.sampling.length_lognormal_sigma)
    sampled = int(round(math.exp(rng.normalvariate(mean, sigma))))
    return max(min_len, min(max_len, sampled))


def _choose_edge(rng: random.Random, edges: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    weights = [max(0.0, float(edge["transition_score"])) for edge in edges]
    total = sum(weights)
    if total <= 0:
        return rng.choice(list(edges))
    threshold = rng.random() * total
    running = 0.0
    for edge, weight in zip(edges, weights):
        running += weight
        if running >= threshold:
            return edge
    return edges[-1]


def _choose_edge_index(
    rng: random.Random,
    *,
    scores: np.ndarray,
    candidate_positions: Sequence[int],
) -> int:
    if not candidate_positions:
        raise ValueError("candidate_positions must not be empty")
    weights = [max(0.0, float(scores[pos])) for pos in candidate_positions]
    total = sum(weights)
    if total <= 0.0:
        return int(rng.choice(list(candidate_positions)))
    threshold = rng.random() * total
    running = 0.0
    for pos, weight in zip(candidate_positions, weights):
        running += weight
        if running >= threshold:
            return int(pos)
    return int(candidate_positions[-1])


def _update_accumulated_state(
    current_state: Dict[str, Any] | None,
    *,
    seed_row: Dict[str, str] | None = None,
    source_row: Dict[str, str] | None = None,
    target_row: Dict[str, str] | None = None,
    delta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if current_state is None:
        if seed_row is None:
            raise ValueError("seed_row is required when initializing accumulated state")
        tags = _tag_set(seed_row)
        return {
            "current_clip_id": str(seed_row.get("clip_id", "") or ""),
            "current_track_id": str(seed_row.get("track_id", "") or ""),
            "current_tags": tags,
            "added_tags_cumulative": [],
            "removed_tags_cumulative": [],
            "latest_caption": str(seed_row.get("normalized_caption", "") or seed_row.get("caption", "") or ""),
            "metadata": _metadata_snapshot(seed_row),
        }

    if source_row is None or target_row is None or delta is None:
        raise ValueError("source_row, target_row, and delta are required when updating accumulated state")

    added_cum = set(current_state.get("added_tags_cumulative", []))
    removed_cum = set(current_state.get("removed_tags_cumulative", []))

    for tag in delta.get("tags_added", []):
        removed_cum.discard(tag)
        added_cum.add(str(tag))
    for tag in delta.get("tags_removed", []):
        added_cum.discard(tag)
        removed_cum.add(str(tag))

    return {
        "current_clip_id": str(target_row.get("clip_id", "") or ""),
        "current_track_id": str(target_row.get("track_id", "") or ""),
        "current_tags": _tag_set(target_row),
        "added_tags_cumulative": sorted(added_cum),
        "removed_tags_cumulative": sorted(removed_cum),
        "latest_caption": str(target_row.get("normalized_caption", "") or target_row.get("caption", "") or ""),
        "metadata": _metadata_snapshot(target_row),
    }


def run_chains(cfg: DictConfig) -> Dict[str, object]:
    structured_path = Path(str(cfg.stage.io.input_manifest_csv))
    nodes_path = Path(str(cfg.stage.io.input_nodes_csv))
    transition_memmap_dir = Path(str(cfg.stage.io.input_transition_memmap_dir))
    transition_shards_dir = Path(str(cfg.stage.io.input_transition_shards_dir))
    transition_shards_metadata_path = Path(str(cfg.stage.io.input_transition_shards_metadata_json))
    out_dir = Path(str(cfg.stage.io.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / str(cfg.stage.io.output_chains_jsonl)
    report_path = out_dir / str(cfg.stage.io.report_file)
    tracker = StageTracker(
        cfg,
        "chains",
        title="Sample Multi-Step Chains",
        subtitle=f"transition_memmap={transition_memmap_dir}",
        total_steps=4,
    )

    for path in (structured_path, nodes_path):
        if not path.exists():
            raise FileNotFoundError(f"Required chains input not found: {path}")

    tracker.step("Load manifests and graph indexes", detail=f"manifest={structured_path.name}, nodes={nodes_path.name}")
    structured_by_clip = _structured_index(structured_path)
    nodes_by_idx = _nodes_index(nodes_path)
    transition_shard_cache_size = max(1, int(getattr(cfg.stage.runtime, "transition_shard_cache_size", 2)))
    transition_mode = "memmap"
    transition_store: _TransitionMemmapStore | _TransitionShardStore
    if (transition_memmap_dir / "metadata.json").exists():
        _log(cfg, f"Using memmap transition adjacency from {transition_memmap_dir}")
        transition_store = _TransitionMemmapStore(
            cfg=cfg,
            memmap_dir=transition_memmap_dir,
        )
        valid_seed_indices = transition_store.valid_seed_indices()
        _log(
            cfg,
            f"Memmap metadata loaded. node_count={transition_store.node_count:,}, "
            f"edge_count={transition_store.edge_count:,}, valid_seed_nodes={len(valid_seed_indices):,}",
        )
    else:
        transition_mode = "shards"
        for path in (transition_shards_dir, transition_shards_metadata_path):
            if not path.exists():
                raise FileNotFoundError(f"Required chains input not found: {path}")
        _log(cfg, f"Using sharded transition adjacency from {transition_shards_dir}")
        transition_store = _TransitionShardStore(
            cfg=cfg,
            shards_dir=transition_shards_dir,
            metadata_path=transition_shards_metadata_path,
            cache_size=transition_shard_cache_size,
        )
        valid_seed_indices = transition_store.valid_seed_indices()
        _log(
            cfg,
            f"Shard metadata loaded. shard_count={transition_store.shard_count:,}, "
            f"cache_size={transition_shard_cache_size:,}, valid_seed_nodes={len(valid_seed_indices):,}",
        )
    if not valid_seed_indices:
        raise ValueError("No valid graph seed nodes with outgoing edges were found.")

    rng = random.Random(int(cfg.stage.behavior.random_seed))
    target_num_chains = int(cfg.stage.behavior.target_num_chains)
    max_attempts = max(target_num_chains, int(cfg.stage.behavior.max_chain_attempts))
    keep_shorter = bool(cfg.stage.behavior.keep_shorter_on_dead_end)
    min_chain_len = max(1, int(cfg.stage.behavior.min_chain_length))
    every_n = max(1, int(cfg.stage.progress.every_n_rows))
    attempt_log_every = max(every_n * 10, 10_000)

    counts = {
        "target_num_chains": target_num_chains,
        "chains_written": 0,
        "attempts": 0,
        "dead_end_attempts": 0,
        "shorter_chains_kept": 0,
        "discarded_attempts_too_short": 0,
    }
    length_counts: Dict[int, int] = {}

    def _emit_attempt_log() -> None:
        if counts["attempts"] % attempt_log_every == 0:
            _log(
                cfg,
                f"Attempts: {counts['attempts']:,} / {max_attempts:,} "
                f"(chains_written={counts['chains_written']:,}, dead_ends={counts['dead_end_attempts']:,}, "
                f"discarded_too_short={counts['discarded_attempts_too_short']:,})",
            )

    tracker.step("Sample chain walks", detail=f"target={target_num_chains:,}, valid_seeds={len(valid_seed_indices):,}")
    with out_jsonl.open("w", encoding="utf-8") as f:
        with rich_tqdm(cfg, total=target_num_chains, desc="Sample chains", unit="chain") as progress:
            while counts["chains_written"] < target_num_chains and counts["attempts"] < max_attempts:
                counts["attempts"] += 1
                sampled_length = _sample_target_chain_length(cfg, rng)
                seed_idx = rng.choice(valid_seed_indices)
                seed_node = nodes_by_idx[seed_idx]
                seed_clip_id = str(seed_node.get("clip_id", "") or "")
                seed_track_id = str(seed_node.get("track_id", "") or "")
                seed_row = structured_by_clip.get(seed_clip_id)
                if seed_row is None:
                    counts["discarded_attempts_too_short"] += 1
                    _emit_attempt_log()
                    continue

                visited_node_idxs = {seed_idx}
                visited_clip_ids = {seed_clip_id}
                visited_track_ids = {seed_track_id}
                current_idx = seed_idx
                accumulated_state = _update_accumulated_state(None, seed_row=seed_row)
                turns: List[Dict[str, Any]] = []

                for turn_index in range(1, sampled_length + 1):
                    candidate_positions: List[int] = []
                    if transition_mode == "memmap":
                        targets_arr, scores_arr, costs_arr = transition_store.get_edge_block(current_idx)
                        for pos in range(len(targets_arr)):
                            target_idx = int(targets_arr[pos])
                            target_node = nodes_by_idx.get(target_idx)
                            if target_node is None:
                                continue
                            target_clip_id = str(target_node.get("clip_id", "") or "")
                            target_track_id = str(target_node.get("track_id", "") or "")
                            if target_idx in visited_node_idxs:
                                continue
                            if target_clip_id in visited_clip_ids:
                                continue
                            if target_track_id in visited_track_ids:
                                continue
                            candidate_positions.append(pos)
                    else:
                        current_edges = transition_store.get_edges(current_idx)
                        for pos, edge in enumerate(current_edges):
                            target_idx = int(edge["target_node_idx"])
                            target_node = nodes_by_idx.get(target_idx)
                            if target_node is None:
                                continue
                            target_clip_id = str(target_node.get("clip_id", "") or "")
                            target_track_id = str(target_node.get("track_id", "") or "")
                            if target_idx in visited_node_idxs:
                                continue
                            if target_clip_id in visited_clip_ids:
                                continue
                            if target_track_id in visited_track_ids:
                                continue
                            candidate_positions.append(pos)

                    if not candidate_positions:
                        counts["dead_end_attempts"] += 1
                        break

                    if transition_mode == "memmap":
                        chosen_pos = _choose_edge_index(
                            rng,
                            scores=scores_arr,
                            candidate_positions=candidate_positions,
                        )
                        target_idx = int(targets_arr[chosen_pos])
                        transition_score = float(scores_arr[chosen_pos])
                        transition_cost = float(costs_arr[chosen_pos])
                        hardness = ""
                    else:
                        edge = _choose_edge(
                            rng,
                            [current_edges[pos] for pos in candidate_positions],
                        )
                        target_idx = int(edge["target_node_idx"])
                        transition_score = float(edge["transition_score"])
                        transition_cost = float(edge["transition_cost"])
                        hardness = str(edge["hardness"])

                    target_node = nodes_by_idx[target_idx]
                    target_clip_id = str(target_node.get("clip_id", "") or "")
                    target_track_id = str(target_node.get("track_id", "") or "")
                    target_row = structured_by_clip.get(target_clip_id)
                    source_row = structured_by_clip.get(str(nodes_by_idx[current_idx].get("clip_id", "") or ""))
                    if target_row is None or source_row is None:
                        break

                    delta = _structured_delta(source_row, target_row)
                    if transition_mode != "memmap":
                        delta = edge["structured_delta"] or delta
                    if not hardness:
                        hardness = _hardness(len(delta["tags_added"]) + len(delta["tags_removed"]))
                    accumulated_state = _update_accumulated_state(
                        accumulated_state,
                        source_row=source_row,
                        target_row=target_row,
                        delta=delta,
                    )
                    turns.append(
                        {
                            "turn_index": turn_index,
                            "source_node_idx": current_idx,
                            "target_node_idx": target_idx,
                            "source_clip_id": str(nodes_by_idx[current_idx].get("clip_id", "") or ""),
                            "target_clip_id": target_clip_id,
                            "source_track_id": str(nodes_by_idx[current_idx].get("track_id", "") or ""),
                            "target_track_id": target_track_id,
                            "split": str(target_row.get("split", "") or source_row.get("split", "") or ""),
                            "transition_score": transition_score,
                            "transition_cost": transition_cost,
                            "hardness": hardness,
                            "structured_delta_json": json.dumps(delta, ensure_ascii=True),
                            "accumulated_intent_state_json": json.dumps(accumulated_state, ensure_ascii=True),
                        }
                    )

                    visited_node_idxs.add(target_idx)
                    visited_clip_ids.add(target_clip_id)
                    visited_track_ids.add(target_track_id)
                    current_idx = target_idx

                realized_length = len(turns)
                if realized_length < min_chain_len:
                    counts["discarded_attempts_too_short"] += 1
                    _emit_attempt_log()
                    continue
                if realized_length < sampled_length and keep_shorter:
                    counts["shorter_chains_kept"] += 1

                chain_id = f"chain_{counts['chains_written']:08d}"
                length_counts[realized_length] = length_counts.get(realized_length, 0) + 1
                chain_record = {
                    "chain_id": chain_id,
                    "chain_length": realized_length,
                    "sampled_target_length": sampled_length,
                    "split": turns[-1]["split"] if turns else str(seed_row.get("split", "") or ""),
                    "seed": {
                        "node_idx": seed_idx,
                        "clip_id": seed_clip_id,
                        "track_id": seed_track_id,
                    },
                    "steps": [
                        {
                            "turn_index": turn["turn_index"],
                            "source_node_idx": turn["source_node_idx"],
                            "target_node_idx": turn["target_node_idx"],
                            "source_clip_id": turn["source_clip_id"],
                            "target_clip_id": turn["target_clip_id"],
                            "source_track_id": turn["source_track_id"],
                            "target_track_id": turn["target_track_id"],
                            "split": turn["split"],
                            "transition_score": round(float(turn["transition_score"]), 8),
                            "transition_cost": round(float(turn["transition_cost"]), 8),
                            "hardness": turn["hardness"],
                            "structured_delta": json.loads(turn["structured_delta_json"]),
                            "accumulated_intent_state": json.loads(turn["accumulated_intent_state_json"]),
                        }
                        for turn in turns
                    ],
                }
                f.write(json.dumps(chain_record, ensure_ascii=True) + "\n")

                counts["chains_written"] += 1
                progress.update(1)
                if counts["chains_written"] % every_n == 0:
                    _log(
                        cfg,
                        f"Chains written: {counts['chains_written']:,} / {target_num_chains:,} "
                        f"(attempts={counts['attempts']:,}, dead_ends={counts['dead_end_attempts']:,}, "
                        f"shorter_kept={counts['shorter_chains_kept']:,})",
                    )
                _emit_attempt_log()

    tracker.step("Write report", detail=report_path.name)
    report = {
        "stage": "chains",
        "input": {
            "input_manifest_csv": str(structured_path),
            "input_nodes_csv": str(nodes_path),
            "input_transition_memmap_dir": str(transition_memmap_dir) if transition_memmap_dir.exists() else None,
            "input_transition_shards_dir": str(transition_shards_dir) if transition_shards_dir.exists() else None,
            "input_transition_shards_metadata_json": str(transition_shards_metadata_path) if transition_shards_metadata_path.exists() else None,
        },
        "counts": {
            **counts,
            "valid_seed_nodes": len(valid_seed_indices),
            "chain_length_histogram": {str(k): v for k, v in sorted(length_counts.items())},
        },
        "config": {
            "behavior": _cfg_section_to_plain(cfg.stage.behavior),
            "sampling": _cfg_section_to_plain(cfg.stage.sampling),
            "runtime": _cfg_section_to_plain(cfg.stage.runtime),
            "transition_source_mode": transition_mode,
        },
        "outputs": {
            "output_chains_jsonl": str(out_jsonl),
            "report": str(report_path),
        },
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)

    tracker.finish(
        f"wrote {counts['chains_written']:,} chains after {counts['attempts']:,} attempts"
    )
    _log(
        cfg,
        f"Chain mining complete. Wrote {counts['chains_written']:,} chains "
        f"after {counts['attempts']:,} attempts",
    )
    return report


def _main_impl(cfg: DictConfig) -> None:
    report = run_chains(cfg)
    print(json.dumps({"status": "ok", "stage": "chains", "outputs": report["outputs"]}, indent=2))


def main() -> None:
    import hydra

    @hydra.main(version_base=None, config_path=CONF_DIR, config_name="config")
    def _wrapped(cfg: DictConfig) -> None:
        _main_impl(cfg)

    _wrapped()


if __name__ == "__main__":
    main()
