from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def _iter_csv_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def _iter_shard_paths(shards_dir: Path) -> list[Path]:
    return sorted(shards_dir.glob("adjacency_shard_*.csv"))


def _node_count_from_nodes_csv(nodes_csv: Path) -> int:
    max_node_idx = -1
    for row in _iter_csv_rows(nodes_csv):
        max_node_idx = max(max_node_idx, int(row["node_idx"]))
    return max_node_idx + 1


def _total_edges_from_shard_metadata(metadata_path: Path) -> int | None:
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    counts = metadata.get("row_counts_by_shard", {})
    if not isinstance(counts, dict):
        return None
    return sum(int(v) for v in counts.values())


def build_memmap(
    *,
    shards_dir: Path,
    shard_metadata_json: Path,
    nodes_csv: Path,
    output_dir: Path,
    overwrite: bool,
) -> None:
    if not shards_dir.exists():
        raise FileNotFoundError(f"Missing shards directory: {shards_dir}")
    if not nodes_csv.exists():
        raise FileNotFoundError(f"Missing nodes CSV: {nodes_csv}")

    shard_paths = _iter_shard_paths(shards_dir)
    if not shard_paths:
        raise FileNotFoundError(f"No shard CSVs found under: {shards_dir}")

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output dir: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_edges_hint = _total_edges_from_shard_metadata(shard_metadata_json)
    node_count = _node_count_from_nodes_csv(nodes_csv)

    print("──────────────── BACKFILL GRAPH ADJACENCY MEMMAP ────────────────", flush=True)
    print(f"shards_dir={shards_dir}", flush=True)
    print(f"nodes_csv={nodes_csv}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(f"num_shards={len(shard_paths):,}", flush=True)
    print(f"node_count={node_count:,}", flush=True)

    degrees = np.zeros(node_count, dtype=np.int64)
    total_edges = 0

    progress = Progress(
        TextColumn("[bold cyan]{task.description}[/bold cyan]"),
        BarColumn(bar_width=28, style="grey35", complete_style="bright_cyan", finished_style="green"),
        TaskProgressColumn(text_format="[bold]{task.percentage:>3.0f}%[/bold]"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(compact=True),
        expand=True,
        transient=False,
    )

    with progress:
        task1 = progress.add_task("Adjacency pass 1/2", total=len(shard_paths))
        for shard_path in shard_paths:
            for row in _iter_csv_rows(shard_path):
                degrees[int(row["source_node_idx"])] += 1
                total_edges += 1
            progress.update(task1, advance=1)

        offsets = np.empty(node_count + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(degrees, out=offsets[1:])

        edge_count = int(offsets[-1])
        if total_edges_hint is not None and edge_count != int(total_edges_hint):
            raise ValueError(
                f"Shard metadata edge count mismatch: metadata={int(total_edges_hint)} vs scanned={edge_count}"
            )

        targets = np.lib.format.open_memmap(
            output_dir / "targets.npy",
            mode="w+",
            dtype=np.int32,
            shape=(edge_count,),
        )
        scores = np.lib.format.open_memmap(
            output_dir / "scores.npy",
            mode="w+",
            dtype=np.float32,
            shape=(edge_count,),
        )
        costs = np.lib.format.open_memmap(
            output_dir / "costs.npy",
            mode="w+",
            dtype=np.float32,
            shape=(edge_count,),
        )
        cursors = offsets[:-1].copy()

        task2 = progress.add_task("Adjacency pass 2/2", total=len(shard_paths))
        for shard_path in shard_paths:
            for row in _iter_csv_rows(shard_path):
                source_idx = int(row["source_node_idx"])
                write_idx = int(cursors[source_idx])
                targets[write_idx] = int(row["target_node_idx"])
                scores[write_idx] = np.float32(float(row["transition_score"]))
                costs[write_idx] = np.float32(float(row["transition_cost"]))
                cursors[source_idx] += 1
            progress.update(task2, advance=1)

    valid_seed_nodes = np.flatnonzero(degrees > 0).astype(np.int32, copy=False)
    np.save(output_dir / "offsets.npy", offsets)
    np.save(output_dir / "valid_seed_nodes.npy", valid_seed_nodes)
    targets.flush()
    scores.flush()
    costs.flush()

    metadata = {
        "format": "csr_memmap_v1",
        "node_count": int(node_count),
        "edge_count": int(edge_count),
        "offsets_file": "offsets.npy",
        "targets_file": "targets.npy",
        "scores_file": "scores.npy",
        "costs_file": "costs.npy",
        "valid_seed_nodes_file": "valid_seed_nodes.npy",
        "offsets_dtype": "int64",
        "targets_dtype": "int32",
        "scores_dtype": "float32",
        "costs_dtype": "float32",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir),
                "node_count": int(node_count),
                "edge_count": int(edge_count),
                "valid_seed_nodes": int(valid_seed_nodes.shape[0]),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill graph adjacency memmap from existing shard CSV artifacts.")
    parser.add_argument("--shards-dir", required=True)
    parser.add_argument("--shard-metadata-json", required=True)
    parser.add_argument("--nodes-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    build_memmap(
        shards_dir=Path(args.shards_dir),
        shard_metadata_json=Path(args.shard_metadata_json),
        nodes_csv=Path(args.nodes_csv),
        output_dir=Path(args.output_dir),
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()
