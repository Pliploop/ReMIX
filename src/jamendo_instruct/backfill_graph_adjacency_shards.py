from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Dict, TextIO

from rich.progress import (
    BarColumn,
    FileSizeColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


class AdjacencyShardWriter:
    def __init__(
        self,
        *,
        shards_dir: Path,
        metadata_path: Path,
        fieldnames: list[str],
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
            _, writer = existing
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

    def write_row(self, row: Dict[str, str]) -> None:
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

def build_shards(
    *,
    input_csv: Path,
    output_dir: Path,
    shard_count: int,
    overwrite: bool,
) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_csv}")

    shards_dir = output_dir / "transition_graph_adjacency_shards"
    metadata_path = output_dir / "transition_graph_adjacency_metadata.json"

    if shards_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing shard directory: {shards_dir}")
        shutil.rmtree(shards_dir)
    shards_dir.mkdir(parents=True, exist_ok=True)

    if metadata_path.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing shard metadata: {metadata_path}")
        metadata_path.unlink()

    total_bytes = int(input_csv.stat().st_size)
    print("──────────────── BACKFILL GRAPH ADJACENCY SHARDS ────────────────", flush=True)
    print(f"input_csv={input_csv}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(f"shard_count={shard_count:,}", flush=True)

    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"Could not read CSV header from {input_csv}")

        writer = AdjacencyShardWriter(
            shards_dir=shards_dir,
            metadata_path=metadata_path,
            fieldnames=fieldnames,
            shard_count=shard_count,
        )
        row_count = 0
        last_bytes = f.buffer.tell()
        progress = Progress(
            TextColumn("[bold cyan]{task.description}[/bold cyan]"),
            BarColumn(bar_width=28, style="grey35", complete_style="bright_cyan", finished_style="green"),
            TaskProgressColumn(text_format="[bold]{task.percentage:>3.0f}%[/bold]"),
            FileSizeColumn(),
            TextColumn("/"),
            TextColumn("{task.total}", justify="right"),
            TransferSpeedColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(compact=True),
            TextColumn("rows={task.fields[rows]:,}"),
            expand=True,
            transient=False,
        )
        try:
            with progress:
                task_id = progress.add_task("Backfill adjacency shards", total=total_bytes, rows=0)
                for row in reader:
                    writer.write_row(row)
                    row_count += 1
                    current_bytes = f.buffer.tell()
                    advance = max(0, current_bytes - last_bytes)
                    last_bytes = current_bytes
                    progress.update(task_id, advance=advance, rows=row_count)
                if last_bytes < total_bytes:
                    progress.update(task_id, advance=total_bytes - last_bytes, rows=row_count)
        finally:
            writer.close()

    writer.write_metadata()
    print(
        json.dumps(
            {
                "status": "ok",
                "input_csv": str(input_csv),
                "rows_processed": row_count,
                "output_adjacency_shards_dir": str(shards_dir),
                "output_adjacency_metadata_json": str(metadata_path),
                "shard_count": shard_count,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill graph adjacency shards from an existing transition graph CSV.")
    parser.add_argument("--input-csv", required=True, help="Path to the existing transition graph CSV.")
    parser.add_argument("--output-dir", required=True, help="Graph output directory where shard artifacts should be written.")
    parser.add_argument("--shard-count", type=int, default=4096, help="Number of modulo shards to write.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite any existing shard artifacts.")
    args = parser.parse_args()

    build_shards(
        input_csv=Path(args.input_csv),
        output_dir=Path(args.output_dir),
        shard_count=int(args.shard_count),
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()
