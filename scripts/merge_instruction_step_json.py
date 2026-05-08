#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple


def _iter_records(input_dirs: Sequence[Path]) -> Iterable[Dict[str, Any]]:
    for input_dir in input_dirs:
        for path in sorted(input_dir.glob("*.json")):
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError(f"{path} did not contain a JSON object")
            yield data


def _record_key(record: Dict[str, Any]) -> Tuple[str, int]:
    return str(record.get("chain_id", "")), int(record.get("turn_index", 0) or 0)


def merge_step_json(input_dirs: Sequence[Path], output_jsonl: Path, *, overwrite: bool = False) -> Dict[str, int]:
    if output_jsonl.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_jsonl}")
    records_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
    duplicate_count = 0
    for record in _iter_records(input_dirs):
        key = _record_key(record)
        if key in records_by_key:
            duplicate_count += 1
        records_by_key[key] = record
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as out_f:
        for key in sorted(records_by_key):
            out_f.write(json.dumps(records_by_key[key], ensure_ascii=True) + "\n")
    return {"records_written": len(records_by_key), "duplicates_dropped": duplicate_count}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge per-step instruction JSON files into validation-ready JSONL.")
    parser.add_argument("input_dirs", nargs="+", type=Path, help="Directory or directories containing per-step *.json records.")
    parser.add_argument("--output", required=True, type=Path, help="Output chain_step_instructions.jsonl path.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output JSONL if it already exists.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    stats = merge_step_json(args.input_dirs, args.output, overwrite=bool(args.overwrite))
    print(json.dumps({"status": "ok", **stats, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
