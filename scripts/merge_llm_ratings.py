#!/usr/bin/env python
"""Merge per-item LLM-rating JSON parts into the canonical llm_ratings*.jsonl.

The judge (scripts/llm_validation_judge.py) writes one atomic JSON per rated item
into a ``<output-name>.parts`` directory, exactly like instruction generation. This
folds those parts -- optionally together with an existing ratings JSONL -- into a
single JSONL in the same record format the validation analysis and the website
already read, so nothing downstream changes.

Dedup key is (chain_id, turn_index, variant_index, instruction_field), mirroring
``_rating_item_key``. A parts dir holds one judge's ratings, so the annotator is
uniform; the key stays judge-agnostic on purpose, so passing an existing JSONL from
the *same* judge lets a new at-scale run extend it in place. Later parts win ties,
except a parse_ok record is never overwritten by a failed one.

Output is a canonical <stem>.validated.jsonl next to the parts dir, and the raw
<stem>.jsonl sidecar is auto-folded in read-only -- so the raw, instructed-but-
unvalidated sidecar is never overwritten.

Examples:
  # Default: folds llm_ratings_qwen_full.jsonl + its .parts -> *.validated.jsonl
  python scripts/merge_llm_ratings.py /path/validation/llm_ratings_qwen_full.parts

  # Parts only, ignore any existing sidecar:
  python scripts/merge_llm_ratings.py /path/validation/llm_ratings_gemma_full.parts --no-existing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple

Key = Tuple[str, int, int, str]


def _item_key(record: Dict[str, Any]) -> Key:
    return (
        str(record.get("chain_id", "") or ""),
        int(record.get("turn_index", 0) or 0),
        int(record.get("variant_index", 0) or 0),
        str(record.get("instruction_field", "") or ""),
    )


def _iter_parts(part_dirs: Sequence[Path]) -> Iterable[Dict[str, Any]]:
    from tqdm import tqdm

    for part_dir in part_dirs:
        files = sorted(part_dir.glob("*.json"))
        for path in tqdm(files, desc=f"parts {part_dir.name}", unit="rec"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                print(f"[merge] skipping unreadable part {path}: {exc}", file=sys.stderr)
                continue
            if isinstance(data, dict):
                yield data


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                yield data


def _prefer(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Later record wins, but never trade a parsed rating for a failed one."""
    if bool(old.get("parse_ok")) and not bool(new.get("parse_ok")):
        return old
    return new


def merge(
    part_dirs: Sequence[Path],
    output_jsonl: Path,
    *,
    existing: Sequence[Path] = (),
) -> Dict[str, int]:
    by_key: Dict[Key, Dict[str, Any]] = {}
    from_existing = 0
    from_parts = 0
    replaced = 0

    # Existing JSONLs first so parts (the fresher work) win ties.
    for path in existing:
        if not path.is_file():
            continue
        for record in _iter_jsonl(path):
            by_key[_item_key(record)] = record
            from_existing += 1

    for record in _iter_parts(part_dirs):
        key = _item_key(record)
        from_parts += 1
        if key in by_key:
            replaced += 1
            by_key[key] = _prefer(by_key[key], record)
        else:
            by_key[key] = record

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    # Atomic swap: never leave a half-written canonical file, even if it is also an
    # --existing input we are rewriting in place.
    tmp_path = output_jsonl.with_name(f"{output_jsonl.name}.tmp.{os.getpid()}.{time.time_ns()}")
    with tmp_path.open("w", encoding="utf-8") as out_f:
        for key in sorted(by_key):
            out_f.write(json.dumps(by_key[key], ensure_ascii=True) + "\n")
    os.replace(tmp_path, output_jsonl)

    return {
        "records_written": len(by_key),
        "from_existing": from_existing,
        "from_parts": from_parts,
        "keys_replaced": replaced,
    }


def _stem(part_dir: Path) -> str:
    n = part_dir.name
    return n[: -len(".parts")] if n.endswith(".parts") else n


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("part_dirs", nargs="+", type=Path, help="One or more <name>.parts directories.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Canonical file to write. Default: <stem>.validated.jsonl next to the parts dir. "
        "The default never equals the raw <stem>.jsonl, so the raw sidecar is never overwritten.",
    )
    parser.add_argument(
        "--existing",
        nargs="*",
        type=Path,
        default=None,
        help="Ratings JSONL(s) to fold in first. Default: auto-fold the sibling raw <stem>.jsonl "
        "if present. --no-existing to fold none.",
    )
    parser.add_argument("--no-existing", action="store_true", help="Merge only the parts; fold in no existing JSONL.")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    part_dirs = args.part_dirs

    if args.output is not None:
        output = args.output
    elif len(part_dirs) == 1:
        output = part_dirs[0].parent / f"{_stem(part_dirs[0])}.validated.jsonl"
    else:
        parser.error("--output is required when merging more than one parts dir")

    if args.existing is not None:
        existing = args.existing
    elif args.no_existing:
        existing = []
    else:  # auto-fold each parts dir's sibling raw sidecar, if it exists
        existing = [
            raw for pd in part_dirs if (raw := pd.parent / f"{_stem(pd)}.jsonl").is_file()
        ]

    stats = merge(part_dirs, output, existing=existing)
    print(json.dumps(
        {"status": "ok", **stats, "existing": [str(e) for e in existing], "output": str(output)},
        indent=2,
    ))


if __name__ == "__main__":
    main()
