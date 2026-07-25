#!/usr/bin/env python
"""LLM-validation coverage per dataset and judge, straight from the live outputs.

The mirror of scripts/dataset_breakdown.py, for validation. It reads the per-item
rating filenames (``chain__turn_NNNNNN__variant_NNN__<sha>.json``) in each judge's
``validation/<name>.parts`` dir -- so it is fast and reflects ratings a judge job
has written but not yet merged. Coverage is measured against the instruction
``step_json`` total (how many variants exist to be judged), and the merged
``<name>.jsonl`` row count is reported alongside as a drift check.

Examples:
  python scripts/validation_breakdown.py
  python scripts/validation_breakdown.py --folder instructions_axis_focused_5
  python scripts/validation_breakdown.py --dataset music4all:/path/to/run_root
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.console import Console
from rich.table import Table

DEFAULT_DATASETS: List[Tuple[str, str]] = [
    ("Music4All", "/gpfs/scratch/acw749/datasets/music4all_instruct/music4all_v1"),
    ("MTG-Jamendo", "/gpfs/scratch/acw749/datasets/mtg_jamendo_instruct/v1"),
]

STEP_RE = re.compile(r"^(?P<chain>.+)__turn_(?P<turn>\d+)__variant_(?P<variant>\d+)__[0-9a-f]+\.json$")


def scan_variants(json_dir: Path) -> Optional[int]:
    """Count distinct (chain, turn, variant) from the part filenames alone."""
    if not json_dir.is_dir():
        return None
    seen: set = set()
    with os.scandir(json_dir) as it:
        for entry in it:
            if not entry.name.endswith(".json"):
                continue
            m = STEP_RE.match(entry.name)
            if m:
                seen.add((m.group("chain"), int(m.group("turn")), int(m.group("variant"))))
    return len(seen)


def jsonl_rows(path: Path) -> Optional[int]:
    if not path.is_file():
        return None
    with path.open("rb") as f:
        return sum(1 for _ in f)


def _fmt(n: float, nd: int = 0) -> str:
    return f"{n:,.{nd}f}"


def merge_command(parts_dir: Path, canonical: Path) -> str:
    """Copy-pasteable merge for one judge, extending its canonical file in place."""
    return (
        "python scripts/merge_llm_ratings.py \\\n"
        f"    {shlex.quote(str(parts_dir))} \\\n"
        f"    --existing {shlex.quote(str(canonical))} \\\n"
        f"    --output {shlex.quote(str(canonical))}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", default="instructions_axis_focused_5", help="Instructions folder under each run root.")
    ap.add_argument("--dataset", action="append", default=None, help="Extra/override dataset as LABEL:RUN_ROOT (repeatable).")
    ap.add_argument("--commands", action="store_true", help="Print the merge command for every judge, not just pending ones.")
    args = ap.parse_args()

    if args.dataset:
        datasets = []
        for spec in args.dataset:
            label, _, root = spec.partition(":")
            if not root:
                sys.exit(f"--dataset expects LABEL:RUN_ROOT, got: {spec}")
            datasets.append((label, root))
    else:
        datasets = DEFAULT_DATASETS

    console = Console()
    pending_cmds: List[str] = []
    found_any = False

    for label, root in datasets:
        folder = Path(root) / args.folder
        val_dir = folder / "validation"
        total = scan_variants(folder / "step_json")  # instruction variants = judging universe

        # One judge per <name>.parts dir. Report against the merged <name>.jsonl.
        parts_dirs = sorted(val_dir.glob("*.parts")) if val_dir.is_dir() else []
        if not parts_dirs:
            console.print(f"[yellow]! {label}: no *.parts under {val_dir}[/yellow]")
            continue
        found_any = True

        t = Table(
            title=f"{label} — validation coverage ({args.folder})",
            title_style="bold",
            header_style="bold cyan",
            box=None,
            pad_edge=False,
        )
        t.add_column("Judge", style="bold")
        t.add_column("Rated", justify="right")
        t.add_column("Merged", justify="right")
        t.add_column("Coverage", justify="right")
        t.add_column("Pending merge", justify="right")

        for parts_dir in parts_dirs:
            stem = parts_dir.name[: -len(".parts")]
            canonical = val_dir / f"{stem}.jsonl"
            rated = scan_variants(parts_dir) or 0
            merged = jsonl_rows(canonical)

            coverage = f"{100 * rated / total:.1f}% of {_fmt(total)}" if total else "[dim]?[/dim]"
            if merged is None:
                pending = f"[yellow]{_fmt(rated)}[/yellow]"  # nothing merged yet
            elif rated > merged:
                pending = f"[yellow]{_fmt(rated - merged)}[/yellow]"
            else:
                pending = "[green]0[/green]"
            t.add_row(stem, _fmt(rated), _fmt(merged) if merged is not None else "[dim]none[/dim]", coverage, pending)

            if args.commands or merged is None or rated > (merged or 0):
                pending_cmds.append(f"# {label} · {stem}\n{merge_command(parts_dir, canonical)}")

        console.print()
        console.print(t)

    if not found_any:
        sys.exit("No validation parts found. Has a judge run with --step-json-dir (the default)?")

    if pending_cmds:
        console.print()
        console.rule("[bold]Merge parts into the canonical JSONL")
        for cmd in pending_cmds:
            console.print()
            console.print(cmd, highlight=False)


if __name__ == "__main__":
    main()
