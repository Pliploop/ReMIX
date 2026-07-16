#!/usr/bin/env python
"""Chain / step / variant breakdown per dataset, straight from the live outputs.

Reads the per-step JSON filenames (which encode `chain__turn_NNNNNN__variant_NNN`)
rather than parsing file contents, so it is fast (~10^5 files in seconds) and — unlike
`chain_step_instructions.jsonl` — reflects work that generation jobs have written but
that has not been merged yet. The merged JSONL is reported alongside as a drift check.

Examples:
  python scripts/dataset_breakdown.py
  python scripts/dataset_breakdown.py --folder instructions_axis_focused_5
  python scripts/dataset_breakdown.py --dataset music4all:/path/to/run_root
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from rich.console import Console
from rich.table import Table

DEFAULT_DATASETS: List[Tuple[str, str]] = [
    ("Music4All", "/gpfs/scratch/acw749/datasets/music4all_instruct/music4all_v1"),
    ("MTG-Jamendo", "/gpfs/scratch/acw749/datasets/mtg_jamendo_instruct/v1"),
]

STEP_RE = re.compile(r"^(?P<chain>.+)__turn_(?P<turn>\d+)__variant_(?P<variant>\d+)__[0-9a-f]+\.json$")


def scan(step_json_dir: Path) -> Dict[str, object]:
    """Chain -> turn -> set(variant), built from filenames alone."""
    chains: Dict[str, Dict[int, set]] = defaultdict(lambda: defaultdict(set))
    files = 0
    unparsed = 0
    if not step_json_dir.is_dir():
        return {"missing": True}
    with os.scandir(step_json_dir) as it:
        for entry in it:
            if not entry.name.endswith(".json"):
                continue
            files += 1
            m = STEP_RE.match(entry.name)
            if not m:
                unparsed += 1
                continue
            chains[m.group("chain")][int(m.group("turn"))].add(int(m.group("variant")))

    steps = sum(len(turns) for turns in chains.values())
    variants = sum(len(v) for turns in chains.values() for v in turns.values())
    variants_per_step = Counter(len(v) for turns in chains.values() for v in turns.values())
    steps_per_chain = Counter(len(turns) for turns in chains.values())
    return {
        "missing": False,
        "files": files,
        "unparsed": unparsed,
        "chains": len(chains),
        "steps": steps,
        "variants": variants,
        "variants_per_step": variants_per_step,
        "steps_per_chain": steps_per_chain,
    }


def merged_count(folder: Path) -> int | None:
    p = folder / "chain_step_instructions.jsonl"
    if not p.is_file():
        return None
    with p.open("rb") as f:
        return sum(1 for _ in f)


def _fmt(n: float, nd: int = 0) -> str:
    return f"{n:,.{nd}f}"


def merge_command(folder: Path, *, overwrite: bool) -> str:
    """The exact, copy-pasteable merge command for one dataset."""
    step_json = shlex.quote(str(folder / "step_json"))
    out = shlex.quote(str(folder / "chain_step_instructions.jsonl"))
    flag = " \\\n    --overwrite" if overwrite else ""
    return (f"python scripts/merge_instruction_step_json.py \\\n"
            f"    {step_json} \\\n"
            f"    --output {out}{flag}")


def dist_table(title: str, key: str, stats: Dict[str, Dict], labels: List[str], index_name: str) -> Table:
    t = Table(title=title, title_style="bold", header_style="bold cyan", box=None, pad_edge=False)
    t.add_column(index_name, justify="right", style="bold")
    for lab in labels:
        t.add_column(lab, justify="right")
        t.add_column("%", justify="right", style="dim")
    keys = sorted({k for lab in labels for k in stats[lab][key]})
    for k in keys:
        row = [str(k)]
        for lab in labels:
            c = stats[lab][key]
            total = sum(c.values()) or 1
            n = c.get(k, 0)
            row += [_fmt(n), f"{100 * n / total:.1f}"]
        t.add_row(*row)
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", default="instructions_axis_focused_5", help="Instructions folder name under each run root.")
    ap.add_argument("--dataset", action="append", default=None,
                    help="Extra/override dataset as LABEL:RUN_ROOT (repeatable).")
    ap.add_argument("--commands", action="store_true",
                    help="Always print the merge commands, even when nothing is pending.")
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
    stats: Dict[str, Dict] = {}
    labels: List[str] = []

    for label, root in datasets:
        folder = Path(root) / args.folder
        with console.status(f"Scanning {label} ..."):
            s = scan(folder / "step_json")
        if s.get("missing"):
            console.print(f"[yellow]! {label}: no step_json at {folder / 'step_json'}[/yellow]")
            continue
        s["merged"] = merged_count(folder)
        s["folder"] = folder
        stats[label] = s
        labels.append(label)

    if not labels:
        sys.exit("No datasets found.")

    # ---- overview -------------------------------------------------------- #
    t = Table(title=f"ReMIX breakdown — {args.folder}", title_style="bold",
              header_style="bold cyan", box=None, pad_edge=False)
    t.add_column("Metric", style="bold")
    for lab in labels:
        t.add_column(lab, justify="right")

    def row(name, fn, style=None):
        t.add_row(name, *[fn(stats[l]) for l in labels], style=style)

    row("Chains", lambda s: _fmt(s["chains"]))
    row("Steps", lambda s: _fmt(s["steps"]))
    row("Instruction variants", lambda s: _fmt(s["variants"]))
    t.add_row("")
    row("Mean steps / chain", lambda s: _fmt(s["steps"] / max(s["chains"], 1), 2))
    row("Mean variants / step", lambda s: _fmt(s["variants"] / max(s["steps"], 1), 2))
    row("Mean variants / chain", lambda s: _fmt(s["variants"] / max(s["chains"], 1), 2))
    t.add_row("")
    row("Merged JSONL rows", lambda s: _fmt(s["merged"]) if s["merged"] is not None else "[dim]none[/dim]")
    row("Unmerged (pending)", lambda s: (
        f"[yellow]{_fmt(s['variants'] - s['merged'])}[/yellow]"
        if s["merged"] is not None and s["variants"] > s["merged"] else "[green]0[/green]"
    ))
    console.print()
    console.print(t)

    # ---- distributions --------------------------------------------------- #
    console.print()
    console.print(dist_table("Variants per step", "variants_per_step", stats, labels, "Variants"))
    console.print()
    console.print(dist_table("Chain length (steps per chain)", "steps_per_chain", stats, labels, "Steps"))

    for l in labels:
        if stats[l]["unparsed"]:
            console.print(f"[yellow]{l}: {stats[l]['unparsed']} filename(s) did not parse.[/yellow]")

    # ---- copy-pasteable commands ----------------------------------------- #
    stale = [l for l in labels if stats[l]["merged"] is not None and stats[l]["variants"] > stats[l]["merged"]]
    show = stale if not args.commands else labels
    if show:
        console.print()
        if stale:
            n = sum(stats[l]["variants"] - stats[l]["merged"] for l in stale)
            console.print(f"[bold yellow]{_fmt(n)} variant(s) not yet in the merged JSONL[/bold yellow] "
                          "[dim]— downstream stages (validation, relevance_pool, sidecars, paper stats)\n"
                          "read the JSONL, not step_json, so they will not see these until you merge.\n"
                          "The merge is a full, idempotent rebuild: safe to run while jobs are still writing.[/dim]")
        console.print()
        console.rule("[bold]Merge commands[/bold]", style="dim")
        for l in show:
            s = stats[l]
            console.print(f"\n[bold cyan]# {l}[/bold cyan]")
            # Printed unstyled so it copy-pastes cleanly.
            print(merge_command(s["folder"], overwrite=s["merged"] is not None))
        console.rule(style="dim")
        console.print("\n[dim]Then re-check with:[/dim]")
        print("python scripts/dataset_breakdown.py")
    console.print()


if __name__ == "__main__":
    main()
