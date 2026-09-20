#!/usr/bin/env python
"""Pretty progress for the ReMIX-B relevance-pool shard jobs.

Counts steps written across each dataset's shard files, compares to the test-split
target, reads the SLURM queue for running/pending shards, and estimates a rate +
ETA from the running jobs' elapsed time.

  python scripts/relevance_pool_progress.py
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple

from rich.console import Console
from rich.table import Table

# (label, run_root). The coverage ceiling is derived live from the validation
# gate, not hardcoded — validation coverage grows, so a fixed number goes stale.
DATASETS: List[Tuple[str, str]] = [
    ("Music4All", "/gpfs/scratch/acw749/datasets/music4all_instruct/music4all_v1"),
    ("MTG-Jamendo", "/gpfs/scratch/acw749/datasets/mtg_jamendo_instruct/v1"),
]
FOLDER = "instructions_axis_focused_5"
# Splits the pool targets (run_relevance_pool.sh SPLITS default is "test").
SPLITS: Set[str] = set((os.environ.get("RELPOOL_SPLITS") or "test").split(","))

Step = Tuple[str, int]  # (chain_id, turn_index)


def _shard_dir(root: str) -> Path:
    return Path(root) / FOLDER / "relevance_pool"


def _gate_accepted(root: str) -> Set[Step]:
    """Accepted steps in the configured splits = the real coverage ceiling."""
    gate = Path(root) / FOLDER / "validation" / "validated_instructions.jsonl"
    out: Set[Step] = set()
    if not gate.exists():
        return out
    with open(gate) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("split") in SPLITS and (r.get("validation") or {}).get("accepted"):
                out.add((r["chain_id"], r["turn_index"]))
    return out


def _pool_steps(root: str) -> Tuple[Set[Step], int]:
    """(pooled steps, shard files) across chain_step_relevance_pools.shard*.jsonl."""
    steps: Set[Step] = set()
    files = sorted(glob.glob(str(_shard_dir(root) / "chain_step_relevance_pools.shard*.jsonl")))
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                steps.add((r["chain_id"], r["turn_index"]))
    return steps, len(files)


def _squeue_rows() -> List[Dict[str, str]]:
    try:
        out = subprocess.run(
            ["squeue", "--me", "-h", "-o", "%i|%j|%T|%M"],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) == 4 and "relpool" in parts[1]:
            rows.append({"id": parts[0], "name": parts[1], "state": parts[2], "elapsed": parts[3]})
    return rows


def _elapsed_hours(s: str) -> float:
    # SLURM elapsed: [D-]HH:MM:SS or MM:SS
    d = 0
    if "-" in s:
        dstr, s = s.split("-", 1)
        d = int(dstr)
    parts = [int(p) for p in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts
    return d * 24 + h + m / 60 + sec / 3600


def main() -> None:
    console = Console()
    jobs = _squeue_rows()
    by_ds_running = {"m4a": [], "mtg": []}
    running = pending = 0
    max_elapsed = 0.0
    for j in jobs:
        running += j["state"] == "RUNNING"
        pending += j["state"] == "PENDING"
        if j["state"] == "RUNNING":
            max_elapsed = max(max_elapsed, _elapsed_hours(j["elapsed"]))

    split_lbl = "/".join(sorted(SPLITS))
    t = Table(title="ReMIX-B relevance-pool progress", title_style="bold", header_style="bold cyan", box=None)
    for col in ("Dataset", "Steps done", f"Target ({split_lbl})", "Coverage", "Shard files", "Rate", "ETA"):
        t.add_column(col, justify="right" if col != "Dataset" else "left")

    total_orphans = 0
    for label, root in DATASETS:
        gate = _gate_accepted(root)
        pooled, nfiles = _pool_steps(root)
        done = len(pooled & gate)                    # in-gate pooled steps
        orphans = len(pooled - gate)                 # stale records from a pre-refresh run
        total_orphans += orphans
        target = len(gate)                           # real ceiling, live from the gate
        cov = f"{100 * done / target:.1f}%" if target else "?"
        # crude rate: steps done / max running elapsed (of any shard), across shards
        rate = done / max_elapsed if max_elapsed > 0 else 0.0
        remaining = max(0, target - done)
        eta = f"{remaining / rate:.0f} h" if rate > 0 else "[dim]—[/dim]"
        rate_s = f"{rate:.0f}/h" if rate > 0 else "[dim]—[/dim]"
        t.add_row(label, f"{done:,}", f"{target:,}",
                  f"[green]{cov}[/green]" if done >= target else cov,
                  str(nfiles), rate_s, eta)

    console.print()
    console.print(t)
    console.print(f"\n  shards: [green]{running} running[/green], {pending} pending "
                  f"(max elapsed {max_elapsed:.1f} h)")
    if total_orphans:
        console.print(f"  [yellow]{total_orphans:,} pooled steps not in the current gate[/yellow] "
                      f"(stale from a pre-refresh run; not counted)")
    if not jobs:
        console.print("  [yellow]no relpool jobs in the queue[/yellow] (finished, or not launched)")


if __name__ == "__main__":
    main()
