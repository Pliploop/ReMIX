#!/usr/bin/env python
"""Run the ReMIX-B composed-retrieval benchmark on one or more baselines.

Everything goes through the common contract in `jamendo_instruct.benchmark`
(docs/evaluation.md): a baseline maps (seed + instruction) -> a ranked list of
corpus clip ids; `evaluate` scores those against the graded relevance pool. This
script wires the registered baselines to that contract, writes per-baseline
metrics to --output-dir, and pretty-prints a combined table.

Baselines (see src/jamendo_instruct/benchmark/baselines/):
  no model:   random, seed_audio_nn, seed_caption_nn, target_caption_oracle
  embedders:  instruction_text, caption_plus_instruction, mulan_zeroshot, late_fusion
Use `--baselines all` for every registered baseline.

Prereqs:
  * scripts/export_benchmark.py has been run (corpus_*.npy, queries.jsonl exist).
  * embedder baselines need a GPU and the local HF cache (offline):
        HF_HOME=/gpfs/scratch/acw749/hf_cache  HF_HUB_OFFLINE=1

Examples:
  PYTHONPATH=src python scripts/run_remix_b.py --baselines seed_audio_nn target_caption_oracle --output-dir results/

  # on slurm (GPU, for the embedder/LLM baselines):
  sbatch -p sae -A pilot_sae_gpu --gres=gpu:1 -c 8 --mem=48G -t 02:00:00 \
    --wrap "export PYTHONPATH=src HF_HOME=/gpfs/scratch/acw749/hf_cache HF_HUB_OFFLINE=1 && \
            python scripts/run_remix_b.py --baselines all --output-dir results/music4all"
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from jamendo_instruct.benchmark import evaluate
from jamendo_instruct.benchmark import baselines as B
from jamendo_instruct.benchmark.graded_pool import _read_qrels
from jamendo_instruct.benchmark.baselines.base import Corpus, load_queries

DATASETS = {
    "music4all": "/gpfs/scratch/acw749/datasets/music4all_instruct/music4all_v1",
    "mtg_jamendo": "/gpfs/scratch/acw749/datasets/mtg_jamendo_instruct/v1",
}
FOLDER = "instructions_axis_focused_5"


def _pretty(rows) -> None:
    """Aligned table via rich if available, else plain text. rows: (name, rep, sec, flops)."""
    metrics = list(rows[0][1]) if rows else []
    try:
        from rich.console import Console
        from rich.table import Table
        t = Table(title="ReMIX-B results", header_style="bold cyan", box=None)
        t.add_column("baseline", justify="left")
        for m in metrics:
            t.add_column(m, justify="right")
        t.add_column("sec", justify="right"); t.add_column("TFLOP", justify="right")
        best = {m: max(r[1][m] for r in rows) for m in metrics}
        for name, rep, sec, fl in rows:
            cells = [name]
            for m in metrics:
                v = f"{rep[m]:.4f}"
                cells.append(f"[bold green]{v}[/bold green]" if rep[m] == best[m] else v)
            cells += [f"{sec:.0f}", f"{fl/1e12:.2f}"]
            t.add_row(*cells)
        Console().print(t)
    except Exception:
        w = max((len(n) for n, *_ in rows), default=8)
        head = [f"{'baseline':<{w}}"] + [f"{m:>12}" for m in metrics] + [f"{'sec':>8}", f"{'TFLOP':>8}"]
        print("  " + " ".join(head))
        for name, rep, sec, fl in rows:
            print("  " + f"{name:<{w}} " + " ".join(f"{rep[m]:>12.4f}" for m in metrics)
                  + f" {sec:>8.0f} {fl/1e12:>8.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baselines", nargs="+", default=["all"], help="baseline names, or 'all'")
    ap.add_argument("--dataset", choices=list(DATASETS), default="music4all")
    ap.add_argument("--output-dir", default=None, help="where per-baseline metrics + the table are written (omit with --dry-run)")
    ap.add_argument("--dry-run", action="store_true", help="print the table only; write nothing (use until the pool is final)")
    ap.add_argument("--k", type=int, default=200, help="ranking depth submitted to the scorer")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="cap #queries (0 = all); for quick validation")
    args = ap.parse_args()
    if not args.dry_run and not args.output_dir:
        ap.error("--output-dir is required unless --dry-run")

    root = Path(DATASETS[args.dataset]) / FOLDER
    bench, pool = root / "benchmark", root / "relevance_pool"
    out = None
    if not args.dry_run:
        out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    corpus = Corpus.load(bench)
    queries = load_queries(bench)
    if args.limit:
        queries = queries[:args.limit]
    qrels = _read_qrels(str(pool), min_grade=3)
    print(f"corpus {len(corpus.ids):,} clips | {len(queries):,} queries | {len(qrels):,} judged\n", flush=True)

    names = B.all_names() if args.baselines == ["all"] else args.baselines
    rows = []
    for name in names:
        t0 = time.time()
        b = B.get(name)()
        b.prepare(corpus)
        if hasattr(b, "rank_all"):
            preds = b.rank_all(queries, args.k)
        else:
            preds = {q.query_id: b.rank(q, args.k) for q in queries}
        rep = evaluate(preds, qrels)
        dt = time.time() - t0
        flops = int(getattr(b, "flops", 0) or 0)
        rows.append((name, rep, dt, flops))
        if out:
            (out / f"{name}.json").write_text(json.dumps(
                {"baseline": name, "dataset": args.dataset, "split": args.split, "k": args.k,
                 "n_queries": len(queries), "n_corpus": len(corpus.ids),
                 "seconds": round(dt, 1), "flops": flops, "tflops": round(flops / 1e12, 3),
                 "metrics": dict(rep)}, indent=2))
        print(f"[{name}] ({dt:.0f}s, {flops/1e12:.2f} TFLOP)\n{rep}\n", flush=True)

    metrics = list(rows[0][1]) if rows else []
    if out:
        (out / "results.json").write_text(json.dumps(
            {n: {"metrics": dict(r), "seconds": round(s, 1), "tflops": round(fl / 1e12, 3)}
             for n, r, s, fl in rows}, indent=2))
        with (out / "results.md").open("w") as f:
            cols = metrics + ["sec", "TFLOP"]
            f.write("| baseline | " + " | ".join(cols) + " |\n")
            f.write("|" + "---|" * (len(cols) + 1) + "\n")
            for name, rep, sec, fl in rows:
                f.write(f"| {name} | " + " | ".join(f"{rep[m]:.4f}" for m in metrics)
                        + f" | {sec:.0f} | {fl/1e12:.2f} |\n")
    print()
    _pretty(rows)
    print(f"\n{'DRY RUN — nothing written' if not out else f'wrote {len(rows)} baselines -> {out}'}")


if __name__ == "__main__":
    main()
