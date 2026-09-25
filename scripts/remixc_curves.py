#!/usr/bin/env python
"""ReMIX-C training curves (validation metrics vs step) for the paper.

  python scripts/remixc_curves.py            # fetch from wandb, cache, plot
  python scripts/remixc_curves.py --offline  # plot from the cached JSON only

Main text (remixc_curve_*): the run reported in the results table, with the
instruction-shuffled and seed-only references on the R@10 panel.
Appendix (remixc_overfit_*): the same metrics for runs on the filtered data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import GREY, THIRD, apply, savefig  # noqa: E402

ENTITY_PROJECT = "jul-guinot/remix-C"
MAIN = "ohdh0i3m"   # small, 2-layer fusion, all train variants
RUNS = {  # run id -> (legend label, colour)
    "ohdh0i3m": ("All data", "#CC79A7"),
    "idpc4krv": ("Filtered", "#0072B2"),
    "fvv04uaf": ("Filtered, 4 layers", "#E69F00"),
}
KEYS = ["trainer/global_step", "val/R@10", "val/MRR", "val/loss", "val/R@10_shuffled", "val/R@10_seed_nn"]
CACHE = Path("results/remix_c/curves.json")


def fetch() -> dict:
    import wandb
    api = wandb.Api()
    out = {}
    for rid in RUNS:
        h = api.run(f"{ENTITY_PROJECT}/{rid}").history(keys=KEYS, pandas=True)
        out[rid] = {k: h[k].tolist() for k in KEYS if k in h}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out))
    return out


def _panel(name, ylabel, series, figdir, refs=()):
    fig, ax = plt.subplots(figsize=THIRD)
    for label, color, steps, ys, ls in list(series) + list(refs):
        ax.plot([s / 1000 for s in steps], ys, color=color, lw=1.1, ls=ls, label=label,
                marker="o" if ls == "-" else None, markersize=2)
    ax.set_xlabel("Training step (k)")
    ax.set_ylabel(ylabel)
    if "loss" in name and len(series) == 1:      # single run: avoid a zoom that makes noise look like a trend
        ys = series[0][3]
        ax.set_ylim(min(ys) - 0.3, max(ys) + 0.3)
    if name.endswith("_r10"):                   # the three panels share one legend
        ax.legend(loc="lower right", handlelength=1.4)
    savefig(fig, figdir / f"{name}.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--figdir", default="paper/figures")
    a = ap.parse_args()
    apply()
    data = json.loads(CACHE.read_text()) if a.offline else fetch()
    figdir = Path(a.figdir)
    step = "trainer/global_step"

    m = data[MAIN]
    label, color = RUNS[MAIN]
    refs = [("Shuffled instruction", GREY, m[step], m["val/R@10_shuffled"], "--"),
            ("Seed only", "#444444", m[step], m["val/R@10_seed_nn"], ":")]
    _panel("remixc_curve_r10", "Validation R@10", [("ReMIX-C", color, m[step], m["val/R@10"], "-")], figdir, refs)
    _panel("remixc_curve_mrr", "Validation MRR", [("ReMIX-C", color, m[step], m["val/MRR"], "-")], figdir)
    _panel("remixc_curve_loss", "Validation loss", [("ReMIX-C", color, m[step][1:], m["val/loss"][1:], "-")], figdir)

    for key, name, ylabel in [("val/R@10", "remixc_overfit_r10", "Validation R@10"),
                              ("val/MRR", "remixc_overfit_mrr", "Validation MRR"),
                              ("val/loss", "remixc_overfit_loss", "Validation loss")]:
        skip = 1 if key == "val/loss" else 0      # step-0 loss of the untrained model dwarfs the rest
        series = [(RUNS[r][0], RUNS[r][1], data[r][step][skip:], data[r][key][skip:], "-") for r in RUNS]
        _panel(name, ylabel, series, figdir)
    print(f"curves -> {figdir}/remixc_curve_*.pdf, remixc_overfit_*.pdf")


if __name__ == "__main__":
    main()
