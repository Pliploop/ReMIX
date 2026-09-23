#!/usr/bin/env python
"""Analyse the ReMIX-B relevance pool: stats, paper figures, qualitative examples.

Reads a dataset's chain_step_relevance_pools.shard*.jsonl (or a merged file),
prints summary stats, writes figures to paper/figures/, and pulls qualitative
example steps (instruction + source/target + candidates across grades) to a
Markdown file for the paper appendix.

  python scripts/relevance_pool_analysis.py --dataset music4all
  python scripts/relevance_pool_analysis.py --dataset mtg_jamendo --examples 8
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO = Path(__file__).resolve().parents[1]
FIG_DIR = REPO / "paper" / "figures"

DATASETS = {
    "music4all": ("Music4All", "/gpfs/scratch/acw749/datasets/music4all_instruct/music4all_v1"),
    "mtg_jamendo": ("MTG-Jamendo", "/gpfs/scratch/acw749/datasets/mtg_jamendo_instruct/v1"),
}
FOLDER = "instructions_axis_focused_5"
# Judge grades are 0-5 (Type_TARGET=5 exact, strong=4, [good=3 unused], partial=2,
# near-miss=1, non-relevant=0). Keep all six so grade 5 is never silently dropped.
GRADE_LABEL = {5: "exact", 4: "strong", 3: "good", 2: "partial", 1: "near-miss", 0: "negative"}
# Match the site/paper palette (theme.js STAGE colours), warm->cool by relevance.
GRADE_COLOR = {5: "#137539", 4: "#1FA347", 3: "#7BC043", 2: "#FB8B24", 1: "#E2843B", 0: "#C3C7CD"}
GRADES = (5, 4, 3, 2, 1, 0)


def _iter_pool(root: str) -> Iterable[Dict[str, Any]]:
    files = sorted(glob.glob(str(Path(root) / FOLDER / "relevance_pool" / "chain_step_relevance_pools.shard*.jsonl")))
    if not files:
        merged = Path(root) / FOLDER / "relevance_pool" / "chain_step_relevance_pools.jsonl"
        files = [str(merged)] if merged.is_file() else []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _instructions_for(root: str, keys, field: str = "history_unaware_instruction") -> Dict[Any, str]:
    """Pull instruction text for the given (chain_id, turn_index) steps from the gate.
    Pool records don't carry the verbalized instruction; the validation gate does."""
    want = set(keys)
    out: Dict[Any, str] = {}
    gate = Path(root) / FOLDER / "validation" / "validated_instructions.jsonl"
    if not gate.is_file():
        return out
    with open(gate, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            k = (r.get("chain_id"), r.get("turn_index"))
            if k in want and k not in out:
                out[k] = r.get(field) or r.get("history_aware_instruction") or ""
                if len(out) == len(want):
                    break
    return out


def _fig(name: str, plotter) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    plotter(ax)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name)
    plt.close(fig)
    print(f"  figure -> paper/figures/{name}")


def analyse(label: str, root: str, n_examples: int) -> None:
    grade = Counter()
    pool_type = Counter()
    per_step_positives = []          # candidates with grade >= 2 per step
    per_step_candidates = []
    grade_by_axis = defaultdict(Counter)
    label_changes = 0
    judged = 0
    examples: List[Dict[str, Any]] = []
    steps = 0

    for row in _iter_pool(root):
        steps += 1
        cands = row.get("candidates") or []
        per_step_candidates.append(len(cands))
        pos = 0
        axes = (row.get("instruction_plan") or {}).get("inferred_change_axes") or (row.get("semantic_constraints") or {}).get("change_axes") or []
        axis = str(axes[0]) if axes else "unknown"
        for c in cands:
            g = int(c.get("grade", 0) or 0)
            grade[g] += 1
            pool_type[str(c.get("pool_type", "") or "")] += 1
            grade_by_axis[axis][g] += 1
            if g >= 2:
                pos += 1
            if c.get("label_source") == "llm_judge":
                judged += 1
                if c.get("candidate_llm_judge") and int(c["candidate_llm_judge"].get("grade", g) or g) != g:
                    label_changes += 1  # (grade already overwritten; kept for future raw diffs)
        per_step_positives.append(pos)
        # keep a few well-graded steps as qualitative examples
        if len(examples) < n_examples and pos >= 1 and any(int(c.get("grade", 0) or 0) == 0 for c in cands):
            examples.append(row)

    if steps == 0:
        print(f"{label}: no pool records yet.")
        return

    total_c = sum(grade.values())
    print(f"\n=== {label}: {steps:,} steps, {total_c:,} candidates ({total_c/steps:.1f}/step), {judged:,} llm-judged")
    print("  grade distribution:")
    for g in GRADES:
        n = grade.get(g, 0)
        print(f"    {g} {GRADE_LABEL[g]:9} {n:>8,} ({100*n/total_c:.1f}%)")
    print("  pool types:", dict(pool_type.most_common()))
    import statistics as st
    print(f"  positives (grade>=2)/step: mean {st.mean(per_step_positives):.1f}, median {int(st.median(per_step_positives))}")

    # ---- figures ----
    slug = label.lower().replace("-", "_").replace(" ", "_")

    def _grade_bar(ax):
        gs = list(GRADES)
        ax.bar([GRADE_LABEL[g] for g in gs], [grade.get(g, 0) for g in gs],
               color=[GRADE_COLOR[g] for g in gs], edgecolor="black", linewidth=0.6)
        ax.set_ylabel("candidates"); ax.set_xlabel("relevance grade")
        ax.tick_params(axis="x", rotation=30)
    _fig(f"{slug}_relpool_grade_dist.pdf", _grade_bar)

    def _pos_hist(ax):
        import numpy as np
        ax.hist(per_step_positives, bins=range(0, max(per_step_positives) + 2), color="#2E6FD6", edgecolor="black", linewidth=0.5)
        ax.set_xlabel("relevant candidates (grade>=2) per query"); ax.set_ylabel("queries")
    _fig(f"{slug}_relpool_positives_per_query.pdf", _pos_hist)

    def _axis_bar(ax):
        top_axes = [a for a, _ in Counter({a: sum(c.values()) for a, c in grade_by_axis.items()}).most_common(6)]
        import numpy as np
        x = np.arange(len(top_axes)); w = 0.8 / len(GRADES); mid = (len(GRADES) - 1) / 2
        for i, g in enumerate(GRADES):
            vals = [grade_by_axis[a].get(g, 0) / max(1, sum(grade_by_axis[a].values())) for a in top_axes]
            ax.bar(x + (i - mid) * w, vals, w, label=GRADE_LABEL[g], color=GRADE_COLOR[g], edgecolor="black", linewidth=0.4)
        ax.set_xticks(x); ax.set_xticklabels(top_axes, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("grade share"); ax.legend(fontsize=6, ncol=len(GRADES), loc="upper center")
    _fig(f"{slug}_relpool_grade_by_axis.pdf", _axis_bar)

    # ---- qualitative examples ----
    ex_path = REPO / "paper" / f"relpool_examples_{slug}.md"
    ex_path.parent.mkdir(parents=True, exist_ok=True)
    instr_map = _instructions_for(root, [(r.get("chain_id"), r.get("turn_index")) for r in examples])
    with ex_path.open("w", encoding="utf-8") as f:
        f.write(f"# ReMIX-B relevance-pool examples — {label}\n\n")
        for row in examples:
            instr = instr_map.get((row.get("chain_id"), row.get("turn_index")), "") \
                or (row.get("semantic_delta_verbalized", {}) or {}).get("instruction", "") or row.get("instruction", "")
            f.write(f"## chain {row.get('chain_id')} turn {row.get('turn_index')}\n")
            f.write(f"- **source**: `{row.get('source_clip_id')}`  →  **target**: `{row.get('target_clip_id')}`\n")
            f.write(f"- **instruction**: {instr}\n\n")
            f.write("| grade | pool type | clip | judge reason |\n|---|---|---|---|\n")
            cands = sorted(row.get("candidates") or [], key=lambda c: -int(c.get("grade", 0) or 0))
            shown = cands[:3] + [c for c in cands if int(c.get("grade", 0) or 0) == 0][:2]
            for c in shown:
                reason = str(c.get("judge_reason", "") or "")[:100].replace("\n", " ")
                f.write(f"| {c.get('grade')} | {c.get('pool_type')} | `{c.get('clip_id')}` | {reason} |\n")
            f.write("\n")
    print(f"  examples -> {ex_path.relative_to(REPO)} ({len(examples)} steps)")

    # ---- LaTeX fragment for the appendix (\input'd; regenerates in sync) ----
    def _tex(s: str) -> str:
        for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                     ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                     ("^", r"\textasciicircum{}")):
            s = s.replace(a, b)
        return s
    tex_path = REPO / "paper" / f"relpool_examples_{slug}.tex"
    with tex_path.open("w", encoding="utf-8") as f:
        f.write(f"% Auto-generated by scripts/relevance_pool_analysis.py --dataset {slug}\n")
        for row in examples[:3]:
            instr = instr_map.get((row.get("chain_id"), row.get("turn_index")), "")
            if not instr:
                continue
            cands = sorted(row.get("candidates") or [], key=lambda c: -int(c.get("grade", 0) or 0))
            shown = cands[:3] + [c for c in cands if int(c.get("grade", 0) or 0) == 0][:2]
            f.write("\\smallskip\\noindent\\textbf{Instruction:} ``" + _tex(instr) + "''\\\\\n")
            f.write("\\begin{tabular}{@{}rll@{}}\\toprule\ngrade & pool type & clip \\\\\\midrule\n")
            for c in shown:
                f.write(f"{c.get('grade')} & {_tex(str(c.get('pool_type','')))} & \\texttt{{{_tex(str(c.get('clip_id','')))}}} \\\\\n")
            f.write("\\bottomrule\\end{tabular}\n\n")
    print(f"  latex    -> {tex_path.relative_to(REPO)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(DATASETS) + ["all"], default="all")
    ap.add_argument("--examples", type=int, default=6)
    args = ap.parse_args()
    keys = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for k in keys:
        label, root = DATASETS[k]
        analyse(label, root, args.examples)


if __name__ == "__main__":
    main()
