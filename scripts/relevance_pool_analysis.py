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
WRITE_EXAMPLES = True  # set False on preview runs (--out-dir) so paper example files aren't touched
# v2 grading (docs/benchmark_grading_v2.md): contiguous 0-6 quality scale.
GRADE_LABEL = {6: "Exact", 5: "Strong", 4: "Good", 3: "Partial", 2: "Soft-fail", 1: "Hard-fail", 0: "Non-rel."}
GRADES = (6, 5, 4, 3, 2, 1, 0)
POSITIVE_MIN = 3  # grade >= 3 (partial or better) counts as a relevant positive
POS_GRADES = (6, 5, 4, 3)  # relevant grades (grade 0-2 dwarf them in raw counts)

# Publication style shared with scripts/paper_data_stats.py (Okabe-Ito, colorblind-safe).
BLUE, ORANGE, GREEN, VERM, PURPLE, SKY, YELLOW, GREY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#999999",
)
BAR = dict(alpha=0.85, edgecolor="black", linewidth=0.7)
# Grade ramp: bright, saturated green (relevant) -> amber/orange -> red -> grey (non-rel).
GRADE_COLOR = {6: "#0A7D3E", 5: "#22C55E", 4: "#A7E32C", 3: "#F7B500", 2: "#FB6A0A", 1: "#E4231B", 0: "#CBD0D6"}
WEDGE_EDGE = dict(edgecolor="#2A2A2A", linewidth=1.0)
FAILURE_MODE_LABEL = {
    "change_missing": "change missing", "change_partial": "change partial",
    "change_overshoot": "overshoot", "preservation_violated": "preservation broken",
    "source_incompatible": "source incompatible", "caption_semantics_miss": "caption miss",
    "surface_only": "surface only", "history_shortcut": "history shortcut",
    "history_dependent_satisfied": "history-satisfied", "near_duplicate": "near-duplicate",
    "off_topic": "off-topic",
}
# Prettier axis / pool-type labels.
AXIS_LABEL = {
    "genre_style": "Genre / style", "texture_production": "Texture / prod.",
    "energy": "Energy", "instrumentation": "Instrumentation", "mood": "Mood",
    "vocals": "Vocals", "tempo": "Tempo", "harmony": "Harmony",
    "structure": "Structure", "other": "Other", "unknown": "Unknown",
}
PTYPE_LABEL = {
    "Type_TARGET": "Target", "Type_STRONG": "Strong", "Type_PARTIAL": "Partial",
    "Type_HARD_NEG": "Hard neg.", "Type_T": "Tag-only", "Type_H": "History",
}


def _setup_style() -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11, "axes.labelsize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 9, "legend.frameon": False, "legend.handlelength": 1.3,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": "#dddddd", "grid.linewidth": 0.6, "figure.dpi": 150,
    })


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


def _fig(name: str, plotter, size=(4.6, 3.2)) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _setup_style()
    fig, ax = plt.subplots(figsize=size, constrained_layout=True)
    plotter(ax)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name)
    plt.close(fig)
    print(f"  figure -> paper/figures/{name}")


def _write_pool_pie(slug: str, pool_frac: Dict[int, float], steps: int) -> None:
    """Average per-step pool composition: a main pie dominated by Non-rel, with the
    relevant + failure tail (grades 1-6) boxed on the pie and zoomed into a proper
    bar-chart inset (spines, ticks, grid) whose bars are the share of that tail."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    _setup_style()
    order = [0, 1, 2, 3, 4, 5, 6]          # Non-rel first; grades 1-6 contiguous after
    tail = [1, 2, 3, 4, 5, 6]              # bar chart, Hard-fail .. Exact
    means = {g: pool_frac.get(g, 0.0) / max(1, steps) for g in order}
    rest_total = sum(means[g] for g in tail) or 1.0

    fig = plt.figure(figsize=(9.6, 4.9))
    ax1 = fig.add_axes([0.00, 0.20, 0.50, 0.76]); ax1.set_aspect("equal")
    w1, _, _ = ax1.pie(
        [means[g] for g in order], colors=[GRADE_COLOR[g] for g in order],
        startangle=90, counterclock=False, explode=[0.0] + [0.06] * 6,
        wedgeprops=WEDGE_EDGE, autopct=lambda p: f"{p:.0f}%" if p > 50 else "",
        pctdistance=0.62, textprops=dict(fontsize=11, color="#222222", fontweight="bold"),
    )

    # bar-chart inset (real axes: spines, ticks, grid) to the right of the pie
    axins = ax1.inset_axes([1.16, 0.06, 0.80, 0.88])
    vals = [means[g] / rest_total * 100 for g in tail]
    axins.bar(range(len(tail)), vals, color=[GRADE_COLOR[g] for g in tail], **WEDGE_EDGE)
    axins.set_xticks(range(len(tail))); axins.set_xticklabels([GRADE_LABEL[g] for g in tail], rotation=25, ha="right")
    axins.set_ylabel("Share of tail (%)")
    axins.set_title(f"Relevant + failure tail  ({rest_total*100:.1f}% of pool)", fontsize=10)
    axins.grid(axis="y", alpha=0.7); axins.grid(axis="x", visible=False)
    for sp in ("left", "bottom"):
        axins.spines[sp].set_visible(True)
    for i, v in enumerate(vals):
        axins.text(i, v + max(vals) * 0.02, f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    # box the source region (tail wedges) on the pie + leader lines to the inset
    pts = []
    for wg in w1[1:]:
        for th in np.linspace(wg.theta1, wg.theta2, 6):
            pts.append((wg.center[0] + wg.r * np.cos(np.deg2rad(th)),
                        wg.center[1] + wg.r * np.sin(np.deg2rad(th))))
        pts.append(tuple(wg.center))
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    pad = 0.06
    bounds = [min(xs) - pad, min(ys) - pad, (max(xs) - min(xs)) + 2 * pad, (max(ys) - min(ys)) + 2 * pad]
    ax1.indicate_inset(bounds, inset_ax=axins, edgecolor="#333333", linewidth=1.2, alpha=0.9)

    # framed legend below, grade + share of the whole pool
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=GRADE_COLOR[g], edgecolor="#2A2A2A", linewidth=0.6) for g in order]
    labels = [f"{GRADE_LABEL[g]} ({means[g]*100:.1f}%)" for g in order]
    leg = fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
                     ncol=7, frameon=True, fontsize=8.5, handlelength=1.1, columnspacing=1.2,
                     title="Grade (share of pool)", borderpad=0.7)
    leg.get_frame().set_edgecolor("#9aa0a6"); leg.get_frame().set_linewidth(0.8)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{slug}_relpool_pool_composition_pie.pdf")
    plt.close(fig)
    print(f"  figure -> paper/figures/{slug}_relpool_pool_composition_pie.pdf")


def analyse(label: str, root: str, n_examples: int) -> None:
    grade = Counter()
    pool_type = Counter()
    per_step_positives = []          # candidates with grade >= 2 per step
    per_step_candidates = []
    grade_by_axis = defaultdict(Counter)
    prov = defaultdict(Counter)            # candidate source -> grade counts
    ptype_grade = defaultdict(Counter)     # heuristic pool_type -> verified grade
    sim_by_grade = defaultdict(lambda: {"audio": [], "caption": []})
    exact_grade = Counter()                # grade the exact target receives
    steps_with_exact = 0
    SIM_CAP = 40000
    label_changes = 0
    judged = 0
    fmodes = Counter()                     # v2 failure-mode taxonomy tallies
    lsrc = Counter()                       # label_source (llm_judge / similarity_gated / deterministic)
    quality = Counter()                    # judge quality enum
    pool_frac = defaultdict(float)         # sum over steps of each grade's within-pool fraction
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
            pt = str(c.get("pool_type", "") or "")
            pool_type[pt] += 1
            grade_by_axis[axis][g] += 1
            ptype_grade[pt][g] += 1
            for s in (c.get("candidate_sources") or ["unknown"]):
                prov[str(s)][g] += 1
            # Exclude the injected designated target: it is trivially self-similar
            # (audio_sim_to_target = 1.0), which would swamp the strong grade.
            if not c.get("is_exact_target"):
                asim, csim = c.get("audio_sim_to_target"), c.get("caption_sim_to_target")
                sg = sim_by_grade[g]
                if isinstance(asim, (int, float)) and len(sg["audio"]) < SIM_CAP:
                    sg["audio"].append(float(asim))
                if isinstance(csim, (int, float)) and len(sg["caption"]) < SIM_CAP:
                    sg["caption"].append(float(csim))
            if g >= POSITIVE_MIN:
                pos += 1
            lsrc[str(c.get("label_source") or "")] += 1
            for m in (c.get("failure_modes") or []):
                fmodes[str(m)] += 1
            j = c.get("candidate_llm_judge") or {}
            if c.get("label_source") == "llm_judge":
                judged += 1
                if j.get("quality"):
                    quality[str(j.get("quality"))] += 1
        ex = [int(c.get("grade", 0) or 0) for c in cands if c.get("is_exact_target")]
        if ex:
            steps_with_exact += 1
            exact_grade[max(ex)] += 1
        if cands:  # within-pool grade composition for this step (averaged later)
            step_grade = Counter(int(c.get("grade", 0) or 0) for c in cands)
            for gg, nn in step_grade.items():
                pool_frac[gg] += nn / len(cands)
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
    print(f"  positives (grade>={POSITIVE_MIN})/step: mean {st.mean(per_step_positives):.1f}, median {int(st.median(per_step_positives))}")
    print(f"  label_source: {dict(lsrc)}")
    print(f"  judge quality: {dict(quality)}")
    print(f"  failure_modes: {dict(fmodes.most_common())}")

    # ---- figures ----
    slug = label.lower().replace("-", "_").replace(" ", "_")

    from matplotlib.ticker import FuncFormatter, PercentFormatter
    _kfmt = FuncFormatter(lambda v, _: f"{v:,.0f}")
    _POS_LEG = dict(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16),
                    columnspacing=1.0, handletextpad=0.4, handlelength=1.1)

    def _grade_bar(ax):
        gs = list(GRADES)
        ax.bar([GRADE_LABEL[g] for g in gs], [grade.get(g, 0) for g in gs],
               color=[GRADE_COLOR[g] for g in gs], **BAR)
        ax.set_ylabel("Candidates"); ax.set_yscale("log")
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(axis="x", visible=False)
    _fig(f"{slug}_relpool_grade_dist.pdf", _grade_bar)

    def _pos_hist(ax):
        m = max(per_step_positives)
        ax.hist(per_step_positives, bins=range(0, m + 2), color=BLUE, **BAR)
        ax.set_xlabel(r"Relevant candidates per query ($\geq$ partial)"); ax.set_ylabel("Queries")
        ax.yaxis.set_major_formatter(_kfmt); ax.grid(axis="x", visible=False)
    _fig(f"{slug}_relpool_positives_per_query.pdf", _pos_hist)

    def _axis_bar(ax):
        import numpy as np
        top_axes = [a for a, _ in Counter({a: sum(c.values()) for a, c in grade_by_axis.items()}).most_common(6)]
        x = np.arange(len(top_axes)); bottoms = np.zeros(len(top_axes))
        for g in POS_GRADES:  # relevant grades only; grade 0-2 would flatten the differences
            vals = np.array([grade_by_axis[a].get(g, 0) / max(1, sum(grade_by_axis[a].values())) for a in top_axes])
            ax.bar(x, vals, bottom=bottoms, color=GRADE_COLOR[g], label=GRADE_LABEL[g], edgecolor="white", linewidth=0.5)
            bottoms += vals
        ax.set_xticks(x); ax.set_xticklabels([AXIS_LABEL.get(a, a.replace("_", " ").title()) for a in top_axes],
                                             rotation=20, ha="right")
        ax.set_ylabel("Share of candidates"); ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.legend(**_POS_LEG); ax.grid(axis="x", visible=False)
    _fig(f"{slug}_relpool_grade_by_axis.pdf", _axis_bar)

    # ---- (1) grade x candidate provenance ----
    SRC_ORDER = ["target_neighborhood", "source_neighborhood", "seed_neighborhood",
                 "history_reference_neighborhood", "chain_history_target", "exact_target"]
    SRC_LABEL = {"target_neighborhood": "Target", "source_neighborhood": "Source",
                 "seed_neighborhood": "Seed", "history_reference_neighborhood": "History",
                 "chain_history_target": "Hist-tgt", "exact_target": "Exact"}

    def _prov_bar(ax):
        import numpy as np
        srcs = [s for s in SRC_ORDER if s in prov] + [s for s in prov if s not in SRC_ORDER]
        srcs = srcs[:6]
        bottoms = np.zeros(len(srcs))
        for g in POS_GRADES:
            vals = np.array([prov[s].get(g, 0) for s in srcs], float)
            ax.bar(range(len(srcs)), vals, bottom=bottoms, color=GRADE_COLOR[g],
                   label=GRADE_LABEL[g], edgecolor="white", linewidth=0.5)
            bottoms += vals
        ax.set_xticks(range(len(srcs))); ax.set_xticklabels([SRC_LABEL.get(s, s) for s in srcs], rotation=20, ha="right")
        ax.set_ylabel("Relevant candidates"); ax.yaxis.set_major_formatter(_kfmt)
        ax.legend(**_POS_LEG); ax.grid(axis="x", visible=False)
    _fig(f"{slug}_relpool_grade_by_source.pdf", _prov_bar)

    # ---- (2) similarity vs grade ----
    def _sim_box(ax):
        import numpy as np
        gs = [g for g in GRADES if sim_by_grade[g]["audio"]]
        x = np.arange(len(gs))
        for off, key, col in ((-0.2, "audio", BLUE), (0.2, "caption", ORANGE)):
            bp = ax.boxplot([sim_by_grade[g][key] for g in gs], positions=x + off, widths=0.36,
                            showfliers=False, patch_artist=True, whis=(5, 95),
                            medianprops=dict(color="black", linewidth=1.2),
                            whiskerprops=dict(color="#555555"), capprops=dict(color="#555555"))
            for b in bp["boxes"]:
                b.set(facecolor=col, alpha=0.85, edgecolor="black", linewidth=0.7)
        ax.set_xticks(x); ax.set_xticklabels([GRADE_LABEL[g] for g in gs], rotation=20, ha="right")
        ax.set_ylabel("Similarity to target"); ax.set_xlabel("Verified grade"); ax.set_ylim(0, 1.02)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor=BLUE, label="Audio"), Patch(facecolor=ORANGE, label="Caption")],
                  loc="upper right", ncol=2, columnspacing=1.0)
        ax.grid(axis="x", visible=False)
    _fig(f"{slug}_relpool_sim_by_grade.pdf", _sim_box)

    # ---- (3) target recoverability ----
    def _target_bar(ax):
        gs = [g for g in GRADES if exact_grade.get(g, 0)]
        ax.bar([GRADE_LABEL[g] for g in gs], [exact_grade[g] for g in gs],
               color=[GRADE_COLOR[g] for g in gs], **BAR)
        ax.set_ylabel("Steps"); ax.set_xlabel("Grade of the designated target")
        ax.yaxis.set_major_formatter(_kfmt)
        ax.set_title(f"Present in {100*steps_with_exact/steps:.1f}% of steps", fontsize=10)
        ax.grid(axis="x", visible=False)
    _fig(f"{slug}_relpool_target_recovery.pdf", _target_bar)

    # ---- (4) judge vs heuristic ----
    def _heur(ax):
        import numpy as np
        pts = [p for p, _ in sorted(ptype_grade.items(), key=lambda kv: -sum(kv[1].values())) if p][:6]
        M = np.array([[ptype_grade[p].get(g, 0) for g in GRADES] for p in pts], float)
        Mn = M / np.clip(M.sum(1, keepdims=True), 1, None)
        im = ax.imshow(Mn, cmap="Blues", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(GRADES))); ax.set_xticklabels([GRADE_LABEL[g] for g in GRADES], rotation=20, ha="right")
        ax.set_yticks(range(len(pts))); ax.set_yticklabels([PTYPE_LABEL.get(p, p) for p in pts])
        ax.set_xlabel("LLM-verified grade"); ax.set_ylabel("Heuristic pool type"); ax.grid(False)
        for i in range(len(pts)):
            for j in range(len(GRADES)):
                if Mn[i, j] >= 0.01:
                    ax.text(j, i, f"{Mn[i, j]*100:.0f}", ha="center", va="center",
                            fontsize=9, color="white" if Mn[i, j] > 0.5 else "#222222")
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0)); cb.outline.set_visible(False)
    _fig(f"{slug}_relpool_judge_vs_heuristic.pdf", _heur, size=(5.2, 3.2))

    # ---- (5) failure-mode taxonomy ----
    def _fmode_bar(ax):
        top = [(m, n) for m, n in fmodes.most_common() if m != "off_topic"][:9]
        top = top[::-1]  # largest ends at the top in barh
        ax.barh([FAILURE_MODE_LABEL.get(m, m) for m, _ in top], [n for _, n in top], color=VERM, **BAR)
        ax.set_xlabel("Candidates flagged"); ax.xaxis.set_major_formatter(_kfmt)
        ax.grid(axis="y", visible=False)
    if any(m != "off_topic" for m in fmodes):
        _fig(f"{slug}_relpool_failure_modes.pdf", _fmode_bar, size=(4.8, 3.2))

    # ---- (6) average per-step pool composition: bar-of-pie zoomed on the relevant tail ----
    _write_pool_pie(slug, pool_frac, steps)

    # ---- extra stats for the paper text ----
    print(f"  target recoverable (exact target in pool): {steps_with_exact:,}/{steps:,} ({100*steps_with_exact/steps:.1f}%)")
    hn = ptype_grade.get("Type_HARD_NEG", Counter())
    hn_tot = sum(hn.values())
    if hn_tot:
        up = sum(hn.get(g, 0) for g in POS_GRADES)
        print(f"  heuristic hard-negs upgraded to >=partial by judge: {up:,}/{hn_tot:,} ({100*up/hn_tot:.1f}%)")
    pos_src = {SRC_LABEL.get(s, s): sum(prov[s].get(g, 0) for g in POS_GRADES) for s in prov}
    print("  positives (>=partial) by source:", dict(sorted(pos_src.items(), key=lambda kv: -kv[1])))

    # ---- qualitative examples ----
    if not WRITE_EXAMPLES:
        return  # preview run: figures + stats only, don't touch the paper's example files
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
    ap.add_argument("--out-dir", default=None, help="Figure output dir (default: paper/figures). Use a scratch dir for previews.")
    args = ap.parse_args()
    if args.out_dir:
        global FIG_DIR, WRITE_EXAMPLES
        FIG_DIR = Path(args.out_dir).expanduser().resolve()
        WRITE_EXAMPLES = False
    keys = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for k in keys:
        label, root = DATASETS[k]
        analyse(label, root, args.examples)


if __name__ == "__main__":
    main()
