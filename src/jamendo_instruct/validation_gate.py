"""Instruction-validity grading + benchmark selection.

Two decoupled phases (see ``docs/validation_gate.md``):

1. **grade** (`grade_records`) — record, per *variant*, the graded rubric score
   and a derived accept flag, plus per-step and per-chain summaries. **Nothing is
   cut.** This is the annotation artifact (`instruction_grades.jsonl`).

2. **select** (`select_chain_variants`) — the *final* cut. Per step, pick the
   best variant that passes (variant fallback: if variant 0 fails, try the next),
   and for the contextual track truncate a chain at the first step where **no**
   variant passes. Emits one record per step in the schema ``relevance_pool``
   consumes (`validated_instructions.jsonl`, `validation.accepted`).

This separation lets us assemble a valid chain by *choosing* a passing variant
per step rather than dropping the step, and keeps every grade on disk so cutting
policy can change without re-judging.

Distinct from ``relevance_pool`` (which grades *candidate tracks* for a query —
the benchmark ground truth). This module is about *instruction* validity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

_OVERALL_QUESTION = "overall_validity"
_CHAIN_QUESTION = "chain_coherence"  # optional future rubric dimension

ContextualPolicy = str  # "truncate" | "drop"
ChainAggregate = str  # "min" | "mean"
VariantSelect = str  # "best" | "first"


@dataclass(frozen=True)
class GateConfig:
    accept_threshold: float = 4.0
    chain_aggregate: ChainAggregate = "min"
    contextual_policy: ContextualPolicy = "truncate"
    variant_select: VariantSelect = "best"
    instruction_field: str = "history_unaware_instruction"
    gate_name: str = "graded_v1"


def _vkey(record: Dict[str, Any]) -> Tuple[str, int, int]:
    return (str(record.get("chain_id", "") or ""), int(record.get("turn_index", 0) or 0),
            int(record.get("variant_index", 0) or 0))


def _skey(record: Dict[str, Any]) -> Tuple[str, int]:
    return (str(record.get("chain_id", "") or ""), int(record.get("turn_index", 0) or 0))


def _score_of(rating: Dict[str, Any] | None, question: str) -> float | None:
    if not rating:
        return None
    ans = dict((rating.get("answers", {}) or {}).get(question, {}) or {})
    s = ans.get("score")
    return float(s) if isinstance(s, (int, float)) else None


def _question_scores(rating: Dict[str, Any] | None) -> Dict[str, Any]:
    if not rating:
        return {}
    return {str(q): dict(a or {}).get("score") for q, a in (rating.get("answers", {}) or {}).items()}


def _dedupe_ratings(ratings: Iterable[Dict[str, Any]], field: str) -> Dict[Tuple[str, int, int], Dict[str, Any]]:
    index: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    for rating in ratings:
        rfield = str(rating.get("instruction_field", "") or "")
        if rfield and field and rfield != field:
            continue
        index[_vkey(rating)] = dict(rating)
    return index


# --------------------------------------------------------------------------- #
# Phase 1: grade (annotate everything, cut nothing)
# --------------------------------------------------------------------------- #
def grade_records(
    instruction_records: Sequence[Dict[str, Any]],
    ratings: Iterable[Dict[str, Any]],
    config: GateConfig = GateConfig(),
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rate_index = _dedupe_ratings(ratings, config.instruction_field)

    # Per-step: which variants pass, best variant, does any pass.
    step_variants: Dict[Tuple[str, int], List[Tuple[int, float]]] = {}
    for record in instruction_records:
        score = _score_of(rate_index.get(_vkey(record)), _OVERALL_QUESTION)
        if score is None:
            continue
        step_variants.setdefault(_skey(record), []).append((int(record.get("variant_index", 0) or 0), score))

    step_summary: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for sk, variants in step_variants.items():
        passing = sorted([(v, s) for v, s in variants if s >= config.accept_threshold],
                         key=lambda vs: (-vs[1], vs[0]))
        best = passing[0] if passing else max(variants, key=lambda vs: vs[1])
        step_summary[sk] = {
            "passing_variants": [v for v, _ in passing],
            "best_variant": best[0],
            "best_score": best[1],
            "has_passing_variant": bool(passing),
        }

    def step_ok(sk: Tuple[str, int]) -> bool:
        return step_summary.get(sk, {}).get("has_passing_variant", False)

    # Per-chain coherence prefix (based on has_passing_variant per step).
    chain_turns: Dict[str, List[int]] = {}
    for record in instruction_records:
        cid, turn = str(record.get("chain_id", "") or ""), int(record.get("turn_index", 0) or 0)
        chain_turns.setdefault(cid, [])
        if turn not in chain_turns[cid]:
            chain_turns[cid].append(turn)
    chain_info: Dict[str, Dict[str, Any]] = {}
    for cid, turns in chain_turns.items():
        turns_sorted = sorted(turns)
        prefix_end, truncate_at = -1, None
        for t in turns_sorted:
            if step_ok((cid, t)):
                if truncate_at is None:
                    prefix_end = t
            elif truncate_at is None:
                truncate_at = t
        scored = [step_summary[(cid, t)]["best_score"] for t in turns_sorted if (cid, t) in step_summary]
        chain_info[cid] = {
            "prefix_end": prefix_end,
            "truncate_at": truncate_at,
            "all_steps_valid": truncate_at is None and all(step_ok((cid, t)) for t in turns_sorted),
            "score": (min(scored) if config.chain_aggregate == "min" else sum(scored) / len(scored)) if scored else None,
        }

    out: List[Dict[str, Any]] = []
    counts = {"total": 0, "rated": 0, "variant_accepted": 0, "steps": len(step_summary),
              "steps_with_passing_variant": sum(1 for s in step_summary.values() if s["has_passing_variant"])}
    for record in instruction_records:
        counts["total"] += 1
        sk = _skey(record)
        rating = rate_index.get(_vkey(record))
        score = _score_of(rating, _OVERALL_QUESTION)
        rated = score is not None
        counts["rated"] += int(rated)
        variant_accepted = bool(rated and score >= config.accept_threshold)
        counts["variant_accepted"] += int(variant_accepted)
        cid = sk[0]
        cinfo = chain_info.get(cid, {})
        within_prefix = int(record.get("turn_index", 0) or 0) <= cinfo.get("prefix_end", -1)

        merged = dict(record)
        merged["validation"] = {
            "gate": config.gate_name,
            "phase": "grades",
            "rated": rated,
            "threshold": config.accept_threshold,
            "overall_score": score,
            "variant_accepted": variant_accepted,        # is THIS variant valid
            "chain_coherence_score": _score_of(rating, _CHAIN_QUESTION),
            "question_scores": _question_scores(rating),
            "step": step_summary.get(sk, {"passing_variants": [], "has_passing_variant": False}),
            "chain": {
                "score": cinfo.get("score"),
                "aggregate": config.chain_aggregate,
                "all_steps_valid": cinfo.get("all_steps_valid", False),
                "truncate_at": cinfo.get("truncate_at"),
                "within_coherent_prefix": bool(within_prefix),
            },
        }
        out.append(merged)

    report = {"phase": "grades", "config": _cfg(config), "counts": counts, "chains": len(chain_turns)}
    return out, report


# --------------------------------------------------------------------------- #
# Phase 2: select (the final cut, with variant fallback)
# --------------------------------------------------------------------------- #
def select_chain_variants(
    graded_records: Sequence[Dict[str, Any]],
    config: GateConfig = GateConfig(),
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """One record per step: the chosen (best-passing) variant, with a benchmark
    accept flag. `validation.accepted` = the step has a passing variant; the
    contextual track is additionally truncated at the first step with none."""
    by_step: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    chain_turns: Dict[str, set] = {}
    for record in graded_records:
        by_step.setdefault(_skey(record), []).append(record)
        chain_turns.setdefault(str(record.get("chain_id", "") or ""), set()).add(int(record.get("turn_index", 0) or 0))

    selected: List[Dict[str, Any]] = []
    counts = {"steps": 0, "accepted": 0, "contextual_accepted": 0, "fallback_used": 0, "no_passing_variant": 0}
    for sk, recs in by_step.items():
        counts["steps"] += 1
        passing = [r for r in recs if bool(r["validation"].get("variant_accepted"))]
        if passing:
            if config.variant_select == "first":
                chosen = min(passing, key=lambda r: int(r.get("variant_index", 0) or 0))
            else:
                chosen = max(passing, key=lambda r: (r["validation"].get("overall_score") or 0,
                                                     -int(r.get("variant_index", 0) or 0)))
            if int(chosen.get("variant_index", 0) or 0) != 0:
                counts["fallback_used"] += 1
        else:
            counts["no_passing_variant"] += 1
            chosen = max(recs, key=lambda r: (r["validation"].get("overall_score") or -1))
        accepted = bool(passing)
        # contextual coherence reuses the prefix computed in phase 1 (step-level)
        within_prefix = bool(dict(chosen.get("validation", {})).get("chain", {}).get("within_coherent_prefix", False))
        contextual = accepted and within_prefix
        counts["accepted"] += int(accepted)
        counts["contextual_accepted"] += int(contextual)

        record = dict(chosen)
        grade = dict(record.get("validation", {}))
        record["validation"] = {
            **grade,
            "phase": "selected",
            "accepted": accepted,                    # <- relevance_pool reads this
            "selected_variant_index": int(chosen.get("variant_index", 0) or 0),
            "used_variant_fallback": bool(passing) and int(chosen.get("variant_index", 0) or 0) != 0,
            # backwards-compatible aliases
            "history_unaware": {"passed": accepted, "reasons": [] if accepted else ["no_passing_variant"]},
            "history_aware": {"passed": contextual, "reasons": [] if contextual else ["history_truncated"]},
        }
        selected.append(record)

    report = {"phase": "selected", "config": _cfg(config), "counts": counts, "chains": len(chain_turns)}
    return selected, report


def _cfg(config: GateConfig) -> Dict[str, Any]:
    return {
        "accept_threshold": config.accept_threshold,
        "chain_aggregate": config.chain_aggregate,
        "contextual_policy": config.contextual_policy,
        "variant_select": config.variant_select,
        "instruction_field": config.instruction_field,
    }
