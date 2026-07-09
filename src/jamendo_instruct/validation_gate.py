"""Graded validation gate: turn per-instruction rubric scores into an accept
decision plus a chain-coherence judgement, and emit ``validated_instructions``
in the schema the ``relevance_pool`` stage already consumes.

Two axes (see ``docs/validation_gate.md``):

* **instruction validity** — the graded rubric ``overall_validity`` (1-5) per
  step/variant, thresholded to a binary accept. This is the human-calibrated
  currency: the same 1-5 scale humans use, so the threshold can be tuned to
  match the human accept rate.
* **chain coherence** — for the *contextual* (history-aware) track a step is
  only coherent if every earlier step in its chain is instruction-valid;
  otherwise the accumulated history is broken. Enforced structurally by
  *truncating* the chain at the first invalid step (configurable).

The output record is ``dict(instruction_record)`` + a ``"validation"`` block
whose ``accepted`` field is what ``relevance_pool`` reads, plus graded fields
and backwards-compatible ``history_unaware`` / ``history_aware`` aliases so the
old binary consumers keep working.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple


PrimaryTrack = str  # "standalone" | "contextual"
ContextualPolicy = str  # "truncate" | "drop" | "per_step"
ChainAggregate = str  # "min" | "mean"
UnratedPolicy = str  # "reject" | "pass"

_OVERALL_QUESTION = "overall_validity"
_CHAIN_QUESTION = "chain_coherence"  # optional future rubric dimension


@dataclass(frozen=True)
class GateConfig:
    accept_threshold: float = 4.0
    chain_aggregate: ChainAggregate = "min"
    contextual_policy: ContextualPolicy = "truncate"
    unrated_policy: UnratedPolicy = "reject"
    primary_track: PrimaryTrack = "standalone"
    instruction_field: str = "history_unaware_instruction"
    gate_name: str = "graded_v1"


def _key(record: Dict[str, Any]) -> Tuple[str, int, int]:
    return (
        str(record.get("chain_id", "") or ""),
        int(record.get("turn_index", 0) or 0),
        int(record.get("variant_index", 0) or 0),
    )


def _question_scores(rating: Dict[str, Any]) -> Dict[str, Any]:
    answers = dict(rating.get("answers", {}) or {})
    out: Dict[str, Any] = {}
    for qid, ans in answers.items():
        ans = dict(ans or {})
        out[str(qid)] = ans.get("score")
    return out


def _overall_score(rating: Dict[str, Any] | None) -> float | None:
    if not rating:
        return None
    ans = dict((rating.get("answers", {}) or {}).get(_OVERALL_QUESTION, {}) or {})
    score = ans.get("score")
    return float(score) if isinstance(score, (int, float)) else None


def _chain_coherence_score(rating: Dict[str, Any] | None) -> float | None:
    """Optional LLM-graded chain coherence, if a future rubric emits it."""
    if not rating:
        return None
    ans = dict((rating.get("answers", {}) or {}).get(_CHAIN_QUESTION, {}) or {})
    score = ans.get("score")
    return float(score) if isinstance(score, (int, float)) else None


def _dedupe_ratings(ratings: Iterable[Dict[str, Any]], instruction_field: str) -> Dict[Tuple[str, int, int], Dict[str, Any]]:
    """Last rating wins per (chain, turn, variant) for the target instruction field."""
    index: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    for rating in ratings:
        field = str(rating.get("instruction_field", "") or "")
        if field and instruction_field and field != instruction_field:
            continue
        index[_key(rating)] = dict(rating)
    return index


def build_validated_records(
    instruction_records: Sequence[Dict[str, Any]],
    ratings: Iterable[Dict[str, Any]],
    config: GateConfig = GateConfig(),
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return (validated_records, report). One output record per input record."""
    rate_index = _dedupe_ratings(ratings, config.instruction_field)

    # Per-step instruction validity: a step is valid if its best-rated variant
    # clears the threshold. Used for chain-coherence prefixes.
    step_best: Dict[Tuple[str, int], float] = {}
    step_rated: Dict[Tuple[str, int], bool] = {}
    for record in instruction_records:
        ck = (str(record.get("chain_id", "") or ""), int(record.get("turn_index", 0) or 0))
        score = _overall_score(rate_index.get(_key(record)))
        if score is not None:
            step_best[ck] = max(step_best.get(ck, float("-inf")), score)
            step_rated[ck] = True
        else:
            step_rated.setdefault(ck, False)

    def step_valid(ck: Tuple[str, int]) -> bool:
        return ck in step_best and step_best[ck] >= config.accept_threshold

    # Per-chain: ordered turns, coherent prefix end, chain score.
    chain_turns: Dict[str, List[int]] = {}
    for record in instruction_records:
        cid = str(record.get("chain_id", "") or "")
        turn = int(record.get("turn_index", 0) or 0)
        chain_turns.setdefault(cid, [])
        if turn not in chain_turns[cid]:
            chain_turns[cid].append(turn)

    chain_prefix_end: Dict[str, int] = {}   # last turn with an all-valid prefix
    chain_truncate_at: Dict[str, int | None] = {}
    chain_all_valid: Dict[str, bool] = {}
    chain_score: Dict[str, float | None] = {}
    for cid, turns in chain_turns.items():
        turns_sorted = sorted(turns)
        prefix_end = -1
        truncate_at: int | None = None
        for t in turns_sorted:
            if step_valid((cid, t)):
                if truncate_at is None:
                    prefix_end = t
            else:
                if truncate_at is None:
                    truncate_at = t
        chain_prefix_end[cid] = prefix_end
        chain_truncate_at[cid] = truncate_at
        chain_all_valid[cid] = truncate_at is None and all(step_valid((cid, t)) for t in turns_sorted)
        scored = [step_best[(cid, t)] for t in turns_sorted if (cid, t) in step_best]
        if scored:
            chain_score[cid] = min(scored) if config.chain_aggregate == "min" else sum(scored) / len(scored)
        else:
            chain_score[cid] = None

    out: List[Dict[str, Any]] = []
    counts = {
        "total": 0, "rated": 0, "unrated": 0,
        "standalone_accepted": 0, "contextual_accepted": 0, "accepted": 0,
    }
    for record in instruction_records:
        counts["total"] += 1
        cid = str(record.get("chain_id", "") or "")
        turn = int(record.get("turn_index", 0) or 0)
        rating = rate_index.get(_key(record))
        score = _overall_score(rating)
        rated = score is not None
        counts["rated" if rated else "unrated"] += 1

        if rated:
            standalone_accepted = score >= config.accept_threshold
        else:
            standalone_accepted = config.unrated_policy == "pass"

        # contextual coherence
        if config.contextual_policy == "per_step":
            coherent = standalone_accepted
        elif config.contextual_policy == "drop":
            coherent = standalone_accepted and chain_all_valid.get(cid, False)
        else:  # truncate
            coherent = standalone_accepted and turn <= chain_prefix_end.get(cid, -1)
        # optional LLM-graded chain coherence overrides the structural rule when present
        cc_score = _chain_coherence_score(rating)
        if cc_score is not None:
            coherent = coherent and cc_score >= config.accept_threshold

        accepted = standalone_accepted if config.primary_track == "standalone" else coherent
        counts["standalone_accepted"] += int(standalone_accepted)
        counts["contextual_accepted"] += int(coherent)
        counts["accepted"] += int(accepted)

        reasons_std = [] if standalone_accepted else (["not_validated"] if not rated else ["below_threshold"])
        reasons_ctx = [] if coherent else (["history_truncated"] if standalone_accepted else reasons_std)

        validation = {
            "accepted": bool(accepted),                # <- relevance_pool reads this
            "gate": config.gate_name,
            "rated": rated,
            "threshold": config.accept_threshold,
            "primary_track": config.primary_track,
            "overall_score": score,
            "question_scores": _question_scores(rating) if rating else {},
            "standalone": {"passed": bool(standalone_accepted), "score": score},
            "contextual": {
                "passed": bool(coherent),
                "score": score,
                "chain_coherence_score": cc_score,
                "policy": config.contextual_policy,
                "chain_truncate_at": chain_truncate_at.get(cid),
            },
            "chain": {
                "score": chain_score.get(cid),
                "aggregate": config.chain_aggregate,
                "all_steps_valid": chain_all_valid.get(cid, False),
            },
            # Backwards-compatible aliases for the old binary validation schema.
            "history_unaware": {"passed": bool(standalone_accepted), "reasons": reasons_std, "checks": {}},
            "history_aware": {"passed": bool(coherent), "reasons": reasons_ctx, "checks": {}},
        }
        merged = dict(record)
        merged["validation"] = validation
        out.append(merged)

    report = {
        "gate": config.gate_name,
        "config": {
            "accept_threshold": config.accept_threshold,
            "chain_aggregate": config.chain_aggregate,
            "contextual_policy": config.contextual_policy,
            "unrated_policy": config.unrated_policy,
            "primary_track": config.primary_track,
            "instruction_field": config.instruction_field,
        },
        "counts": counts,
        "chains": len(chain_turns),
    }
    return out, report
