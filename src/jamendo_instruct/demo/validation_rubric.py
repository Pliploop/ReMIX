"""Shared validation rubric for human and LLM judges.

This module is the single source of truth for the questions raters answer, the
answer scales, and the helpers that assemble the per-item evidence and select
which items get judged. Both the Streamlit human-validation app
(:mod:`jamendo_instruct.demo.human_validation_app`) and the offline LLM judge
(``scripts/llm_validation_judge.py``) import from here so that the questions and
inputs stay identical across the two. It contains no Streamlit dependency and is
safe to import from a plain CLI.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from jamendo_instruct.demo.chains_demo import (
    ArtifactPaths,
    ChainView,
    DemoDataset,
    StepView,
    _format_tags,
    _instruction_text,
    _iter_jsonl,
)


LIKERT_OPTIONS: Sequence[tuple[str, int]] = (
    ("Strongly disagree", 1),
    ("Disagree", 2),
    ("Mixed", 3),
    ("Agree", 4),
    ("Strongly agree", 5),
)

# Some questions ask "how much", not "do you agree". They use their own scale but
# still map onto the same 1-5 scoring so the admin aggregates stay comparable.
DEGREE_OPTIONS: Sequence[tuple[str, int]] = (
    ("None of it", 1),
    ("A little", 2),
    ("Some", 3),
    ("Most", 4),
    ("All of it", 5),
)

RATING_QUESTIONS: Sequence[Dict[str, Any]] = (
    {
        "id": "meaningful_change",
        "statement": "The instruction asks for a real musical change from the source track to the target track.",
        "polarity": "positive",
        "help": "It should ask to change something real about the music. It does not need to mention every difference between the two tracks.",
    },
    {
        "id": "target_follows",
        "statement": "How much of what the instruction asks for can you actually hear in the target track?",
        "polarity": "positive",
        "scale": DEGREE_OPTIONS,
        "help": "Only judge the change the instruction asks for. The target is allowed to change in other ways too; that does not lower the score.",
    },
    {
        "id": "source_support",
        "statement": "Where the instruction describes the source track, that description is correct.",
        "polarity": "positive",
        "help": "If the instruction says something about the source (its style, mood, instruments...), check that the source track really is that way.",
    },
    {
        "id": "source_compatible",
        "statement": "The instruction does not say anything about the source track that is clearly wrong.",
        "polarity": "positive",
        "help": "Disagree only when the instruction states something about the source that the audio, caption, tags, or metadata show is false. If you cannot tell, choose Cannot judge.",
    },
    {
        "id": "conservation_supported",
        "statement": "If the instruction says to keep or preserve something, the target actually keeps it.",
        "polarity": "positive",
        "allow_na": True,
        "help": "Only for keep/preserve/maintain clauses. If the instruction does not ask to keep anything, choose Not applicable.",
    },
    {
        "id": "edit_specificity",
        "statement": "The instruction is written as an edit or a keep/preserve clause, not just a description of the target.",
        "polarity": "positive",
        "help": "Good: 'make it more electronic' or 'keep the vocals'. Weaker target-only: 'an upbeat synth-pop track'.",
    },
    {
        "id": "clarity_actionability",
        "statement": "The instruction is clear and easy to understand.",
        "polarity": "positive",
        "help": "Disagree when the instruction is vague, confusing, or contradicts itself.",
    },
    {
        "id": "overall_validity",
        "statement": "Overall, this is a good instruction for turning the source track into the target track.",
        "polarity": "positive",
        "help": "Your overall keep-or-reject judgment after looking at everything.",
    },
)

ISSUE_TAGS: Sequence[str] = (
    "wrong direction",
    "source mischaracterized",
    "target mischaracterized",
    "too generic / target-only",
    "unclear instruction",
    "unsupported genre/style claim",
    "unsupported instrument claim",
    "unsupported vocal claim",
    "unsupported speed/tempo claim",
    "unsupported mood/energy claim",
    "bad conservation clause",
    "audio contradicts caption/metadata",
    "insufficient evidence",
)

PAIRWISE_OPTIONS = [
    "A is better",
    "B is better",
    "Tie: both valid",
    "Tie: both flawed",
    "Cannot judge",
]

INSTRUCTION_FIELD_LABELS = {
    "history_unaware_instruction": "Standalone",
    "history_aware_instruction": "Contextual",
}

CANNOT_JUDGE_LABEL = "Cannot judge"
NOT_APPLICABLE_LABEL = "Not applicable"


def _validation_output_dir(dataset: DemoDataset) -> Path:
    if dataset.paths.instructions_jsonl is not None:
        return dataset.paths.instructions_jsonl.parent / "validation"
    if dataset.paths.run_root is not None:
        return dataset.paths.run_root / "human_validation"
    return dataset.paths.chains_jsonl.parent / "human_validation"


def _record_variant_index(record: Dict[str, Any]) -> int:
    try:
        return int(record.get("variant_index", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _record_from_sidecar(record: Dict[str, Any]) -> Dict[str, Any]:
    return dict(record)


def _dataset_from_frozen_sidecar(path: Path) -> tuple[DemoDataset, List[Dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    path_data = dict(data.get("paths", {}) or {})
    paths = ArtifactPaths(
        run_root=Path(path_data["run_root"]) if path_data.get("run_root") else None,
        manifest_csv=Path(path_data["manifest_csv"]),
        chains_jsonl=Path(path_data["chains_jsonl"]),
        instructions_jsonl=Path(path_data["instructions_jsonl"]) if path_data.get("instructions_jsonl") else None,
    )
    manifest_by_clip = {
        str(clip_id): {str(key): str(value) for key, value in dict(row or {}).items()}
        for clip_id, row in dict(data.get("manifest_by_clip", {}) or {}).items()
    }
    chains: List[ChainView] = []
    for raw_chain in data.get("chains", []) or []:
        steps: List[StepView] = []
        for raw_step in raw_chain.get("steps", []) or []:
            records = [_record_from_sidecar(record) for record in raw_step.get("instruction_records", []) or []]
            steps.append(
                StepView(
                    turn_index=int(raw_step.get("turn_index", 0) or 0),
                    source_clip_id=str(raw_step.get("source_clip_id", "") or ""),
                    target_clip_id=str(raw_step.get("target_clip_id", "") or ""),
                    split=str(raw_step.get("split", "") or ""),
                    hardness=str(raw_step.get("hardness", "") or ""),
                    transition_score=float(raw_step.get("transition_score", 0.0) or 0.0),
                    structured_delta=dict(raw_step.get("structured_delta", {}) or {}),
                    accumulated_intent_state=dict(raw_step.get("accumulated_intent_state", {}) or {}),
                    instruction_record=records[0] if records else None,
                    instruction_records=records,
                )
            )
        chains.append(
            ChainView(
                chain_id=str(raw_chain.get("chain_id", "") or ""),
                chain_length=int(raw_chain.get("chain_length", len(steps)) or len(steps)),
                sampled_target_length=int(raw_chain.get("sampled_target_length", len(steps)) or len(steps)),
                split=str(raw_chain.get("split", "") or ""),
                seed_clip_id=str(raw_chain.get("seed_clip_id", "") or ""),
                seed_row=manifest_by_clip.get(str(raw_chain.get("seed_clip_id", "") or "")),
                steps=steps,
            )
        )
    dataset = DemoDataset(
        paths=paths,
        chains=chains,
        chain_ids=[chain.chain_id for chain in chains],
        manifest_by_clip=manifest_by_clip,
        summary=dict(data.get("summary", {}) or {}),
    )
    return dataset, [dict(row) for row in data.get("assignments", []) or []]


def _available_samples(dataset: DemoDataset, instruction_field: str) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for chain in dataset.chains:
        for step in chain.steps:
            for record in step.instruction_records:
                text = _instruction_text(record, instruction_field).strip()
                if not text:
                    continue
                samples.append({"chain": chain, "step": step, "record": record, "instruction": text})
    return samples


def _available_pairs(dataset: DemoDataset, instruction_field: str) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for chain in dataset.chains:
        for step in chain.steps:
            variants = [
                record
                for record in step.instruction_records
                if _instruction_text(record, instruction_field).strip()
            ]
            if len(variants) < 2:
                continue
            variants = sorted(variants, key=_record_variant_index)
            for left_pos, left in enumerate(variants):
                for right in variants[left_pos + 1 :]:
                    a, b = left, right
                    digest = hashlib.sha1(
                        f"{chain.chain_id}:{step.turn_index}:{_record_variant_index(left)}:{_record_variant_index(right)}".encode(
                            "utf-8"
                        )
                    ).hexdigest()
                    if int(digest[:2], 16) % 2:
                        a, b = b, a
                    pairs.append({"chain": chain, "step": step, "a": a, "b": b})
    return pairs


def _assignment_step_key(record: Dict[str, Any]) -> tuple[str, int]:
    return str(record.get("chain_id", "") or ""), int(record.get("turn_index", 0) or 0)


def _read_assignments(path: Path | None) -> List[Dict[str, Any]]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"Assignment manifest not found: {path}")
    return list(_iter_jsonl(path))


def _assignment_index(assignments: Sequence[Dict[str, Any]]) -> Dict[tuple[str, int], Dict[str, Any]]:
    out: Dict[tuple[str, int], Dict[str, Any]] = {}
    for assignment in assignments:
        key = _assignment_step_key(assignment)
        if not key[0]:
            continue
        out[key] = dict(assignment)
    return out


def _filter_by_assignments(
    samples: Sequence[Dict[str, Any]],
    pairs: Sequence[Dict[str, Any]],
    assignments: Sequence[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not assignments:
        return list(samples), list(pairs)
    by_step = _assignment_index(assignments)
    filtered_samples: List[Dict[str, Any]] = []
    for sample in samples:
        chain: ChainView = sample["chain"]
        step: StepView = sample["step"]
        assignment = by_step.get((chain.chain_id, step.turn_index))
        if assignment is None:
            continue
        assigned_variants = {int(value) for value in assignment.get("variant_indices", []) or []}
        if assigned_variants and _record_variant_index(sample["record"]) not in assigned_variants:
            continue
        filtered_samples.append({**sample, "assignment": assignment})

    filtered_pairs: List[Dict[str, Any]] = []
    for pair in pairs:
        chain = pair["chain"]
        step = pair["step"]
        assignment = by_step.get((chain.chain_id, step.turn_index))
        if assignment is None:
            continue
        assigned_variants = {int(value) for value in assignment.get("variant_indices", []) or []}
        if assigned_variants and (
            _record_variant_index(pair["a"]) not in assigned_variants
            or _record_variant_index(pair["b"]) not in assigned_variants
        ):
            continue
        filtered_pairs.append({**pair, "assignment": assignment})
    return filtered_samples, filtered_pairs


def _sample_identity(chain: ChainView, step: StepView, record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "chain_id": chain.chain_id,
        "turn_index": step.turn_index,
        "variant_index": _record_variant_index(record),
        "split": step.split,
        "source_clip_id": step.source_clip_id,
        "target_clip_id": step.target_clip_id,
    }


def _question_scale(question: Dict[str, Any]) -> Sequence[tuple[str, int]]:
    return question.get("scale") or LIKERT_OPTIONS


def _rating_value(label: str | None, scale: Sequence[tuple[str, int]] = LIKERT_OPTIONS) -> tuple[int | None, bool, bool]:
    if label == CANNOT_JUDGE_LABEL:
        return None, True, False
    if label == NOT_APPLICABLE_LABEL:
        return None, False, True
    for option, score in scale:
        if option == label:
            return score, False, False
    return None, False, False


def _rating_help(question_id: str) -> str:
    for question in RATING_QUESTIONS:
        if question["id"] == question_id:
            return str(question.get("help", "") or "")
    return ""


def _metadata_view(row: Dict[str, str] | None) -> Dict[str, Any]:
    row = dict(row or {})
    fields = [
        "clip_id",
        "track_id",
        "artist_name",
        "title",
        "start_time",
        "end_time",
        "vocals",
        "speed",
        "license_ccurl",
    ]
    out = {field: row.get(field, "") for field in fields if str(row.get(field, "") or "").strip()}
    tags = _format_tags(row)
    if tags:
        out["tags"] = tags
    return out


def _clip_label(row: Dict[str, str] | None, fallback: str) -> str:
    row = dict(row or {})
    title = str(row.get("title", "") or "").strip()
    artist = str(row.get("artist_name", "") or "").strip()
    clip_id = str(row.get("clip_id", "") or fallback or "").strip()
    if title and artist:
        return f"{artist} - {title}"
    return title or artist or clip_id or fallback


def _read_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return list(_iter_jsonl(path))


def _rating_item_key(record: Dict[str, Any]) -> tuple[str, int, int, str]:
    return (
        str(record.get("chain_id", "") or ""),
        int(record.get("turn_index", 0) or 0),
        int(record.get("variant_index", 0) or 0),
        str(record.get("instruction_field", "") or ""),
    )


def _answered_rating_keys(
    records: Sequence[Dict[str, Any]], annotator_id: str | None = None
) -> set[tuple[str, int, int, str]]:
    keys: set[tuple[str, int, int, str]] = set()
    for record in records:
        if annotator_id is not None and str(record.get("annotator_id", "") or "") != annotator_id:
            continue
        keys.add(_rating_item_key(record))
    return keys
