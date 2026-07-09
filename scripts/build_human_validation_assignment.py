#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

from jamendo_instruct.demo.chains_demo import (
    DemoDataset,
    _caption_only_change,
    _instruction_axes,
    _instruction_text,
    _load_dataset_for_streamlit,
)


SPLIT_ALIASES = {
    "validation": {"validation", "val", "valid", "dev"},
    "val": {"validation", "val", "valid", "dev"},
    "valid": {"validation", "val", "valid", "dev"},
    "dev": {"validation", "val", "valid", "dev"},
    "test": {"test"},
    "train": {"train"},
}


def _stable_digest(*parts: object) -> str:
    return hashlib.sha1(":".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def _score_band(score: float) -> str:
    if score < 0.75:
        return "low"
    if score < 0.85:
        return "mid"
    return "high"


def _turn_band(turn_index: int) -> str:
    if turn_index <= 1:
        return "early"
    if turn_index <= 3:
        return "middle"
    return "late"


def _variant_band(count: int) -> str:
    if count <= 1:
        return "single"
    if count < 5:
        return "partial"
    return "full5"


def _axis_count_band(count: int) -> str:
    if count <= 0:
        return "none"
    if count == 1:
        return "single_axis"
    return "multi_axis"


def _variant_indices(records: Sequence[Dict[str, Any]], instruction_field: str) -> List[int]:
    indices: List[int] = []
    for record in records:
        if not _instruction_text(record, instruction_field).strip():
            continue
        try:
            indices.append(int(record.get("variant_index", 0) or 0))
        except (TypeError, ValueError):
            indices.append(0)
    return sorted(set(indices))


def _candidate_steps(
    dataset: DemoDataset,
    *,
    split: str,
    instruction_field: str,
    min_variants: int,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    accepted_splits = SPLIT_ALIASES.get(split, {split}) if split else set()
    for chain in dataset.chains:
        for step in chain.steps:
            step_split = step.split or chain.split or ""
            if split and step_split not in accepted_splits:
                continue
            variant_indices = _variant_indices(step.instruction_records, instruction_field)
            if len(variant_indices) < min_variants:
                continue
            representative_record = next(
                (record for record in step.instruction_records if _instruction_text(record, instruction_field).strip()),
                step.instruction_record,
            )
            axes = _instruction_axes(representative_record)
            primary_axis = axes[0] if axes else "none"
            axis_count = len(axes)
            caption_only = _caption_only_change(representative_record)
            transition_score = float(step.transition_score or 0.0)
            stratum = {
                "primary_axis": primary_axis,
                "axis_count_band": _axis_count_band(axis_count),
                "caption_scope": "caption_only" if caption_only else "metadata_or_mixed",
                "hardness": step.hardness or "unknown",
                "score_band": _score_band(transition_score),
                "variant_band": _variant_band(len(variant_indices)),
            }
            candidates.append(
                {
                    "chain_id": chain.chain_id,
                    "turn_index": step.turn_index,
                    "split": step_split,
                    "source_clip_id": step.source_clip_id,
                    "target_clip_id": step.target_clip_id,
                    "variant_indices": variant_indices,
                    "hardness": step.hardness,
                    "transition_score": transition_score,
                    "primary_axis": primary_axis,
                    "change_axes": axes,
                    "change_axis_count": axis_count,
                    "caption_only_change": caption_only,
                    "turn_band": _turn_band(step.turn_index),
                    "stratum": stratum,
                    "stratum_key": "|".join(stratum.values()),
                    "stable_hash": _stable_digest(chain.chain_id, step.turn_index, ",".join(map(str, variant_indices))),
                }
            )
    return candidates


def _available_split_counts(dataset: DemoDataset, instruction_field: str, min_variants: int) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for chain in dataset.chains:
        for step in chain.steps:
            variant_indices = _variant_indices(step.instruction_records, instruction_field)
            if len(variant_indices) < min_variants:
                continue
            split = step.split or chain.split or "<missing>"
            counts[split] = counts.get(split, 0) + 1
    return dict(sorted(counts.items()))


def _one_step_per_chain(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_chain: Dict[str, Dict[str, Any]] = {}
    for candidate in sorted(candidates, key=lambda row: (row["chain_id"], row["stable_hash"])):
        by_chain.setdefault(str(candidate["chain_id"]), candidate)
    return list(by_chain.values())


def _select_assignments(
    candidates: Sequence[Dict[str, Any]],
    *,
    total_steps: int,
    sentinel_steps: int,
    core_overlap_steps: int,
    seed: int,
    stratify: bool,
) -> List[Dict[str, Any]]:
    if total_steps <= 0:
        raise ValueError("--total-steps must be positive")
    if sentinel_steps + core_overlap_steps > total_steps:
        raise ValueError("--sentinel-steps + --core-overlap-steps must be <= --total-steps")
    if len(candidates) < total_steps:
        raise ValueError(f"Need {total_steps} candidate steps but only found {len(candidates)}")

    sentinel_pool = _balanced_order(candidates, "stable_hash") if stratify else sorted(candidates, key=lambda row: row["stable_hash"])
    selected: List[Dict[str, Any]] = []
    used = set()

    for candidate in sentinel_pool[:sentinel_steps]:
        used.add((candidate["chain_id"], candidate["turn_index"]))
        selected.append({**candidate, "bucket": "sentinel", "is_sentinel": True})

    remaining = [
        candidate
        for candidate in candidates
        if (candidate["chain_id"], candidate["turn_index"]) not in used
    ]
    rng = random.Random(seed)
    if stratify:
        remaining = _balanced_order(remaining, "random_key", rng=rng)
    else:
        rng.shuffle(remaining)

    for candidate in remaining[:core_overlap_steps]:
        used.add((candidate["chain_id"], candidate["turn_index"]))
        selected.append({**candidate, "bucket": "core_overlap", "is_sentinel": False})

    remaining = [
        candidate
        for candidate in remaining[core_overlap_steps:]
        if (candidate["chain_id"], candidate["turn_index"]) not in used
    ]
    extension_steps = total_steps - len(selected)
    for candidate in remaining[:extension_steps]:
        selected.append({**candidate, "bucket": "extension", "is_sentinel": False})

    selected.sort(key=lambda row: ({"sentinel": 0, "core_overlap": 1, "extension": 2}[row["bucket"]], row["stable_hash"]))
    for index, row in enumerate(selected):
        row["assignment_id"] = f"hv_{index:05d}"
    return selected


def _balanced_order(candidates: Sequence[Dict[str, Any]], key: str, *, rng: random.Random | None = None) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for candidate in candidates:
        groups.setdefault(str(candidate.get("stratum_key", "unknown")), []).append(candidate)
    for rows in groups.values():
        if key == "random_key":
            assert rng is not None
            rng.shuffle(rows)
        else:
            rows.sort(key=lambda row: row[key])
    ordered: List[Dict[str, Any]] = []
    group_keys = sorted(groups)
    while group_keys:
        next_keys: List[str] = []
        for group_key in group_keys:
            rows = groups[group_key]
            if rows:
                ordered.append(rows.pop(0))
            if rows:
                next_keys.append(group_key)
        group_keys = next_keys
    return ordered


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            f.write("\n")


def _assignment_key(row: Dict[str, Any]) -> tuple[str, int]:
    return str(row.get("chain_id", "") or ""), int(row.get("turn_index", 0) or 0)


def _load_manifest_subset(manifest_csv: Path, clip_ids: set[str]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    if not clip_ids:
        return out
    with manifest_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip_id = str(row.get("clip_id", "") or "").strip()
            if clip_id in clip_ids:
                out[clip_id] = {str(key): str(value) for key, value in row.items()}
                if len(out) >= len(clip_ids):
                    break
    return out


def _write_sidecar(path: Path, dataset: DemoDataset, assignments: Sequence[Dict[str, Any]]) -> None:
    assignment_by_step = {_assignment_key(row): row for row in assignments}
    assigned_clip_ids: set[str] = set()
    chains: List[Dict[str, Any]] = []
    for chain in dataset.chains:
        raw_steps: List[Dict[str, Any]] = []
        for step in chain.steps:
            assignment = assignment_by_step.get((chain.chain_id, step.turn_index))
            if assignment is None:
                continue
            allowed_variants = {int(value) for value in assignment.get("variant_indices", []) or []}
            records = [
                dict(record)
                for record in step.instruction_records
                if not allowed_variants or int(record.get("variant_index", 0) or 0) in allowed_variants
            ]
            if not records:
                continue
            assigned_clip_ids.update([step.source_clip_id, step.target_clip_id])
            if chain.seed_clip_id:
                assigned_clip_ids.add(chain.seed_clip_id)
            raw_steps.append(
                {
                    "turn_index": step.turn_index,
                    "source_clip_id": step.source_clip_id,
                    "target_clip_id": step.target_clip_id,
                    "split": step.split,
                    "hardness": step.hardness,
                    "transition_score": step.transition_score,
                    "structured_delta": step.structured_delta,
                    "accumulated_intent_state": step.accumulated_intent_state,
                    "instruction_records": records,
                }
            )
        if raw_steps:
            chains.append(
                {
                    "chain_id": chain.chain_id,
                    "chain_length": chain.chain_length,
                    "sampled_target_length": chain.sampled_target_length,
                    "split": chain.split,
                    "seed_clip_id": chain.seed_clip_id,
                    "steps": raw_steps,
                }
            )
    manifest_by_clip = _load_manifest_subset(dataset.paths.manifest_csv, assigned_clip_ids)
    payload = {
        "version": 1,
        "paths": {
            "run_root": str(dataset.paths.run_root) if dataset.paths.run_root else None,
            "manifest_csv": str(dataset.paths.manifest_csv),
            "chains_jsonl": str(dataset.paths.chains_jsonl),
            "instructions_jsonl": str(dataset.paths.instructions_jsonl) if dataset.paths.instructions_jsonl else None,
        },
        "summary": {
            **dict(dataset.summary or {}),
            "frozen_assignment_steps": len(assignments),
            "frozen_chains": len(chains),
            "frozen_manifest_rows": len(manifest_by_clip),
        },
        "assignments": list(assignments),
        "manifest_by_clip": manifest_by_clip,
        "chains": chains,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True), encoding="utf-8")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a frozen human-validation assignment manifest.")
    parser.add_argument("--run-root", required=True, help="Run artifact root.")
    parser.add_argument("--instructions-jsonl", help="Optional explicit chain_step_instructions.jsonl.")
    parser.add_argument("--output-jsonl", help="Output assignment JSONL path.")
    parser.add_argument("--sidecar-json", help="Output compact frozen sidecar JSON path.")
    parser.add_argument("--split", default="validation", help="Dataset split to sample from.")
    parser.add_argument("--instruction-field", default="history_unaware_instruction")
    parser.add_argument("--total-steps", type=int, default=60)
    parser.add_argument("--sentinel-steps", type=int, default=30)
    parser.add_argument("--core-overlap-steps", type=int, default=30)
    parser.add_argument("--min-variants", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--allow-multiple-steps-per-chain", action="store_true")
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable balanced sampling across primary-axis, axis-count, caption-scope, hardness, score, and variant-count strata.",
    )
    parser.add_argument("--max-chains", type=int, default=0, help="Maximum chains to load while building; 0 loads all.")
    parser.add_argument(
        "--full-coverage",
        action="store_true",
        help="Ignore sampling/stratification and emit one assignment per step covering ALL variants. "
        "Use with --split all to include every split. Produces a full sidecar for at-scale LLM validation.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    output_jsonl = (
        Path(args.output_jsonl).expanduser().resolve()
        if args.output_jsonl
        else run_root / "validation" / "assignment_v1.jsonl"
    )
    sidecar_json = (
        Path(args.sidecar_json).expanduser().resolve()
        if args.sidecar_json
        else output_jsonl.with_suffix(".sidecar.json")
    )
    max_chains = None if int(args.max_chains) <= 0 else int(args.max_chains)
    dataset = _load_dataset_for_streamlit(
        str(run_root),
        None,
        None,
        args.instructions_jsonl,
        0,
        max_chains,
    )
    split_arg = "" if str(args.split).strip().lower() in {"all", "any", ""} else str(args.split)
    candidates = _candidate_steps(
        dataset,
        split=split_arg,
        instruction_field=str(args.instruction_field),
        min_variants=max(1, int(args.min_variants)),
    )
    if not args.full_coverage and not args.allow_multiple_steps_per_chain:
        candidates = _one_step_per_chain(candidates)
    if not candidates:
        split_counts = _available_split_counts(
            dataset,
            str(args.instruction_field),
            max(1, int(args.min_variants)),
        )
        raise ValueError(
            "No candidate steps matched the requested split/min-variant filter. "
            f"requested_split={args.split!r}; accepted_splits={sorted(SPLIT_ALIASES.get(str(args.split), {str(args.split)}))}; "
            f"available_split_counts={split_counts}"
        )
    if args.full_coverage:
        # Every step, every variant; no sampling. bucket="full" so the loaders and
        # sidecar include the whole set.
        assignments = [
            {**candidate, "bucket": "full", "is_sentinel": False}
            for candidate in candidates
        ]
    else:
        assignments = _select_assignments(
            candidates,
            total_steps=int(args.total_steps),
            sentinel_steps=int(args.sentinel_steps),
            core_overlap_steps=int(args.core_overlap_steps),
            seed=int(args.seed),
            stratify=not bool(args.no_stratify),
        )
    for row in assignments:
        row["instruction_field"] = str(args.instruction_field)
        row["selection_seed"] = int(args.seed)
    _write_jsonl(output_jsonl, assignments)
    _write_sidecar(sidecar_json, dataset, assignments)
    bucket_counts: Dict[str, int] = {}
    for row in assignments:
        bucket_counts[row["bucket"]] = bucket_counts.get(row["bucket"], 0) + 1
    print(
        json.dumps(
            {
                "status": "ok",
                "output_jsonl": str(output_jsonl),
                "sidecar_json": str(sidecar_json),
                "assignments": len(assignments),
                "candidate_steps": len(candidates),
                "bucket_counts": bucket_counts,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
