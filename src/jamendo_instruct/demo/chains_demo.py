from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class ArtifactPaths:
    run_root: Path | None
    manifest_csv: Path
    chains_jsonl: Path
    instructions_jsonl: Path | None


@dataclass(frozen=True)
class StepView:
    turn_index: int
    source_clip_id: str
    target_clip_id: str
    split: str
    hardness: str
    transition_score: float
    structured_delta: Dict[str, Any]
    accumulated_intent_state: Dict[str, Any]
    instruction_record: Dict[str, Any] | None


@dataclass(frozen=True)
class ChainView:
    chain_id: str
    chain_length: int
    sampled_target_length: int
    split: str
    seed_clip_id: str
    seed_row: Dict[str, str] | None
    steps: Sequence[StepView]


@dataclass(frozen=True)
class DemoDataset:
    paths: ArtifactPaths
    chains: Sequence[ChainView]
    chain_ids: Sequence[str]
    manifest_by_clip: Dict[str, Dict[str, str]]
    summary: Dict[str, Any]


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_no}") from exc


def _instruction_record_key(record: Dict[str, Any]) -> tuple[str, int] | None:
    chain_id = str(record.get("chain_id", "") or "").strip()
    if not chain_id:
        return None
    try:
        turn_index = int(record.get("turn_index", 0) or 0)
    except (TypeError, ValueError):
        return None
    return chain_id, turn_index


def _instruction_path_key(path: Path) -> tuple[str, int] | None:
    stem = path.stem
    if "__turn_" not in stem:
        return None
    chain_id, tail = stem.split("__turn_", 1)
    turn_text = tail.split("__", 1)[0]
    try:
        return chain_id, int(turn_text)
    except ValueError:
        return None


def _iter_instruction_records(path: Path | None) -> Iterable[Dict[str, Any]]:
    if path is None:
        return
    seen: set[tuple[str, int]] = set()
    if path.exists():
        for record in _iter_jsonl(path):
            key = _instruction_record_key(record)
            if key is None or key in seen:
                continue
            seen.add(key)
            yield record

    sidecar_dir = path.parent / "step_json"
    if not sidecar_dir.exists():
        return
    for json_path in sorted(sidecar_dir.glob("*.json")):
        try:
            with json_path.open("r", encoding="utf-8") as f:
                record = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed instruction JSON at {json_path}") from exc
        if not isinstance(record, dict):
            continue
        key = _instruction_record_key(record)
        if key is None or key in seen:
            continue
        seen.add(key)
        yield record


def _parse_json_list(raw: str) -> List[str]:
    value = str(raw or "").strip()
    if not value:
        return []
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [str(item).strip() for item in data if str(item).strip()]


def _resolve_paths(args: argparse.Namespace) -> ArtifactPaths:
    run_root = Path(args.run_root).expanduser().resolve() if args.run_root else None
    manifest_csv = Path(args.manifest_csv).expanduser().resolve() if args.manifest_csv else None
    chains_jsonl = Path(args.chains_jsonl).expanduser().resolve() if args.chains_jsonl else None
    instructions_jsonl = Path(args.instructions_jsonl).expanduser().resolve() if args.instructions_jsonl else None

    if run_root is not None:
        manifest_csv = manifest_csv or (run_root / "structured_view" / "structured_clip_manifest.csv")
        chains_jsonl = chains_jsonl or (run_root / "chains" / "sampled_chains.jsonl")
        inferred_instructions = run_root / "instructions" / "chain_step_instructions.jsonl"
        inferred_step_json = inferred_instructions.parent / "step_json"
        instructions_jsonl = instructions_jsonl or (
            inferred_instructions if inferred_instructions.exists() or inferred_step_json.is_dir() else None
        )

    if manifest_csv is None or chains_jsonl is None:
        raise ValueError("Provide --run-root or both --manifest-csv and --chains-jsonl.")

    if not manifest_csv.exists():
        raise FileNotFoundError(f"Structured manifest not found: {manifest_csv}")
    if not chains_jsonl.exists():
        raise FileNotFoundError(f"Chains artifact not found: {chains_jsonl}")
    if (
        instructions_jsonl is not None
        and not instructions_jsonl.exists()
        and not (instructions_jsonl.parent / "step_json").is_dir()
    ):
        raise FileNotFoundError(f"Instructions artifact not found: {instructions_jsonl}")

    return ArtifactPaths(
        run_root=run_root,
        manifest_csv=manifest_csv,
        chains_jsonl=chains_jsonl,
        instructions_jsonl=instructions_jsonl,
    )


def _instruction_folder_options(run_root: str | None) -> List[Path]:
    if not run_root:
        return []
    root = Path(run_root).expanduser()
    if not root.exists():
        return []
    options: List[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        instructions_jsonl = child / "chain_step_instructions.jsonl"
        if instructions_jsonl.exists() or (child / "step_json").is_dir():
            options.append(child)
    return options


def _load_instruction_chain_ids(path: Path | None) -> tuple[List[str], int]:
    if path is None:
        return [], 0
    ordered: List[str] = []
    seen_chains: set[str] = set()
    seen_steps: set[tuple[str, int]] = set()
    count = 0
    if path.exists():
        for record in _iter_jsonl(path):
            key = _instruction_record_key(record)
            if key is None or key in seen_steps:
                continue
            seen_steps.add(key)
            count += 1
            chain_id = key[0]
            if chain_id not in seen_chains:
                ordered.append(chain_id)
                seen_chains.add(chain_id)

    sidecar_dir = path.parent / "step_json"
    if sidecar_dir.exists():
        for json_path in sorted(sidecar_dir.glob("*.json")):
            key = _instruction_path_key(json_path)
            if key is None or key in seen_steps:
                continue
            seen_steps.add(key)
            count += 1
            chain_id = key[0]
            if chain_id not in seen_chains:
                ordered.append(chain_id)
                seen_chains.add(chain_id)
    return ordered, count


def _load_chain_records(
    path: Path,
    *,
    chain_offset: int,
    max_chains: int | None,
    preferred_chain_ids: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    if preferred_chain_ids:
        selected_ids = list(preferred_chain_ids[chain_offset:])
        if max_chains is not None:
            selected_ids = selected_ids[:max_chains]
        selected = set(selected_ids)
        found: Dict[str, Dict[str, Any]] = {}
        for record in _iter_jsonl(path):
            chain_id = str(record.get("chain_id", "") or "").strip()
            if chain_id in selected:
                found[chain_id] = record
                if len(found) >= len(selected):
                    break
        records = [found[chain_id] for chain_id in selected_ids if chain_id in found]
        if not records:
            raise ValueError(
                f"No preferred instructed chains were loaded from {path}. "
                f"Check that instruction chain IDs are present in the chain artifact."
            )
        return records

    records: List[Dict[str, Any]] = []
    for idx, record in enumerate(_iter_jsonl(path)):
        if idx < chain_offset:
            continue
        records.append(record)
        if max_chains is not None and len(records) >= max_chains:
            break
    if not records:
        raise ValueError(
            f"No chains were loaded from {path}. "
            f"Check --chain-offset/--max-chains or confirm the artifact is populated."
        )
    return records


def _referenced_clip_ids(chains: Sequence[Dict[str, Any]]) -> set[str]:
    clip_ids: set[str] = set()
    for chain in chains:
        seed = dict(chain.get("seed", {}) or {})
        seed_clip_id = str(seed.get("clip_id", "") or "").strip()
        if seed_clip_id:
            clip_ids.add(seed_clip_id)
        for step in chain.get("steps", []) or []:
            for key in ("source_clip_id", "target_clip_id"):
                clip_id = str(step.get(key, "") or "").strip()
                if clip_id:
                    clip_ids.add(clip_id)
    return clip_ids


def _load_manifest_rows(path: Path, keep_clip_ids: set[str]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    if not keep_clip_ids:
        return out
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip_id = str(row.get("clip_id", "") or "").strip()
            if clip_id and clip_id in keep_clip_ids:
                out[clip_id] = row
                if len(out) >= len(keep_clip_ids):
                    break
    return out


def _load_instruction_index(path: Path | None, keep_chain_ids: set[str]) -> Dict[tuple[str, int], Dict[str, Any]]:
    if path is None:
        return {}
    out: Dict[tuple[str, int], Dict[str, Any]] = {}
    if path.exists():
        for record in _iter_jsonl(path):
            key = _instruction_record_key(record)
            if key is None or key[0] not in keep_chain_ids:
                continue
            out[key] = record

    sidecar_dir = path.parent / "step_json"
    if not sidecar_dir.exists():
        return out
    for json_path in sorted(sidecar_dir.glob("*.json")):
        key = _instruction_path_key(json_path)
        if key is None or key[0] not in keep_chain_ids or key in out:
            continue
        try:
            with json_path.open("r", encoding="utf-8") as f:
                record = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed instruction JSON at {json_path}") from exc
        if isinstance(record, dict):
            out[key] = record
    return out


def _structured_delta_from_rows(source_row: Dict[str, str], target_row: Dict[str, str]) -> Dict[str, Any]:
    source_tags = set(_format_tags(source_row))
    target_tags = set(_format_tags(target_row))
    return {
        "tags_added": sorted(target_tags - source_tags),
        "tags_removed": sorted(source_tags - target_tags),
        "tags_preserved": sorted(source_tags & target_tags),
        "source_vocals": str(source_row.get("vocals", "") or ""),
        "target_vocals": str(target_row.get("vocals", "") or ""),
        "source_speed": str(source_row.get("speed", "") or ""),
        "target_speed": str(target_row.get("speed", "") or ""),
        "source_caption": str(source_row.get("normalized_caption", "") or source_row.get("caption", "") or ""),
        "target_caption": str(target_row.get("normalized_caption", "") or target_row.get("caption", "") or ""),
        "source_lyrics": str(source_row.get("normalized_lyrics", "") or source_row.get("lyrics", "") or ""),
        "target_lyrics": str(target_row.get("normalized_lyrics", "") or target_row.get("lyrics", "") or ""),
    }


def _load_dataset_from_instruction_records(
    paths: ArtifactPaths,
    *,
    chain_offset: int,
    max_chains: int | None,
    instructed_chain_ids: Sequence[str],
    total_instruction_records: int,
) -> DemoDataset:
    selected_chain_ids = list(instructed_chain_ids[chain_offset:])
    if max_chains is not None:
        selected_chain_ids = selected_chain_ids[:max_chains]
    if not selected_chain_ids:
        raise ValueError("No instructed chains were selected. Check --chain-offset/--max-chains.")

    instruction_index = _load_instruction_index(paths.instructions_jsonl, set(selected_chain_ids))
    records_by_chain: Dict[str, List[Dict[str, Any]]] = {chain_id: [] for chain_id in selected_chain_ids}
    clip_ids: set[str] = set()
    for (chain_id, _turn_index), record in instruction_index.items():
        records_by_chain.setdefault(chain_id, []).append(record)
        for key in ("seed_clip_id", "source_clip_id", "target_clip_id"):
            clip_id = str(record.get(key, "") or "").strip()
            if clip_id:
                clip_ids.add(clip_id)

    manifest_by_clip = _load_manifest_rows(paths.manifest_csv, clip_ids)
    chains: List[ChainView] = []
    instructions_found = 0
    missing_manifest_rows = 0

    for chain_id in selected_chain_ids:
        records = sorted(records_by_chain.get(chain_id, []), key=lambda record: int(record.get("turn_index", 0) or 0))
        if not records:
            continue
        first = records[0]
        seed_clip_id = str(first.get("seed_clip_id", "") or "").strip()
        seed_row = manifest_by_clip.get(seed_clip_id)
        if seed_clip_id and seed_row is None:
            missing_manifest_rows += 1

        steps: List[StepView] = []
        for record in records:
            source_clip_id = str(record.get("source_clip_id", "") or "").strip()
            target_clip_id = str(record.get("target_clip_id", "") or "").strip()
            source_row = manifest_by_clip.get(source_clip_id, {})
            target_row = manifest_by_clip.get(target_clip_id, {})
            if source_clip_id and source_clip_id not in manifest_by_clip:
                missing_manifest_rows += 1
            if target_clip_id and target_clip_id not in manifest_by_clip:
                missing_manifest_rows += 1
            instructions_found += 1
            steps.append(
                StepView(
                    turn_index=int(record.get("turn_index", 0) or 0),
                    source_clip_id=source_clip_id,
                    target_clip_id=target_clip_id,
                    split=str(record.get("split", "") or ""),
                    hardness=str(record.get("hardness", "") or ""),
                    transition_score=float(record.get("transition_score", 0.0) or 0.0),
                    structured_delta=_structured_delta_from_rows(source_row, target_row),
                    accumulated_intent_state={},
                    instruction_record=record,
                )
            )

        chain_length = max((step.turn_index for step in steps), default=len(steps))
        chains.append(
            ChainView(
                chain_id=chain_id,
                chain_length=chain_length,
                sampled_target_length=chain_length,
                split=str(first.get("split", "") or ""),
                seed_clip_id=seed_clip_id,
                seed_row=seed_row,
                steps=steps,
            )
        )

    if not chains:
        raise ValueError("No instructed chains were loaded from the instruction artifact.")

    summary = {
        "chains_loaded": len(chains),
        "referenced_clips": len(clip_ids),
        "manifest_rows_found": len(manifest_by_clip),
        "instructions_found": instructions_found,
        "total_instruction_records": total_instruction_records,
        "total_instruction_chains": len(instructed_chain_ids),
        "instructions_source": str(paths.instructions_jsonl) if paths.instructions_jsonl else None,
        "missing_manifest_row_refs": missing_manifest_rows,
        "chain_offset": chain_offset,
        "max_chains": max_chains,
        "load_mode": "instruction_records",
    }
    return DemoDataset(
        paths=paths,
        chains=chains,
        chain_ids=[chain.chain_id for chain in chains],
        manifest_by_clip=manifest_by_clip,
        summary=summary,
    )


def _load_dataset(paths: ArtifactPaths, *, chain_offset: int, max_chains: int | None) -> DemoDataset:
    instructed_chain_ids, total_instruction_records = _load_instruction_chain_ids(paths.instructions_jsonl)
    if paths.instructions_jsonl is not None and instructed_chain_ids:
        return _load_dataset_from_instruction_records(
            paths,
            chain_offset=chain_offset,
            max_chains=max_chains,
            instructed_chain_ids=instructed_chain_ids,
            total_instruction_records=total_instruction_records,
        )

    raw_chains = _load_chain_records(
        paths.chains_jsonl,
        chain_offset=chain_offset,
        max_chains=max_chains,
    )
    chain_ids = [str(record.get("chain_id", "") or "") for record in raw_chains]
    clip_ids = _referenced_clip_ids(raw_chains)
    manifest_by_clip = _load_manifest_rows(paths.manifest_csv, clip_ids)
    instruction_index = _load_instruction_index(paths.instructions_jsonl, set(chain_ids))

    chains: List[ChainView] = []
    instructions_found = 0
    missing_manifest_rows = 0

    for raw_chain in raw_chains:
        chain_id = str(raw_chain.get("chain_id", "") or "").strip()
        seed_clip_id = str(raw_chain.get("seed", {}).get("clip_id", "") or "").strip()
        seed_row = manifest_by_clip.get(seed_clip_id)
        if seed_row is None:
            missing_manifest_rows += 1

        steps: List[StepView] = []
        for raw_step in raw_chain.get("steps", []) or []:
            source_clip_id = str(raw_step.get("source_clip_id", "") or "").strip()
            target_clip_id = str(raw_step.get("target_clip_id", "") or "").strip()
            instruction_record = instruction_index.get((chain_id, int(raw_step.get("turn_index", 0) or 0)))
            if instruction_record is not None:
                instructions_found += 1
            if source_clip_id not in manifest_by_clip:
                missing_manifest_rows += 1
            if target_clip_id not in manifest_by_clip:
                missing_manifest_rows += 1
            steps.append(
                StepView(
                    turn_index=int(raw_step.get("turn_index", 0) or 0),
                    source_clip_id=source_clip_id,
                    target_clip_id=target_clip_id,
                    split=str(raw_step.get("split", "") or ""),
                    hardness=str(raw_step.get("hardness", "") or ""),
                    transition_score=float(raw_step.get("transition_score", 0.0) or 0.0),
                    structured_delta=dict(raw_step.get("structured_delta", {}) or {}),
                    accumulated_intent_state=dict(raw_step.get("accumulated_intent_state", {}) or {}),
                    instruction_record=instruction_record,
                )
            )

        chains.append(
            ChainView(
                chain_id=chain_id,
                chain_length=int(raw_chain.get("chain_length", len(steps)) or len(steps)),
                sampled_target_length=int(raw_chain.get("sampled_target_length", len(steps)) or len(steps)),
                split=str(raw_chain.get("split", "") or ""),
                seed_clip_id=seed_clip_id,
                seed_row=seed_row,
                steps=steps,
            )
        )

    summary = {
        "chains_loaded": len(chains),
        "referenced_clips": len(clip_ids),
        "manifest_rows_found": len(manifest_by_clip),
        "instructions_found": instructions_found,
        "total_instruction_records": total_instruction_records,
        "total_instruction_chains": len(instructed_chain_ids),
        "instructions_source": str(paths.instructions_jsonl) if paths.instructions_jsonl else None,
        "missing_manifest_row_refs": missing_manifest_rows,
        "chain_offset": chain_offset,
        "max_chains": max_chains,
        "load_mode": "chains_jsonl",
    }
    return DemoDataset(
        paths=paths,
        chains=chains,
        chain_ids=chain_ids,
        manifest_by_clip=manifest_by_clip,
        summary=summary,
    )


def _safe_row(row: Dict[str, str] | None) -> Dict[str, str]:
    return row or {}


def _parse_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _step_primary_edit(step: StepView) -> str:
    record = step.instruction_record or {}
    for key in ("semantic_delta_verbalized", "semantic_delta_full"):
        delta = record.get(key)
        if isinstance(delta, dict):
            primary_edit = str(delta.get("primary_edit", "") or "").strip()
            if primary_edit:
                return primary_edit
    added = [str(x).strip() for x in step.structured_delta.get("tags_added", []) if str(x).strip()]
    removed = [str(x).strip() for x in step.structured_delta.get("tags_removed", []) if str(x).strip()]
    if added and removed:
        return f"add {added[0]} and remove {removed[0]}"
    if added:
        return f"add {added[0]}"
    if removed:
        return f"remove {removed[0]}"
    return "caption or metadata shift"


def _format_caption(row: Dict[str, str]) -> str:
    return str(row.get("normalized_caption", "") or row.get("caption", "") or "").strip() or "Unavailable"


def _format_tags(row: Dict[str, str]) -> List[str]:
    tags = _parse_json_list(row.get("normalized_tags_json", ""))
    if tags:
        return tags
    raw = str(row.get("tags", "") or "").strip()
    return [part.strip() for part in raw.split(",") if part.strip()]


def _audio_preview(row: Dict[str, str] | None, *, cache_dir: Path) -> tuple[str | None, str]:
    data = _safe_row(row)
    file_path = str(data.get("file_path", "") or "").strip()
    if not file_path:
        return None, "Missing `file_path` in the manifest."

    source = Path(file_path)
    if not source.exists():
        return None, f"Audio file not found: `{source}`"

    start_time = _parse_float(data.get("start_time"))
    end_time = _parse_float(data.get("end_time"))
    if start_time is None or end_time is None or end_time <= start_time:
        return str(source), "Playing the full source file."

    try:
        import soundfile as sf
    except Exception:
        return str(source), "Clip slicing unavailable because `soundfile` is not installed; playing the full source file."

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha1(f"{source}:{start_time:.3f}:{end_time:.3f}".encode("utf-8")).hexdigest()
    clip_path = cache_dir / f"{cache_key}.wav"
    if clip_path.exists():
        return str(clip_path), f"Playing a cached {end_time - start_time:.1f}s clip preview."

    try:
        info = sf.info(str(source))
        sample_rate = int(info.samplerate)
        frame_start = max(0, int(round(start_time * sample_rate)))
        frame_stop = max(frame_start + 1, int(round(end_time * sample_rate)))
        with sf.SoundFile(str(source)) as f:
            f.seek(frame_start)
            frames = min(frame_stop - frame_start, len(f) - frame_start)
            audio = f.read(frames=frames, dtype="float32", always_2d=True)
        if len(audio) == 0:
            return str(source), "Clip window decoded empty audio; playing the full source file."
        sf.write(str(clip_path), audio, sample_rate)
        return str(clip_path), f"Playing a cached {end_time - start_time:.1f}s clip preview."
    except Exception as exc:
        return str(source), f"Clip slicing failed ({exc.__class__.__name__}); playing the full source file."


def _timeline_rows(chain: ChainView) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for step in chain.steps:
        rows.append(
            [
                step.turn_index,
                step.source_clip_id,
                step.target_clip_id,
                step.hardness or "unknown",
                round(step.transition_score, 4),
                _step_primary_edit(step),
            ]
        )
    return rows


def _empty_step_placeholder() -> StepView:
    return StepView(
        turn_index=0,
        source_clip_id="",
        target_clip_id="",
        split="",
        hardness="",
        transition_score=0.0,
        structured_delta={},
        accumulated_intent_state={},
        instruction_record=None,
    )


def _instruction_text(record: Dict[str, Any] | None, field: str) -> str:
    if not record:
        return "Instruction artifact not loaded for this step yet."
    value = str(record.get(field, "") or "").strip()
    if value:
        return value
    status = str(record.get("status", "") or "").strip()
    if status and status != "ok":
        return f"Instruction generation status: {status}"
    return "No instruction text available."


def _html(text: Any) -> str:
    return html.escape(str(text or ""), quote=True)


def _pill(text: Any, *, tone: str = "neutral") -> str:
    value = _html(text)
    if not value:
        return ""
    return f'<span class="ji-pill ji-pill-{tone}">{value}</span>'


def _pill_list(items: Sequence[Any], *, tone: str = "neutral", limit: int = 14, empty_text: str = "None") -> str:
    clean = [str(item).strip() for item in items if str(item).strip()]
    if not clean:
        return _pill(empty_text, tone="muted")
    shown = clean[:limit]
    extra = len(clean) - len(shown)
    suffix = [_pill(f"+{extra}", tone="muted")] if extra > 0 else []
    return "".join([_pill(item, tone=tone) for item in shown] + suffix)


def _instruction_axes(record: Dict[str, Any] | None) -> List[str]:
    if not record:
        return []
    axes = record.get("selected_change_axes", [])
    if isinstance(axes, list):
        return [str(axis).strip() for axis in axes if str(axis).strip()]
    return []


def _preservation_axes(record: Dict[str, Any] | None) -> List[str]:
    if not record:
        return []
    axes = record.get("selected_preservation_axes", [])
    if isinstance(axes, list):
        return [str(axis).strip() for axis in axes if str(axis).strip()]
    return []


def _delta_terms(record: Dict[str, Any] | None, field: str) -> List[str]:
    if not record:
        return []
    delta = record.get("semantic_delta_verbalized")
    if not isinstance(delta, dict):
        delta = record.get("semantic_delta_full")
    if not isinstance(delta, dict):
        return []
    values = delta.get(field, [])
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _chain_axes(chain: ChainView) -> List[str]:
    axes: List[str] = []
    for step in chain.steps:
        for axis in _instruction_axes(step.instruction_record):
            if axis not in axes:
                axes.append(axis)
    return axes


def _metric_card(label: str, value: Any, detail: str = "") -> str:
    detail_html = f'<div class="ji-metric-detail">{_html(detail)}</div>' if detail else ""
    return (
        '<div class="ji-metric">'
        f'<div class="ji-metric-label">{_html(label)}</div>'
        f'<div class="ji-metric-value">{_html(value)}</div>'
        f"{detail_html}</div>"
    )


def _app_summary_html(dataset: DemoDataset) -> str:
    run_root = str(dataset.paths.run_root) if dataset.paths.run_root else "custom paths"
    instructions_src = dataset.summary["instructions_source"] or "not provided"
    instruction_detail = "matched in loaded chains"
    total_instruction_records = int(dataset.summary.get("total_instruction_records", 0) or 0)
    if total_instruction_records:
        instruction_detail = f"{total_instruction_records:,} in artifact"
    metrics = "".join(
        [
            _metric_card("Chains", f"{dataset.summary['chains_loaded']:,}", "loaded in this page"),
            _metric_card("Clips", f"{dataset.summary['referenced_clips']:,}", "referenced by chains"),
            _metric_card("Instructions", f"{dataset.summary['instructions_found']:,}", instruction_detail),
            _metric_card("Manifest", f"{dataset.summary['manifest_rows_found']:,}", "clip rows found"),
        ]
    )
    return (
        '<section class="ji-hero">'
        '<div>'
        '<div class="ji-eyebrow">Jamendo-Instruct</div>'
        '<h1>Chain Explorer</h1>'
        '<p>Browse multi-turn music edits, inspect the requested semantic delta, and listen to source and target clips side by side.</p>'
        f'<div class="ji-path"><strong>Run:</strong> {_html(run_root)}</div>'
        f'<div class="ji-path"><strong>Instructions:</strong> {_html(instructions_src)}</div>'
        '</div>'
        f'<div class="ji-metrics">{metrics}</div>'
        '</section>'
    )


def _chain_summary_html(chain: ChainView, dataset: DemoDataset, chain_pos: int, *, total_chains: int | None = None) -> str:
    axes = _chain_axes(chain)
    loaded_start = int(dataset.summary["chain_offset"])
    loaded_end = loaded_start + int(dataset.summary["chains_loaded"]) - 1
    chain_total = total_chains if total_chains is not None else len(dataset.chains)
    return (
        '<section class="ji-card ji-chain-card">'
        '<div class="ji-card-head">'
        '<div>'
        f'<div class="ji-eyebrow">Chain {chain_pos:,} of {chain_total:,}</div>'
        f'<h2>{_html(chain.chain_id)}</h2>'
        '</div>'
        f'<div class="ji-pill-row">{_pill(chain.split or "unknown split", tone="blue")}{_pill(f"{chain.chain_length} step(s)", tone="green")}</div>'
        '</div>'
        '<div class="ji-grid-2">'
        f'{_metric_card("Sampled Length", chain.sampled_target_length)}'
        f'{_metric_card("Loaded Slice", f"{loaded_start:,} - {loaded_end:,}")}'
        '</div>'
        '<div class="ji-section-label">Edit Axes</div>'
        f'<div class="ji-pill-row">{_pill_list(axes, tone="purple", empty_text="No instruction axes loaded")}</div>'
        '</section>'
    )


def _step_summary_html(chain: ChainView, step: StepView) -> str:
    record = step.instruction_record or {}
    status = str(record.get("status", "missing")).strip() or "missing"
    axes = _instruction_axes(record)
    preservations = _preservation_axes(record)
    new_terms = _delta_terms(record, "new")
    lost_terms = _delta_terms(record, "lost")
    preserved_terms = _delta_terms(record, "preserved")
    return (
        '<section class="ji-card ji-step-card">'
        '<div class="ji-card-head">'
        '<div>'
        f'<div class="ji-eyebrow">Step {step.turn_index} of {chain.chain_length}</div>'
        f'<h2>{_html(_step_primary_edit(step))}</h2>'
        '</div>'
        f'<div class="ji-pill-row">{_pill(status, tone="green" if status == "ok" else "muted")}{_pill(step.hardness or "unknown", tone="blue")}</div>'
        '</div>'
        '<div class="ji-grid-3">'
        f'{_metric_card("Transition Score", f"{step.transition_score:.4f}")}'
        f'{_metric_card("Source", step.source_clip_id)}'
        f'{_metric_card("Target", step.target_clip_id)}'
        '</div>'
        '<div class="ji-section-label">Change Axes</div>'
        f'<div class="ji-pill-row">{_pill_list(axes, tone="purple", empty_text="No axes")}</div>'
        '<div class="ji-section-label">Explicit Preservation Axes</div>'
        f'<div class="ji-pill-row">{_pill_list(preservations, tone="green", empty_text="None explicit")}</div>'
        '<div class="ji-delta-grid">'
        f'<div><div class="ji-section-label">New</div><div class="ji-pill-row">{_pill_list(new_terms, tone="green")}</div></div>'
        f'<div><div class="ji-section-label">Lost</div><div class="ji-pill-row">{_pill_list(lost_terms, tone="red")}</div></div>'
        f'<div><div class="ji-section-label">Preserved</div><div class="ji-pill-row">{_pill_list(preserved_terms, tone="blue", empty_text="None explicit")}</div></div>'
        '</div>'
        '</section>'
    )


def _instruction_html(title: str, text: str) -> str:
    return (
        '<section class="ji-card ji-instruction-card">'
        f'<div class="ji-section-label">{_html(title)}</div>'
        f'<p>{_html(text)}</p>'
        '</section>'
    )


def _clip_html(title: str, row: Dict[str, str] | None) -> str:
    data = _safe_row(row)
    clip_id = str(data.get("clip_id", "") or "Unavailable")
    track_id = str(data.get("track_id", "") or "Unavailable")
    artist = str(data.get("artist_name", "") or "Unknown artist")
    item_title = str(data.get("title", "") or "Untitled")
    start_time = str(data.get("start_time", "") or "").strip()
    end_time = str(data.get("end_time", "") or "").strip()
    time_window = f"{start_time}s to {end_time}s" if start_time or end_time else "Full track"
    vocals = str(data.get("vocals", "") or "unknown")
    speed = str(data.get("speed", "") or "unknown")
    caption = _format_caption(data)
    tags = _format_tags(data)
    return (
        '<section class="ji-card ji-clip-card">'
        f'<div class="ji-eyebrow">{_html(title)}</div>'
        f'<h3>{_html(artist)} / {_html(item_title)}</h3>'
        f'<div class="ji-path"><strong>Clip:</strong> {_html(clip_id)}</div>'
        f'<div class="ji-path"><strong>Track:</strong> {_html(track_id)}</div>'
        f'<div class="ji-path"><strong>Window:</strong> {_html(time_window)}</div>'
        f'<div class="ji-pill-row">{_pill(f"vocals: {vocals}", tone="blue")}{_pill(f"speed: {speed}", tone="green")}</div>'
        f'<p class="ji-caption">{_html(caption)}</p>'
        f'<div class="ji-pill-row">{_pill_list(tags, tone="neutral", limit=12)}</div>'
        '</section>'
    )


def _audio_note_html(note: str) -> str:
    return f'<div class="ji-audio-note">{_html(note)}</div>'


def _streamlit_css() -> str:
    return """
    <style>
    :root {
      color-scheme: light;
      --ji-bg: #f5f7f4;
      --ji-panel: #ffffff;
      --ji-panel-soft: #f9fbf8;
      --ji-text: #171917;
      --ji-muted: #626b62;
      --ji-line: #dfe6df;
      --ji-green: #286145;
      --ji-green-bg: #e7f3ec;
      --ji-blue: #245982;
      --ji-blue-bg: #e7f0f7;
      --ji-purple: #694882;
      --ji-purple-bg: #f0e9f5;
      --ji-red: #944746;
      --ji-red-bg: #f7e9e6;
    }
    .stApp {
      background: var(--ji-bg);
      color: var(--ji-text);
    }
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
      background: var(--ji-bg) !important;
      color: var(--ji-text) !important;
    }
    p, label, span, div {
      color: inherit;
    }
    div.block-container {
      max-width: 1480px;
      padding-top: 1.4rem;
      padding-bottom: 3rem;
    }
    .ji-hero, .ji-card {
      background: var(--ji-panel);
      border: 1px solid var(--ji-line);
      border-radius: 8px;
      box-shadow: 0 10px 24px rgba(20, 28, 22, 0.05);
    }
    .ji-hero {
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(320px, 0.8fr);
      gap: 18px;
      padding: 22px;
      margin-bottom: 14px;
    }
    .ji-card {
      padding: 16px;
      margin-bottom: 12px;
    }
    .ji-hero h1, .ji-card h2, .ji-card h3 {
      margin: 0;
      letter-spacing: 0;
      color: var(--ji-text);
    }
    .ji-hero h1 {
      font-size: 2.2rem;
      line-height: 1.05;
    }
    .ji-hero p {
      margin: 10px 0 16px;
      color: var(--ji-muted);
      max-width: 70ch;
      line-height: 1.5;
    }
    .ji-card-head {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 14px;
    }
    .ji-card h2 {
      font-size: 1.1rem;
      line-height: 1.25;
    }
    .ji-card h3 {
      font-size: 1rem;
      line-height: 1.25;
      margin: 3px 0 10px;
    }
    .ji-eyebrow, .ji-section-label {
      color: var(--ji-muted);
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .ji-section-label {
      margin: 13px 0 6px;
    }
    .ji-path {
      color: var(--ji-muted);
      font-size: 0.88rem;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }
    .ji-metrics, .ji-grid-2, .ji-grid-3, .ji-delta-grid {
      display: grid;
      gap: 10px;
    }
    .ji-metrics, .ji-grid-2 {
      grid-template-columns: repeat(2, minmax(0, 1fr));
      align-content: start;
    }
    .ji-grid-3, .ji-delta-grid {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
    .ji-metric {
      min-width: 0;
      padding: 10px 12px;
      border: 1px solid var(--ji-line);
      border-radius: 8px;
      background: var(--ji-panel-soft);
    }
    .ji-metric-label {
      color: var(--ji-muted);
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
    }
    .ji-metric-value {
      margin-top: 3px;
      color: var(--ji-text);
      font-size: 1rem;
      font-weight: 700;
      overflow-wrap: anywhere;
    }
    .ji-metric-detail {
      margin-top: 2px;
      color: var(--ji-muted);
      font-size: 0.78rem;
    }
    .ji-pill-row {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      align-items: center;
    }
    .ji-pill {
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      max-width: 100%;
      padding: 2px 8px;
      border-radius: 999px;
      border: 1px solid var(--ji-line);
      color: var(--ji-text);
      background: #f3f6f2;
      font-size: 0.76rem;
      font-weight: 650;
      overflow-wrap: anywhere;
    }
    .ji-pill-green { color: var(--ji-green); background: var(--ji-green-bg); border-color: #c7e3d2; }
    .ji-pill-blue { color: var(--ji-blue); background: var(--ji-blue-bg); border-color: #c9ddeb; }
    .ji-pill-purple { color: var(--ji-purple); background: var(--ji-purple-bg); border-color: #dfcdea; }
    .ji-pill-red { color: var(--ji-red); background: var(--ji-red-bg); border-color: #eccbc4; }
    .ji-pill-muted { color: var(--ji-muted); background: #eef1ed; }
    .ji-caption {
      margin: 6px 0 0;
      color: #283028;
      line-height: 1.48;
    }
    .ji-instruction-card {
      min-height: 150px;
    }
    .ji-instruction-card p {
      margin: 8px 0 0;
      color: #202720;
      font-size: 1.03rem;
      line-height: 1.55;
    }
    .ji-clip-card {
      min-height: 310px;
    }
    .ji-audio-note {
      color: var(--ji-muted);
      font-size: 0.83rem;
      margin-top: -0.35rem;
      min-height: 22px;
    }
    [data-testid="stSidebar"] {
      background: #eef3ee;
      border-right: 1px solid var(--ji-line);
    }
    [data-testid="stMetricValue"] {
      font-size: 1.05rem;
      color: var(--ji-text) !important;
    }
    .stButton > button {
      background: var(--ji-panel) !important;
      color: var(--ji-text) !important;
      border: 1px solid #b9cabb !important;
      border-radius: 8px !important;
      box-shadow: none !important;
    }
    .stButton > button:hover {
      background: var(--ji-green-bg) !important;
      color: var(--ji-green) !important;
      border-color: #9fc9af !important;
    }
    .stSelectbox [data-baseweb="select"],
    .stNumberInput input,
    .stSlider,
    .stTextInput input {
      color: var(--ji-text) !important;
    }
    @media (max-width: 980px) {
      .ji-hero, .ji-grid-2, .ji-grid-3, .ji-delta-grid {
        grid-template-columns: 1fr;
      }
      .ji-metrics {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }
    @media (max-width: 620px) {
      .ji-metrics {
        grid-template-columns: 1fr;
      }
      .ji-hero {
        padding: 16px;
      }
      .ji-hero h1 {
        font-size: 1.7rem;
      }
      .ji-card-head {
        display: block;
      }
      .ji-card-head .ji-pill-row {
        margin-top: 10px;
      }
    }
    </style>
    """


def _step_has_instruction(step: StepView) -> bool:
    return step.instruction_record is not None


def _visible_chains(dataset: DemoDataset, *, instructions_only: bool) -> List[ChainView]:
    if not instructions_only:
        return list(dataset.chains)
    return [chain for chain in dataset.chains if any(_step_has_instruction(step) for step in chain.steps)]


def _visible_steps(chain: ChainView, *, instructions_only: bool) -> List[StepView]:
    if not instructions_only:
        return list(chain.steps)
    return [step for step in chain.steps if _step_has_instruction(step)]


def _timeline_dicts(steps: Sequence[StepView]) -> List[Dict[str, Any]]:
    return [
        {
            "turn": step.turn_index,
            "source_clip_id": step.source_clip_id,
            "target_clip_id": step.target_clip_id,
            "hardness": step.hardness or "unknown",
            "score": round(step.transition_score, 4),
            "primary_edit": _step_primary_edit(step),
            "has_instruction": _step_has_instruction(step),
        }
        for step in steps
    ]


def _counter_rows(counter: Counter[str], label: str, *, limit: int = 25) -> List[Dict[str, Any]]:
    return [{label: key, "count": count} for key, count in counter.most_common(limit)]


def _clip_duration(row: Dict[str, str]) -> float | None:
    start = _parse_float(row.get("start_time"))
    end = _parse_float(row.get("end_time"))
    if start is None or end is None or end <= start:
        return None
    return end - start


def _caption_word_count(row: Dict[str, str]) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", _format_caption(row)))


def _analysis_manifest_rows(dataset: DemoDataset, keep_clip_ids: set[str] | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for clip_id, row in dataset.manifest_by_clip.items():
        if keep_clip_ids is not None and clip_id not in keep_clip_ids:
            continue
        tags = _format_tags(row)
        rows.append(
            {
                "clip_id": str(row.get("clip_id", "") or ""),
                "track_id": str(row.get("track_id", "") or ""),
                "artist_id": str(row.get("artist_id", "") or ""),
                "artist_name": str(row.get("artist_name", "") or ""),
                "split": str(row.get("split", "") or "unknown"),
                "vocals": str(row.get("vocals", "") or "unknown"),
                "speed": str(row.get("speed", "") or "unknown"),
                "lyrics_status": str(row.get("lyrics_status", "") or "unknown"),
                "tag_count": len(tags),
                "caption_words": _caption_word_count(row),
                "duration_sec": _clip_duration(row),
                "has_file_path": bool(str(row.get("file_path", "") or "").strip()),
                "has_caption": bool(_format_caption(row) != "Unavailable"),
                "tags": tags,
            }
        )
    return rows


def _analysis_chains_limited(chains: Sequence[ChainView], max_steps: int) -> List[ChainView]:
    remaining = max(0, int(max_steps))
    selected: List[ChainView] = []
    for chain in chains:
        if remaining <= 0:
            break
        steps = list(chain.steps[:remaining])
        if not steps:
            continue
        selected.append(
            ChainView(
                chain_id=chain.chain_id,
                chain_length=chain.chain_length,
                sampled_target_length=chain.sampled_target_length,
                split=chain.split,
                seed_clip_id=chain.seed_clip_id,
                seed_row=chain.seed_row,
                steps=steps,
            )
        )
        remaining -= len(steps)
    return selected


def _analysis_clip_ids(chains: Sequence[ChainView]) -> set[str]:
    clip_ids: set[str] = set()
    for chain in chains:
        if chain.seed_clip_id:
            clip_ids.add(chain.seed_clip_id)
        for step in chain.steps:
            if step.source_clip_id:
                clip_ids.add(step.source_clip_id)
            if step.target_clip_id:
                clip_ids.add(step.target_clip_id)
    return clip_ids


def _semantic_delta_counts(record: Dict[str, Any] | None, field: str) -> int:
    if not record:
        return 0
    delta = record.get("semantic_delta_verbalized")
    if not isinstance(delta, dict):
        delta = record.get("semantic_delta_full")
    if not isinstance(delta, dict):
        return 0
    values = delta.get(field, [])
    return len(values) if isinstance(values, list) else 0


def _caption_only_change(record: Dict[str, Any] | None) -> bool:
    if not record:
        return False
    for key in ("semantic_delta_verbalized", "semantic_delta_full"):
        delta = record.get(key)
        if isinstance(delta, dict) and isinstance(delta.get("caption_only_change"), bool):
            return bool(delta.get("caption_only_change"))
    return False


def _analysis_step_rows(chains: Sequence[ChainView], dataset: DemoDataset) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for chain in chains:
        for step in chain.steps:
            record = step.instruction_record
            axes = _instruction_axes(record)
            preservations = _preservation_axes(record)
            source_row = dataset.manifest_by_clip.get(step.source_clip_id, {})
            target_row = dataset.manifest_by_clip.get(step.target_clip_id, {})
            unaware = _instruction_text(record, "history_unaware_instruction") if record else ""
            aware = _instruction_text(record, "history_aware_instruction") if record else ""
            structured = step.structured_delta or {}
            rows.append(
                {
                    "chain_id": chain.chain_id,
                    "turn_index": step.turn_index,
                    "split": step.split or chain.split or "unknown",
                    "hardness": step.hardness or "unknown",
                    "transition_score": step.transition_score,
                    "source_clip_id": step.source_clip_id,
                    "target_clip_id": step.target_clip_id,
                    "has_instruction": record is not None,
                    "status": str((record or {}).get("status", "missing") or "missing"),
                    "change_axes": axes,
                    "primary_axis": axes[0] if axes else "none",
                    "change_axis_count": len(axes),
                    "preservation_axes": preservations,
                    "caption_only_change": _caption_only_change(record),
                    "new_count": _semantic_delta_counts(record, "new"),
                    "lost_count": _semantic_delta_counts(record, "lost"),
                    "preserved_count": _semantic_delta_counts(record, "preserved"),
                    "tags_added_count": len(structured.get("tags_added", []) or []),
                    "tags_removed_count": len(structured.get("tags_removed", []) or []),
                    "tags_preserved_count": len(structured.get("tags_preserved", []) or []),
                    "source_vocals": str(structured.get("source_vocals") or source_row.get("vocals") or "unknown"),
                    "target_vocals": str(structured.get("target_vocals") or target_row.get("vocals") or "unknown"),
                    "source_speed": str(structured.get("source_speed") or source_row.get("speed") or "unknown"),
                    "target_speed": str(structured.get("target_speed") or target_row.get("speed") or "unknown"),
                    "primary_edit": _step_primary_edit(step),
                    "history_unaware_words": len(re.findall(r"[A-Za-z0-9']+", unaware)),
                    "history_aware_words": len(re.findall(r"[A-Za-z0-9']+", aware)),
                }
            )
    return rows


def _flatten_terms_from_records(chains: Sequence[ChainView], field: str) -> Counter[str]:
    terms: Counter[str] = Counter()
    for chain in chains:
        for step in chain.steps:
            for term in _delta_terms(step.instruction_record, field):
                terms[term] += 1
    return terms


def _word_counter(texts: Sequence[str], *, limit: int = 40) -> Counter[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "be",
        "but",
        "for",
        "from",
        "in",
        "into",
        "it",
        "keep",
        "less",
        "make",
        "more",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
    counts: Counter[str] = Counter()
    for text in texts:
        for token in re.findall(r"[A-Za-z][A-Za-z']+", str(text).lower()):
            if len(token) < 3 or token in stopwords:
                continue
            counts[token] += 1
    return Counter(dict(counts.most_common(limit)))


def _render_plotly_bar(st: Any, px: Any, rows: List[Dict[str, Any]], *, x: str, y: str, title: str, color: str | None = None) -> None:
    if not rows:
        st.info(f"No data for {title.lower()}.")
        return
    fig = px.bar(rows, x=x, y=y, color=color, title=title)
    fig.update_layout(margin=dict(l=10, r=10, t=48, b=10), height=360)
    st.plotly_chart(fig, width="stretch")


def _render_plotly_hist(st: Any, px: Any, rows: List[Dict[str, Any]], *, x: str, title: str, color: str | None = None) -> None:
    values = [row for row in rows if row.get(x) is not None]
    if not values:
        st.info(f"No data for {title.lower()}.")
        return
    fig = px.histogram(values, x=x, color=color, nbins=40, title=title)
    fig.update_layout(margin=dict(l=10, r=10, t=48, b=10), height=360)
    st.plotly_chart(fig, width="stretch")


def _render_analysis_tab(st: Any, dataset: DemoDataset, visible_chains: Sequence[ChainView]) -> None:
    try:
        import pandas as pd
        import plotly.express as px
    except Exception as exc:
        st.warning(f"Analysis plots need pandas and plotly in the Streamlit environment ({exc.__class__.__name__}).")
        return

    total_available_steps = sum(len(chain.steps) for chain in visible_chains)
    if total_available_steps <= 0:
        st.info("No steps are available for analysis with the current sidebar filter.")
        return
    if total_available_steps > 5000:
        analysis_limit = st.slider(
            "Analysis step cap",
            min_value=500,
            max_value=total_available_steps,
            value=min(5000, total_available_steps),
            step=500,
            help="Keeps the analysis tab responsive on large full-run loads. The explorer still uses the full loaded slice.",
        )
    else:
        analysis_limit = total_available_steps

    chains = _analysis_chains_limited(visible_chains, analysis_limit)
    manifest_rows = _analysis_manifest_rows(dataset, _analysis_clip_ids(chains))
    step_rows = _analysis_step_rows(chains, dataset)
    step_df = pd.DataFrame(step_rows)

    st.subheader("Analysis")
    st.caption(
        "Charts reflect the sidebar instruction filter and the analysis step cap; "
        "the explorer still browses the full loaded slice."
    )

    total_steps = len(step_rows)
    instructed_steps = int(sum(1 for row in step_rows if row["has_instruction"]))
    unique_tracks = len({row["track_id"] for row in manifest_rows if row["track_id"]})
    unique_artists = len({row["artist_id"] or row["artist_name"] for row in manifest_rows if row["artist_id"] or row["artist_name"]})
    avg_score = sum(float(row["transition_score"]) for row in step_rows) / max(1, total_steps)
    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Chains", f"{len(chains):,}")
    kpi_cols[1].metric("Steps", f"{total_steps:,}")
    kpi_cols[2].metric("Instructed", f"{instructed_steps:,}", f"{instructed_steps / max(1, total_steps):.1%}")
    kpi_cols[3].metric("Tracks", f"{unique_tracks:,}", f"{unique_artists:,} artists")
    kpi_cols[4].metric("Mean Score", f"{avg_score:.4f}")

    overview_tab, metadata_tab, steps_tab, instruction_tab, quality_tab = st.tabs(
        ["Overview", "Metadata", "Steps", "Instructions", "Quality"]
    )

    with overview_tab:
        left, right = st.columns(2)
        with left:
            _render_plotly_hist(st, px, step_rows, x="transition_score", color="hardness", title="Transition Score Distribution")
        with right:
            chain_lengths = [
                {"chain_id": chain.chain_id, "chain_length": len(chain.steps), "split": chain.split or "unknown"}
                for chain in chains
            ]
            _render_plotly_hist(st, px, chain_lengths, x="chain_length", color="split", title="Chain Length Distribution")
        left, right = st.columns(2)
        with left:
            hardness_counts = Counter(row["hardness"] for row in step_rows)
            _render_plotly_bar(st, px, _counter_rows(hardness_counts, "hardness"), x="hardness", y="count", title="Hardness Mix")
        with right:
            split_counts = Counter(row["split"] for row in step_rows)
            _render_plotly_bar(st, px, _counter_rows(split_counts, "split"), x="split", y="count", title="Step Split Mix")

    with metadata_tab:
        tag_counts: Counter[str] = Counter()
        for row in manifest_rows:
            tag_counts.update(row["tags"])
        left, right = st.columns(2)
        with left:
            _render_plotly_bar(st, px, _counter_rows(tag_counts, "tag", limit=30), x="count", y="tag", title="Top Tags")
        with right:
            _render_plotly_hist(st, px, manifest_rows, x="tag_count", title="Tags Per Referenced Clip")
        left, right = st.columns(2)
        with left:
            _render_plotly_bar(st, px, _counter_rows(Counter(row["vocals"] for row in manifest_rows), "vocals"), x="vocals", y="count", title="Vocals Distribution")
        with right:
            _render_plotly_bar(st, px, _counter_rows(Counter(row["speed"] for row in manifest_rows), "speed"), x="speed", y="count", title="Speed Distribution")
        left, right = st.columns(2)
        with left:
            _render_plotly_hist(st, px, manifest_rows, x="caption_words", title="Caption Length")
        with right:
            caption_words = _word_counter([_format_caption(dataset.manifest_by_clip.get(row["clip_id"], {})) for row in manifest_rows])
            _render_plotly_bar(st, px, _counter_rows(caption_words, "word", limit=30), x="count", y="word", title="Common Caption Terms")

    with steps_tab:
        left, right = st.columns(2)
        with left:
            _render_plotly_bar(st, px, _counter_rows(Counter(row["primary_axis"] for row in step_rows), "axis"), x="axis", y="count", title="Primary Change Axis")
        with right:
            if not step_df.empty:
                fig = px.box(step_df, x="primary_axis", y="transition_score", color="hardness", title="Score by Axis and Hardness")
                fig.update_layout(margin=dict(l=10, r=10, t=48, b=10), height=420)
                st.plotly_chart(fig, width="stretch")
        left, right = st.columns(2)
        with left:
            vocal_moves = Counter(f"{row['source_vocals']} -> {row['target_vocals']}" for row in step_rows)
            _render_plotly_bar(st, px, _counter_rows(vocal_moves, "transition", limit=20), x="count", y="transition", title="Vocals Transitions")
        with right:
            speed_moves = Counter(f"{row['source_speed']} -> {row['target_speed']}" for row in step_rows)
            _render_plotly_bar(st, px, _counter_rows(speed_moves, "transition", limit=20), x="count", y="transition", title="Speed Transitions")
        left, right = st.columns(2)
        with left:
            _render_plotly_hist(st, px, step_rows, x="tags_added_count", title="Tags Added Per Step")
        with right:
            _render_plotly_hist(st, px, step_rows, x="tags_removed_count", title="Tags Removed Per Step")

    with instruction_tab:
        left, right = st.columns(2)
        with left:
            _render_plotly_bar(st, px, _counter_rows(Counter(row["primary_axis"] for row in step_rows if row["has_instruction"]), "axis"), x="axis", y="count", title="Instruction Axes")
        with right:
            _render_plotly_bar(st, px, _counter_rows(Counter(row["change_axis_count"] for row in step_rows if row["has_instruction"]), "axis_count"), x="axis_count", y="count", title="Single vs Multi-Axis Instructions")
        left, right = st.columns(2)
        with left:
            _render_plotly_hist(st, px, [row for row in step_rows if row["has_instruction"]], x="history_unaware_words", color="primary_axis", title="History-Unaware Length")
        with right:
            _render_plotly_hist(st, px, [row for row in step_rows if row["has_instruction"]], x="history_aware_words", color="primary_axis", title="History-Aware Length")
        semantic_cols = st.columns(3)
        semantic_cols[0].dataframe(_counter_rows(_flatten_terms_from_records(chains, "new"), "new_term", limit=25), width="stretch", hide_index=True)
        semantic_cols[1].dataframe(_counter_rows(_flatten_terms_from_records(chains, "lost"), "lost_term", limit=25), width="stretch", hide_index=True)
        semantic_cols[2].dataframe(_counter_rows(_flatten_terms_from_records(chains, "preserved"), "preserved_term", limit=25), width="stretch", hide_index=True)
        caption_only = Counter("caption_only" if row["caption_only_change"] else "metadata_or_mixed" for row in step_rows if row["has_instruction"])
        _render_plotly_bar(st, px, _counter_rows(caption_only, "type"), x="type", y="count", title="Caption-Only Change Rate")

    with quality_tab:
        st.caption("Quick checks for records that are often worth inspecting by hand.")
        missing_axis = [row for row in step_rows if row["has_instruction"] and row["primary_axis"] == "none"]
        generic_primary = [row for row in step_rows if row["primary_edit"] in {"caption or metadata shift", "refines the current sound"}]
        long_instruction = sorted(
            [row for row in step_rows if row["has_instruction"]],
            key=lambda row: max(row["history_unaware_words"], row["history_aware_words"]),
            reverse=True,
        )[:50]
        score_outliers = sorted(step_rows, key=lambda row: row["transition_score"])[:25] + sorted(
            step_rows, key=lambda row: row["transition_score"], reverse=True
        )[:25]
        issue_cols = st.columns(4)
        issue_cols[0].metric("Missing Axis", f"{len(missing_axis):,}")
        issue_cols[1].metric("Generic Primary Edit", f"{len(generic_primary):,}")
        issue_cols[2].metric("Caption-Only", f"{sum(1 for row in step_rows if row['caption_only_change']):,}")
        issue_cols[3].metric("No Instruction", f"{sum(1 for row in step_rows if not row['has_instruction']):,}")
        table_choice = st.selectbox(
            "Inspect",
            ["Score outliers", "Longest instructions", "Missing axis", "Generic primary edit"],
        )
        selected_rows = {
            "Score outliers": score_outliers,
            "Longest instructions": long_instruction,
            "Missing axis": missing_axis,
            "Generic primary edit": generic_primary,
        }[table_choice]
        columns = [
            "chain_id",
            "turn_index",
            "hardness",
            "transition_score",
            "primary_axis",
            "primary_edit",
            "source_clip_id",
            "target_clip_id",
        ]
        st.dataframe(pd.DataFrame(selected_rows)[columns] if selected_rows else pd.DataFrame(columns=columns), width="stretch", hide_index=True)


def _streamlit_runtime_active() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False
    return get_script_run_ctx(suppress_warning=True) is not None


def _load_dataset_for_streamlit(
    run_root: str | None,
    manifest_csv: str | None,
    chains_jsonl: str | None,
    instructions_jsonl: str | None,
    chain_offset: int,
    max_chains: int | None,
) -> DemoDataset:
    args = argparse.Namespace(
        run_root=run_root,
        manifest_csv=manifest_csv,
        chains_jsonl=chains_jsonl,
        instructions_jsonl=instructions_jsonl,
    )
    paths = _resolve_paths(args)
    return _load_dataset(paths, chain_offset=chain_offset, max_chains=max_chains)


def _render_clip_panel(st: Any, title: str, row: Dict[str, str] | None, *, cache_dir: Path) -> None:
    audio_path, note = _audio_preview(row, cache_dir=cache_dir)
    st.markdown(_clip_html(title, row), unsafe_allow_html=True)
    if audio_path:
        st.audio(audio_path)
    st.markdown(_audio_note_html(note), unsafe_allow_html=True)


def _render_streamlit_app(args: argparse.Namespace) -> None:
    try:
        import streamlit as st
    except Exception as exc:
        raise RuntimeError(
            "Streamlit is required for the demo. Install it with `pip install -e .[demo]` "
            "or add `streamlit` to the current environment."
        ) from exc

    st.set_page_config(page_title="Jamendo-Instruct Chain Explorer", layout="wide")
    st.markdown(_streamlit_css(), unsafe_allow_html=True)

    max_chains = None if args.max_chains is not None and args.max_chains <= 0 else args.max_chains
    default_run_root = str(Path(args.run_root).expanduser()) if args.run_root else ""

    if "active_run_root" not in st.session_state:
        st.session_state.active_run_root = default_run_root
    if "run_root_input" not in st.session_state:
        st.session_state.run_root_input = st.session_state.active_run_root
    if "active_instruction_folder" not in st.session_state:
        st.session_state.active_instruction_folder = ""

    @st.cache_data(show_spinner="Loading active instruction chains into memory...")
    def _cached_dataset(
        run_root: str | None,
        manifest_csv: str | None,
        chains_jsonl: str | None,
        instructions_jsonl: str | None,
        chain_offset: int,
        max_chains_value: int | None,
    ) -> DemoDataset:
        return _load_dataset_for_streamlit(
            run_root,
            manifest_csv,
            chains_jsonl,
            instructions_jsonl,
            chain_offset,
            max_chains_value,
        )

    with st.sidebar:
        st.header("Data")
        st.text_input(
            "Run folder",
            key="run_root_input",
            placeholder="/path/to/run_root",
            help="Folder containing structured_view/, chains/, and optionally instructions/.",
        )
        load_requested = st.button("Load Folder", use_container_width=True)
        if load_requested:
            st.session_state.active_run_root = str(st.session_state.run_root_input or "").strip()
            st.session_state.active_instruction_folder = ""

        active_run_root = str(st.session_state.active_run_root or "").strip() or None
        instruction_folders = _instruction_folder_options(active_run_root)
        instruction_folder_names = [folder.name for folder in instruction_folders]
        if instruction_folder_names:
            if st.session_state.active_instruction_folder not in instruction_folder_names:
                default_folder = "instructions" if "instructions" in instruction_folder_names else instruction_folder_names[0]
                st.session_state.active_instruction_folder = default_folder
            selected_instruction_folder = st.selectbox(
                "Instruction folder",
                options=instruction_folder_names,
                index=instruction_folder_names.index(st.session_state.active_instruction_folder),
                help="Switch between instruction experiment outputs under the selected run folder.",
            )
            if selected_instruction_folder != st.session_state.active_instruction_folder:
                st.session_state.active_instruction_folder = selected_instruction_folder
                st.rerun()
        elif active_run_root:
            st.caption("No instruction folders found under this run folder.")

    active_run_root = str(st.session_state.active_run_root or "").strip() or None
    active_manifest_csv = None if active_run_root else args.manifest_csv
    active_chains_jsonl = None if active_run_root else args.chains_jsonl
    if active_run_root and st.session_state.active_instruction_folder:
        active_instructions_jsonl = str(
            Path(active_run_root) / st.session_state.active_instruction_folder / "chain_step_instructions.jsonl"
        )
    else:
        active_instructions_jsonl = None if active_run_root else args.instructions_jsonl
    try:
        dataset = _cached_dataset(
            active_run_root,
            active_manifest_csv,
            active_chains_jsonl,
            active_instructions_jsonl,
            max(0, int(args.chain_offset)),
            max_chains,
        )
    except (FileNotFoundError, ValueError) as exc:
        with st.sidebar:
            st.error(str(exc))
        st.stop()

    cache_dir = Path(tempfile.gettempdir()) / "jamendo_instruct_chain_demo"
    dataset_key = (
        str(dataset.paths.run_root or ""),
        str(dataset.paths.manifest_csv),
        str(dataset.paths.chains_jsonl),
        str(dataset.paths.instructions_jsonl or ""),
    )

    if st.session_state.get("dataset_key") != dataset_key:
        st.session_state.dataset_key = dataset_key
        if st.session_state.get("chain_id") not in dataset.chain_ids:
            st.session_state.chain_id = dataset.chain_ids[0]
            st.session_state.step_pos = 1
    if "chain_id" not in st.session_state:
        st.session_state.chain_id = dataset.chain_ids[0]
    if "step_pos" not in st.session_state:
        st.session_state.step_pos = 1

    with st.sidebar:
        st.header("Browse")
        instructions_only = st.toggle(
            "Only show instructed steps",
            value=True,
            help="Limit the chain list and step controls to steps with matched instruction records.",
        )
        visible_chains = _visible_chains(dataset, instructions_only=instructions_only)
        if not visible_chains:
            st.warning("No instruction records were found in this loaded chain slice.")
            st.caption("Increase --max-chains, change --chain-offset, or turn off the filter after loading instructions.")
            return

        visible_chain_ids = [chain.chain_id for chain in visible_chains]
        if st.session_state.chain_id not in visible_chain_ids:
            st.session_state.chain_id = visible_chain_ids[0]
            st.session_state.step_pos = 1

        current_chain_pos = visible_chain_ids.index(st.session_state.chain_id) + 1
        selected_chain_id = st.selectbox(
            "Chain",
            options=visible_chain_ids,
            index=current_chain_pos - 1,
        )
        if selected_chain_id != st.session_state.chain_id:
            st.session_state.chain_id = selected_chain_id
            st.session_state.step_pos = 1
            st.rerun()

        prev_col, next_col = st.columns(2)
        if prev_col.button("Previous", use_container_width=True):
            previous_pos = max(1, current_chain_pos - 1)
            st.session_state.chain_id = visible_chain_ids[previous_pos - 1]
            st.session_state.step_pos = 1
            st.rerun()
        if next_col.button("Next", use_container_width=True):
            next_pos = min(len(visible_chains), current_chain_pos + 1)
            st.session_state.chain_id = visible_chain_ids[next_pos - 1]
            st.session_state.step_pos = 1
            st.rerun()

        if len(visible_chains) > 1:
            position = st.number_input(
                "Chain Position",
                min_value=1,
                max_value=len(visible_chains),
                value=current_chain_pos,
                step=1,
            )
            if int(position) != current_chain_pos:
                st.session_state.chain_id = visible_chain_ids[int(position) - 1]
                st.session_state.step_pos = 1
                st.rerun()
        else:
            st.caption("Chain Position: 1 / 1")

        st.divider()
        st.caption("Loaded slice")
        st.metric("Chains", f"{dataset.summary['chains_loaded']:,}")
        st.metric("Visible Chains", f"{len(visible_chains):,}")
        st.metric("Instructions", f"{dataset.summary['instructions_found']:,}")
        if int(dataset.summary.get("total_instruction_records", 0) or 0):
            st.metric("Instruction Artifact", f"{dataset.summary['total_instruction_records']:,}")
        st.metric("Manifest Rows", f"{dataset.summary['manifest_rows_found']:,}")

    visible_chain_ids = [chain.chain_id for chain in visible_chains]
    chain = visible_chains[visible_chain_ids.index(st.session_state.chain_id)]
    visible_steps = _visible_steps(chain, instructions_only=instructions_only)
    if not visible_steps:
        st.warning("This chain has no visible steps for the current filter.")
        return
    current_chain_pos = visible_chain_ids.index(chain.chain_id) + 1
    step_count = len(visible_steps)
    st.session_state.step_pos = max(1, min(st.session_state.step_pos, step_count))
    step = visible_steps[st.session_state.step_pos - 1]
    record = step.instruction_record

    st.markdown(_app_summary_html(dataset), unsafe_allow_html=True)
    explorer_tab, analysis_tab = st.tabs(["Explorer", "Analysis"])

    with explorer_tab:
        instruction_left, instruction_right = st.columns(2, gap="large")
        with instruction_left:
            st.markdown(
                _instruction_html("History-Unaware Instruction", _instruction_text(record, "history_unaware_instruction")),
                unsafe_allow_html=True,
            )
        with instruction_right:
            st.markdown(
                _instruction_html("History-Aware Instruction", _instruction_text(record, "history_aware_instruction")),
                unsafe_allow_html=True,
            )

        source_col, target_col = st.columns(2, gap="large")
        with source_col:
            _render_clip_panel(st, "Source Clip", dataset.manifest_by_clip.get(step.source_clip_id), cache_dir=cache_dir)
        with target_col:
            _render_clip_panel(st, "Target Clip", dataset.manifest_by_clip.get(step.target_clip_id), cache_dir=cache_dir)

        step_prev, step_mid, step_next = st.columns([1, 1.4, 1])
        if step_prev.button("Previous Step", width="stretch"):
            st.session_state.step_pos = max(1, st.session_state.step_pos - 1)
            st.rerun()
        visible_label = "instructed step" if instructions_only else "step"
        step_mid.markdown(
            _metric_card(
                "Step",
                f"{st.session_state.step_pos:,} / {step_count:,}",
                f"{visible_label}; original turn {step.turn_index}",
            ),
            unsafe_allow_html=True,
        )
        if step_next.button("Next Step", width="stretch"):
            st.session_state.step_pos = min(step_count, st.session_state.step_pos + 1)
            st.rerun()

        chain_prev, chain_next = st.columns(2)
        if chain_prev.button("Previous Chain", width="stretch"):
            previous_pos = max(1, current_chain_pos - 1)
            st.session_state.chain_id = visible_chain_ids[previous_pos - 1]
            st.session_state.step_pos = 1
            st.rerun()
        if chain_next.button("Next Chain", width="stretch"):
            next_pos = min(len(visible_chains), current_chain_pos + 1)
            st.session_state.chain_id = visible_chain_ids[next_pos - 1]
            st.session_state.step_pos = 1
            st.rerun()

        st.caption(
            f"Chain {current_chain_pos} / {len(visible_chains)} · "
            f"{visible_label} {st.session_state.step_pos} / {step_count} · original turn {step.turn_index}"
        )

        st.subheader("Timeline")
        st.dataframe(_timeline_dicts(visible_steps), width="stretch", hide_index=True)

        summary_left, summary_right = st.columns([1.15, 0.85])
        with summary_left:
            st.markdown(
                _chain_summary_html(chain, dataset, current_chain_pos, total_chains=len(visible_chains)),
                unsafe_allow_html=True,
            )
        with summary_right:
            st.markdown(_step_summary_html(chain, step), unsafe_allow_html=True)

        with st.expander("Semantic and Chain Internals"):
            json_left, json_right = st.columns(2)
            with json_left:
                st.caption("Semantic Delta Full")
                st.json(dict(record.get("semantic_delta_full", {}) or {}) if record else {})
                st.caption("Structured Delta")
                st.json(dict(step.structured_delta or {}))
            with json_right:
                st.caption("Semantic Delta Verbalized")
                st.json(dict(record.get("semantic_delta_verbalized", {}) or {}) if record else {})
                st.caption("Accumulated Intent State")
                st.json(dict(step.accumulated_intent_state or {}))

    with analysis_tab:
        _render_analysis_tab(st, dataset, visible_chains)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch a Streamlit demo for browsing sampled Jamendo chains.")
    parser.add_argument("--run-root", help="Run artifact root, e.g. /path/to/<run_name>.")
    parser.add_argument("--manifest-csv", help="Explicit path to structured_clip_manifest.csv.")
    parser.add_argument("--chains-jsonl", help="Explicit path to sampled_chains.jsonl.")
    parser.add_argument("--instructions-jsonl", help="Optional path to chain_step_instructions.jsonl.")
    parser.add_argument("--chain-offset", type=int, default=0, help="How many chains to skip before loading.")
    parser.add_argument(
        "--max-chains",
        type=int,
        default=0,
        help="Maximum number of chains to load into memory at startup; 0 loads all chains.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host for the Streamlit app.")
    parser.add_argument("--port", type=int, default=7860, help="Server port for the Streamlit app.")
    return parser


def _streamlit_forwarded_args(args: argparse.Namespace) -> List[str]:
    forwarded: List[str] = []
    for attr, flag in (
        ("run_root", "--run-root"),
        ("manifest_csv", "--manifest-csv"),
        ("chains_jsonl", "--chains-jsonl"),
        ("instructions_jsonl", "--instructions-jsonl"),
    ):
        value = getattr(args, attr)
        if value:
            forwarded.extend([flag, str(value)])
    forwarded.extend(["--chain-offset", str(max(0, int(args.chain_offset)))])
    forwarded.extend(["--max-chains", str(int(args.max_chains))])
    forwarded.extend(["--host", str(args.host)])
    forwarded.extend(["--port", str(int(args.port))])
    return forwarded


def _launch_streamlit(args: argparse.Namespace) -> None:
    try:
        from streamlit.web import cli as stcli
    except Exception as exc:
        raise RuntimeError(
            "Streamlit is required for the demo. Install it with `pip install -e .[demo]` "
            "or add `streamlit` to the current environment."
        ) from exc

    sys.argv = [
        "streamlit",
        "run",
        str(Path(__file__).resolve()),
        "--server.address",
        str(args.host),
        "--server.port",
        str(int(args.port)),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--theme.base",
        "light",
        "--theme.backgroundColor",
        "#f5f7f4",
        "--theme.secondaryBackgroundColor",
        "#ffffff",
        "--theme.textColor",
        "#171917",
        "--theme.primaryColor",
        "#286145",
        "--",
        *_streamlit_forwarded_args(args),
    ]
    stcli.main()


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if _streamlit_runtime_active():
        _render_streamlit_app(args)
    else:
        _launch_streamlit(args)


if __name__ == "__main__":
    main()
