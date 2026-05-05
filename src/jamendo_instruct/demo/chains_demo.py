from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
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
        instructions_jsonl = instructions_jsonl or (inferred_instructions if inferred_instructions.exists() else None)

    if manifest_csv is None or chains_jsonl is None:
        raise ValueError("Provide --run-root or both --manifest-csv and --chains-jsonl.")

    if not manifest_csv.exists():
        raise FileNotFoundError(f"Structured manifest not found: {manifest_csv}")
    if not chains_jsonl.exists():
        raise FileNotFoundError(f"Chains artifact not found: {chains_jsonl}")
    if instructions_jsonl is not None and not instructions_jsonl.exists():
        raise FileNotFoundError(f"Instructions artifact not found: {instructions_jsonl}")

    return ArtifactPaths(
        run_root=run_root,
        manifest_csv=manifest_csv,
        chains_jsonl=chains_jsonl,
        instructions_jsonl=instructions_jsonl,
    )


def _load_chain_records(path: Path, *, chain_offset: int, max_chains: int | None) -> List[Dict[str, Any]]:
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
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip_id = str(row.get("clip_id", "") or "").strip()
            if clip_id and clip_id in keep_clip_ids:
                out[clip_id] = row
    return out


def _load_instruction_index(path: Path | None, keep_chain_ids: set[str]) -> Dict[tuple[str, int], Dict[str, Any]]:
    if path is None:
        return {}
    out: Dict[tuple[str, int], Dict[str, Any]] = {}
    for record in _iter_jsonl(path):
        chain_id = str(record.get("chain_id", "") or "").strip()
        if not chain_id or chain_id not in keep_chain_ids:
            continue
        turn_index = int(record.get("turn_index", 0) or 0)
        out[(chain_id, turn_index)] = record
    return out


def _load_dataset(paths: ArtifactPaths, *, chain_offset: int, max_chains: int | None) -> DemoDataset:
    raw_chains = _load_chain_records(paths.chains_jsonl, chain_offset=chain_offset, max_chains=max_chains)
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
        "instructions_source": str(paths.instructions_jsonl) if paths.instructions_jsonl else None,
        "missing_manifest_row_refs": missing_manifest_rows,
        "chain_offset": chain_offset,
        "max_chains": max_chains,
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


def _bullet_list(items: Sequence[str], *, empty_text: str = "None") -> str:
    clean = [str(item).strip() for item in items if str(item).strip()]
    if not clean:
        return empty_text
    return "\n".join(f"- `{item}`" for item in clean)


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


def _clip_markdown(title: str, row: Dict[str, str] | None) -> str:
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
        f"### {title}\n"
        f"**Clip:** `{clip_id}`  \n"
        f"**Track:** `{track_id}`  \n"
        f"**Artist / Title:** {artist} / {item_title}  \n"
        f"**Window:** {time_window}  \n"
        f"**Vocals / Speed:** `{vocals}` / `{speed}`\n\n"
        f"**Caption**\n{caption}\n\n"
        f"**Tags**\n{_bullet_list(tags)}"
    )


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


def _chain_summary_markdown(chain: ChainView, dataset: DemoDataset) -> str:
    seed_caption = _format_caption(_safe_row(chain.seed_row))
    return (
        f"## Chain `{chain.chain_id}`\n"
        f"**Loaded slice:** {dataset.summary['chain_offset']} to "
        f"{dataset.summary['chain_offset'] + dataset.summary['chains_loaded'] - 1}  \n"
        f"**Length:** {chain.chain_length} step(s) "
        f"(sampled target {chain.sampled_target_length})  \n"
        f"**Split:** `{chain.split or 'unknown'}`  \n"
        f"**Seed clip:** `{chain.seed_clip_id}`\n\n"
        f"**Seed caption**\n{seed_caption}"
    )


def _step_summary_markdown(chain: ChainView, step: StepView) -> str:
    record = step.instruction_record or {}
    status = str(record.get("status", "missing")).strip() or "missing"
    history_awareness = "present" if record else "not loaded"
    return (
        f"## Step {step.turn_index} of {chain.chain_length}\n"
        f"**Transition score:** `{step.transition_score:.4f}`  \n"
        f"**Hardness:** `{step.hardness or 'unknown'}`  \n"
        f"**Instruction record:** `{status}` ({history_awareness})  \n"
        f"**Primary edit:** {_step_primary_edit(step)}"
    )


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


def _app_summary_markdown(dataset: DemoDataset) -> str:
    run_root = str(dataset.paths.run_root) if dataset.paths.run_root else "custom paths"
    instructions_src = dataset.summary["instructions_source"] or "not provided"
    return (
        "# Chain Explorer\n"
        "This scaffold focuses on browsing sampled chains, their stepwise instructions, semantic deltas, and audio.\n\n"
        f"**Run root:** `{run_root}`  \n"
        f"**Chains loaded:** {dataset.summary['chains_loaded']}  \n"
        f"**Referenced clips:** {dataset.summary['referenced_clips']}  \n"
        f"**Manifest rows found:** {dataset.summary['manifest_rows_found']}  \n"
        f"**Instruction records found:** {dataset.summary['instructions_found']}  \n"
        f"**Instruction source:** `{instructions_src}`"
    )


def build_app(dataset: DemoDataset):
    try:
        import gradio as gr
    except Exception as exc:
        raise RuntimeError(
            "Gradio is required for the demo. Install it with `pip install -e .[demo]` "
            "or add `gradio` to the current environment."
        ) from exc

    cache_dir = Path(tempfile.gettempdir()) / "jamendo_instruct_chain_demo"
    chain_choices = list(dataset.chain_ids)
    first_chain = dataset.chains[0]

    def _chain_at(position: int) -> tuple[int, ChainView]:
        index = max(1, min(int(position), len(dataset.chains)))
        return index, dataset.chains[index - 1]

    def _render(position: int, requested_step: int) -> List[Any]:
        chain_pos, chain = _chain_at(position)
        step_count = max(1, len(chain.steps))
        step_index = max(1, min(int(requested_step), step_count))
        step = chain.steps[step_index - 1] if chain.steps else _empty_step_placeholder()
        seed_row = chain.seed_row
        source_row = dataset.manifest_by_clip.get(step.source_clip_id)
        target_row = dataset.manifest_by_clip.get(step.target_clip_id)
        seed_audio, seed_audio_note = _audio_preview(seed_row, cache_dir=cache_dir)
        source_audio, source_audio_note = _audio_preview(source_row, cache_dir=cache_dir)
        target_audio, target_audio_note = _audio_preview(target_row, cache_dir=cache_dir)
        record = step.instruction_record

        return [
            chain.chain_id,
            chain_pos,
            gr.update(value=step_index, minimum=1, maximum=step_count, step=1),
            _chain_summary_markdown(chain, dataset),
            _timeline_rows(chain),
            _step_summary_markdown(chain, step),
            _instruction_text(record, "history_unaware_instruction"),
            _instruction_text(record, "history_aware_instruction"),
            dict(record.get("semantic_delta_full", {}) or {}) if record else {},
            dict(record.get("semantic_delta_verbalized", {}) or {}) if record else {},
            dict(step.structured_delta or {}),
            dict(step.accumulated_intent_state or {}),
            _clip_markdown("Seed Clip", seed_row),
            seed_audio,
            seed_audio_note,
            _clip_markdown("Source Clip", source_row),
            source_audio,
            source_audio_note,
            _clip_markdown("Target Clip", target_row),
            target_audio,
            target_audio_note,
        ]

    def _on_chain_pick(chain_id: str) -> List[Any]:
        position = dataset.chain_ids.index(chain_id) + 1 if chain_id in dataset.chain_ids else 1
        return _render(position, 1)

    def _on_chain_position(position: int) -> List[Any]:
        return _render(position, 1)

    def _on_step_change(position: int, step_index: int) -> List[Any]:
        return _render(position, step_index)

    def _shift_chain(position: int, delta: int) -> List[Any]:
        return _render(max(1, min(len(dataset.chains), int(position) + delta)), 1)

    def _shift_step(position: int, step_index: int, delta: int) -> List[Any]:
        _, chain = _chain_at(position)
        return _render(position, max(1, min(max(1, len(chain.steps)), int(step_index) + delta)))

    css = """
    .instruction-box textarea {
      font-size: 1rem;
      line-height: 1.45;
    }
    .clip-card {
      min-height: 18rem;
    }
    """

    with gr.Blocks(css=css, title="Jamendo-Instruct Chain Explorer") as demo:
        gr.Markdown(_app_summary_markdown(dataset))

        with gr.Row():
            prev_chain_btn = gr.Button("Previous Chain")
            next_chain_btn = gr.Button("Next Chain")
            chain_selector = gr.Dropdown(
                choices=chain_choices,
                value=first_chain.chain_id,
                label="Chain",
                interactive=True,
            )
        chain_position = gr.Slider(
            minimum=1,
            maximum=max(1, len(dataset.chains)),
            value=1,
            step=1,
            label="Chain Position",
        )

        chain_summary = gr.Markdown()
        timeline = gr.Dataframe(
            headers=["turn", "source_clip_id", "target_clip_id", "hardness", "score", "primary_edit"],
            datatype=["number", "str", "str", "str", "number", "str"],
            interactive=False,
            label="Chain Timeline",
        )

        with gr.Row():
            prev_step_btn = gr.Button("Previous Step")
            next_step_btn = gr.Button("Next Step")
            step_slider = gr.Slider(minimum=1, maximum=max(1, first_chain.chain_length), value=1, step=1, label="Step")

        step_summary = gr.Markdown()

        with gr.Row():
            history_unaware = gr.Textbox(
                label="History-Unaware Instruction",
                lines=4,
                interactive=False,
                elem_classes=["instruction-box"],
            )
            history_aware = gr.Textbox(
                label="History-Aware Instruction",
                lines=4,
                interactive=False,
                elem_classes=["instruction-box"],
            )

        with gr.Row():
            semantic_full = gr.JSON(label="Semantic Delta Full")
            semantic_verbalized = gr.JSON(label="Semantic Delta Verbalized")
        with gr.Row():
            structured_delta = gr.JSON(label="Structured Delta")
            accumulated_state = gr.JSON(label="Accumulated Intent State")

        with gr.Row():
            with gr.Column():
                seed_clip = gr.Markdown(elem_classes=["clip-card"])
                seed_audio = gr.Audio(label="Seed Audio", interactive=False)
                seed_audio_note = gr.Markdown()
            with gr.Column():
                source_clip = gr.Markdown(elem_classes=["clip-card"])
                source_audio = gr.Audio(label="Source Audio", interactive=False)
                source_audio_note = gr.Markdown()
            with gr.Column():
                target_clip = gr.Markdown(elem_classes=["clip-card"])
                target_audio = gr.Audio(label="Target Audio", interactive=False)
                target_audio_note = gr.Markdown()

        render_outputs = [
            chain_selector,
            chain_position,
            step_slider,
            chain_summary,
            timeline,
            step_summary,
            history_unaware,
            history_aware,
            semantic_full,
            semantic_verbalized,
            structured_delta,
            accumulated_state,
            seed_clip,
            seed_audio,
            seed_audio_note,
            source_clip,
            source_audio,
            source_audio_note,
            target_clip,
            target_audio,
            target_audio_note,
        ]

        chain_selector.change(_on_chain_pick, inputs=[chain_selector], outputs=render_outputs, queue=False)
        chain_position.change(_on_chain_position, inputs=[chain_position], outputs=render_outputs, queue=False)
        step_slider.change(_on_step_change, inputs=[chain_position, step_slider], outputs=render_outputs, queue=False)
        prev_chain_btn.click(lambda pos: _shift_chain(pos, -1), inputs=[chain_position], outputs=render_outputs, queue=False)
        next_chain_btn.click(lambda pos: _shift_chain(pos, 1), inputs=[chain_position], outputs=render_outputs, queue=False)
        prev_step_btn.click(
            lambda pos, step: _shift_step(pos, step, -1),
            inputs=[chain_position, step_slider],
            outputs=render_outputs,
            queue=False,
        )
        next_step_btn.click(
            lambda pos, step: _shift_step(pos, step, 1),
            inputs=[chain_position, step_slider],
            outputs=render_outputs,
            queue=False,
        )

        demo.load(lambda: _render(1, 1), outputs=render_outputs, queue=False)

    return demo


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch a Gradio demo for browsing sampled Jamendo chains.")
    parser.add_argument("--run-root", help="Run artifact root, e.g. /path/to/<run_name>.")
    parser.add_argument("--manifest-csv", help="Explicit path to structured_clip_manifest.csv.")
    parser.add_argument("--chains-jsonl", help="Explicit path to sampled_chains.jsonl.")
    parser.add_argument("--instructions-jsonl", help="Optional path to chain_step_instructions.jsonl.")
    parser.add_argument("--chain-offset", type=int, default=0, help="How many chains to skip before loading.")
    parser.add_argument(
        "--max-chains",
        type=int,
        default=250,
        help="Maximum number of chains to load into the demo at startup.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host for the Gradio app.")
    parser.add_argument("--port", type=int, default=7860, help="Server port for the Gradio app.")
    parser.add_argument("--share", action="store_true", help="Enable Gradio sharing.")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    paths = _resolve_paths(args)
    max_chains = None if args.max_chains is not None and args.max_chains <= 0 else args.max_chains
    dataset = _load_dataset(paths, chain_offset=max(0, int(args.chain_offset)), max_chains=max_chains)
    demo = build_app(dataset)
    demo.launch(server_name=args.host, server_port=int(args.port), share=bool(args.share))


if __name__ == "__main__":
    main()
