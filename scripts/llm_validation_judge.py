#!/usr/bin/env python
"""Run a text-only LLM over the *same* rubric and inputs the human raters see.

The human validation app (``jamendo_instruct.demo.human_validation_app``) asks
each rater the eight questions in ``RATING_QUESTIONS`` about a source->target
instruction, showing them the source and target captions, tags, and metadata
(plus audio). This script feeds the identical questions and the identical
*text* evidence to an instruction-tuned LLM and writes ``llm_ratings.jsonl``
next to ``human_ratings.jsonl``, using the same record schema so the Admin tab
and any human-vs-LLM agreement analysis can consume both.

Only difference vs. the human: a text-only judge does not hear the audio. Each
record therefore carries ``modality: "text_only"`` / ``audio_available: false``.

Backends mirror ``jamendo_instruct.stages.validation`` (transformers, offline
vLLM, or an OpenAI-compatible vLLM/SGLang server), so Gemma, Qwen, and other
chat LLMs work by swapping ``--model-id``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

# Allow running straight from a checkout without an editable install.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from jamendo_instruct.demo.chains_demo import (  # noqa: E402
    ChainView,
    DemoDataset,
    StepView,
    _format_caption,
    _format_tags,
    _load_dataset_for_streamlit,
)
from jamendo_instruct.demo.validation_rubric import (  # noqa: E402
    CANNOT_JUDGE_LABEL,
    ISSUE_TAGS,
    NOT_APPLICABLE_LABEL,
    RATING_QUESTIONS,
    _answered_rating_keys,
    _available_pairs,
    _available_samples,
    _clip_label,
    _dataset_from_frozen_sidecar,
    _filter_by_assignments,
    _metadata_view,
    _question_scale,
    _rating_help,
    _rating_item_key,
    _rating_value,
    _read_assignments,
    _read_jsonl_records,
    _record_variant_index,
    _sample_identity,
    _validation_output_dir,
)
from jamendo_instruct.llm_backends import (  # noqa: E402
    OPENAI_COMPAT_BACKENDS,
    build_openai_chat_client,
    build_vllm_offline_chat_model,
    decode_openai_chat_completion,
    decode_vllm_chat_completions,
    load_chat_processor_and_model,
    resolve_backend_name,
)


# --------------------------------------------------------------------------- #
# Prompt construction (same questions + text evidence as the human app)
# --------------------------------------------------------------------------- #
def _clip_evidence(dataset: DemoDataset, clip_id: str) -> Dict[str, Any]:
    row = dataset.manifest_by_clip.get(clip_id)
    return {
        "label": _clip_label(row, clip_id),
        "caption": _format_caption(row or {}),
        "tags": _format_tags(row or {}),
        "metadata": _metadata_view(row),
    }


def _question_options(question: Dict[str, Any]) -> List[str]:
    options = [label for label, _score in _question_scale(question)] + [CANNOT_JUDGE_LABEL]
    if bool(question.get("allow_na", False)):
        options.append(NOT_APPLICABLE_LABEL)
    return options


_SYSTEM_PROMPT = (
    "You are a careful music-annotation judge. You rate whether a text instruction "
    "correctly turns a source music track into a target music track, using only the "
    "provided evidence. You cannot listen to the audio, so judge from the captions, "
    "tags, and metadata. When the evidence is not enough to decide a question, answer "
    f'"{CANNOT_JUDGE_LABEL}" rather than guessing. Return exactly one JSON object and '
    "nothing else. Do not use markdown fences or add commentary outside the JSON."
)


def build_rubric_prompt(instruction: str, source: Dict[str, Any], target: Dict[str, Any]) -> List[Dict[str, str]]:
    question_lines: List[str] = []
    for question in RATING_QUESTIONS:
        qid = str(question["id"])
        options = _question_options(question)
        help_text = _rating_help(qid)
        question_lines.append(
            f'- "{qid}": {question["statement"]}\n'
            f"    Guidance: {help_text}\n"
            f"    Allowed answers (choose exactly one): {json.dumps(options, ensure_ascii=True)}"
        )
    questions_block = "\n".join(question_lines)

    user_content = (
        "Task:\n"
        "An instruction was written to transform a SOURCE music track into a TARGET music track. "
        "Judge the instruction against the evidence below by answering every question. The target is "
        "allowed to differ from the source in ways the instruction does not mention; only judge what "
        "each question asks. Absence of evidence is not contradiction.\n\n"
        f"SOURCE track:\n{json.dumps(source, ensure_ascii=True, indent=2)}\n\n"
        f"TARGET track:\n{json.dumps(target, ensure_ascii=True, indent=2)}\n\n"
        f"INSTRUCTION (source -> target):\n{json.dumps(instruction, ensure_ascii=True)}\n\n"
        "Questions:\n"
        f"{questions_block}\n\n"
        "Also list any applicable diagnostic issue tags from this set (empty list if none apply):\n"
        f"{json.dumps(list(ISSUE_TAGS), ensure_ascii=True)}\n\n"
        "Output format (use the exact answer strings from each question's allowed answers):\n"
        '{"answers": {'
        + ", ".join(f'"{q["id"]}": "<one allowed answer>"' for q in RATING_QUESTIONS)
        + '}, "issue_tags": [], "notes": "<short justification>"}'
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #
def _strip_code_fences(text: str) -> str:
    import re

    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


def _extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = _strip_code_fences(text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        data = json.loads(cleaned[start : end + 1])
        if isinstance(data, dict):
            return data
    raise json.JSONDecodeError("Unable to parse JSON object from model output", cleaned, 0)


def _coerce_label(raw: Any, options: Sequence[str]) -> str | None:
    """Map a model-returned string onto one of the allowed answer labels."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    for option in options:
        if option == text:
            return option
    lowered = text.lower()
    for option in options:
        if option.lower() == lowered:
            return option
    for option in options:  # tolerate trailing punctuation / extra words
        if lowered.startswith(option.lower()) or option.lower().startswith(lowered):
            return option
    return None


def _score_answers(parsed: Dict[str, Any]) -> tuple[Dict[str, Any], List[str], int]:
    raw_answers = dict(parsed.get("answers", {}) or {})
    scored: Dict[str, Any] = {}
    unparsed = 0
    for question in RATING_QUESTIONS:
        qid = str(question["id"])
        options = _question_options(question)
        label = _coerce_label(raw_answers.get(qid), options)
        if label is None:
            unparsed += 1
            scored[qid] = {
                "label": None,
                "raw_label": raw_answers.get(qid),
                "score": None,
                "cannot_judge": False,
                "not_applicable": False,
                "polarity": question["polarity"],
                "parsed": False,
            }
            continue
        score, cannot_judge, not_applicable = _rating_value(label, _question_scale(question))
        scored[qid] = {
            "label": label,
            "score": score,
            "cannot_judge": cannot_judge,
            "not_applicable": not_applicable,
            "polarity": question["polarity"],
            "parsed": True,
        }
    valid_tags = {str(tag) for tag in ISSUE_TAGS}
    issue_tags = [str(tag) for tag in (parsed.get("issue_tags", []) or []) if str(tag) in valid_tags]
    return scored, issue_tags, unparsed


# --------------------------------------------------------------------------- #
# Judge backend (text-only; mirrors stages.validation._build_judge)
# --------------------------------------------------------------------------- #
def build_judge(args: argparse.Namespace) -> SimpleNamespace:
    model_id = str(args.model_id)
    resolved = resolve_backend_name(
        configured_backend=str(args.backend),
        model_id=model_id,
        model_params_b=args.params_b,
        allow_sglang=bool(args.auto_allow_sglang),
    )
    backend = str(resolved["backend"])
    if str(args.backend) == "auto":
        print(
            f"[llm-judge] auto-selected backend={backend} "
            f"({resolved.get('reason', 'unknown')}; GPUs={resolved.get('gpu_names', [])})",
            flush=True,
        )

    if backend in ("vllm_local", "sglang_local"):
        print(f"[llm-judge] using OpenAI-compatible {backend} server for {model_id}", flush=True)
        ctx = build_openai_chat_client(
            model_id=model_id,
            host=str(args.server_host),
            port=int(args.server_port),
            api_key=str(args.server_api_key),
            backend=backend,
        )
        return SimpleNamespace(kind=backend, backend=backend, ctx=ctx, model_id=model_id)

    if backend == "vllm":
        tensor_parallel_size = int(args.tensor_parallel_size or 0) or int(resolved.get("tensor_parallel_size", 1) or 1)
        quantization = args.quantization if args.quantization is not None else resolved.get("quantization")
        kv_cache_dtype = str(args.kv_cache_dtype or resolved.get("kv_cache_dtype", "auto"))
        print(
            f"[llm-judge] loading offline vLLM {model_id} "
            f"(tp={tensor_parallel_size}, quantization={quantization}, kv_cache_dtype={kv_cache_dtype})",
            flush=True,
        )
        ctx = build_vllm_offline_chat_model(
            model_id=model_id,
            tensor_parallel_size=tensor_parallel_size,
            dtype=str(args.dtype),
            quantization=quantization,
            kv_cache_dtype=kv_cache_dtype,
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            max_model_len=int(args.max_model_len or 0),
            trust_remote_code=bool(args.trust_remote_code),
            enforce_eager=bool(args.enforce_eager),
        )
        return SimpleNamespace(kind="vllm", backend="vllm", ctx=ctx, model_id=model_id)

    if backend not in ("transformers", "transformers_bnb"):
        raise ValueError(f"Unsupported backend: {backend}")

    import torch

    device_str = str(args.device)
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device requests CUDA, but torch.cuda.is_available() is false.")
    device = torch.device(device_str)

    dtype_value = str(args.torch_dtype)
    if dtype_value == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_available() else "auto"
    else:
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_value]

    token = os.environ.get(str(args.hf_token_env), "").strip() or None
    if token is None:
        print(f"[llm-judge] no HF token in ${args.hf_token_env}; gated downloads may fail.", flush=True)
    quantization = "nf4" if backend == "transformers_bnb" else None
    print(f"[llm-judge] loading {model_id} on {device} with backend={backend}", flush=True)
    processor, model, model_family = load_chat_processor_and_model(
        model_id=model_id,
        token=token,
        torch_dtype=dtype,
        device=device,
        model_family=str(args.llm_model_family),
        quantization=quantization,
    )
    return SimpleNamespace(
        kind="transformers",
        backend=backend,
        processor=processor,
        model=model,
        torch=torch,
        device=device,
        model_family=model_family,
        model_id=model_id,
    )


def decode_batch(judge: SimpleNamespace, args: argparse.Namespace, messages_batch: Sequence[List[Dict[str, str]]]) -> List[str]:
    if judge.kind == "vllm":
        return decode_vllm_chat_completions(
            judge.ctx,
            messages_batch=list(messages_batch),
            max_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            enable_thinking=bool(args.enable_thinking),
        )
    if judge.backend in OPENAI_COMPAT_BACKENDS:
        return [
            decode_openai_chat_completion(
                judge.ctx,
                messages=messages,
                max_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
            )
            for messages in messages_batch
        ]
    # transformers
    outputs: List[str] = []
    processor = judge.processor
    model = judge.model
    torch = judge.torch
    for messages in messages_batch:
        chat_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(args.enable_thinking),
        )
        inputs = processor(text=chat_text, return_tensors="pt")
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]
        gen_kwargs = {
            "max_new_tokens": int(args.max_new_tokens),
            "do_sample": float(args.temperature) > 0.0,
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
        }
        with torch.no_grad():
            generated = model.generate(**inputs, **gen_kwargs)
        outputs.append(str(processor.decode(generated[0][input_len:], skip_special_tokens=True) or "").strip())
    return outputs


# --------------------------------------------------------------------------- #
# Dataset loading (same item set as the human app)
# --------------------------------------------------------------------------- #
def load_samples(args: argparse.Namespace) -> tuple[DemoDataset, List[Dict[str, Any]]]:
    instruction_field = str(args.instruction_field)
    if args.frozen_sidecar_json:
        dataset, assignments = _dataset_from_frozen_sidecar(Path(args.frozen_sidecar_json).expanduser().resolve())
    else:
        run_root = str(Path(args.run_root).expanduser()) if args.run_root else None
        if args.instructions_jsonl:
            instructions_jsonl = str(Path(args.instructions_jsonl).expanduser())
        elif run_root:
            instructions_jsonl = str(Path(run_root) / args.instruction_folder / "chain_step_instructions.jsonl")
        else:
            instructions_jsonl = None
        max_chains = None if int(args.max_chains) <= 0 else int(args.max_chains)
        dataset = _load_dataset_for_streamlit(
            run_root,
            None if run_root else args.manifest_csv,
            None if run_root else args.chains_jsonl,
            instructions_jsonl,
            max(0, int(args.chain_offset)),
            max_chains,
        )
        assignment_path = Path(args.assignment_jsonl).expanduser().resolve() if args.assignment_jsonl else None
        assignments = _read_assignments(assignment_path)

    samples, _pairs = _filter_by_assignments(
        _available_samples(dataset, instruction_field),
        _available_pairs(dataset, instruction_field),
        assignments,
    )
    return dataset, samples


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    data = parser.add_argument_group("data inputs (mirror the human app)")
    data.add_argument("--run-root", help="Run artifact root, e.g. /path/to/<run_name>.")
    data.add_argument("--manifest-csv", help="Explicit path to structured_clip_manifest.csv (when no --run-root).")
    data.add_argument("--chains-jsonl", help="Explicit path to sampled_chains.jsonl (when no --run-root).")
    data.add_argument("--instructions-jsonl", help="Explicit path to chain_step_instructions.jsonl.")
    data.add_argument(
        "--instruction-folder",
        default=os.environ.get("INSTRUCTION_NAME", "instructions_axis_focused_5"),
        help="Folder under --run-root holding chain_step_instructions.jsonl.",
    )
    data.add_argument("--assignment-jsonl", help="Optional frozen human-validation assignment JSONL (same slice as raters).")
    data.add_argument("--frozen-sidecar-json", help="Optional compact validation sidecar generated with the assignment.")
    data.add_argument("--instruction-field", default="history_unaware_instruction")
    data.add_argument("--chain-offset", type=int, default=0)
    data.add_argument("--max-chains", type=int, default=0, help="0 loads all chains.")
    data.add_argument("--limit", type=int, default=0, help="Cap number of items judged (0 = all).")
    data.add_argument("--overwrite", action="store_true", help="Re-judge items already rated by this model.")

    model = parser.add_argument_group("model / backend")
    model.add_argument("--model-id", required=True, help="e.g. google/gemma-2-9b-it or Qwen/Qwen2.5-7B-Instruct.")
    model.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "vllm", "vllm_local", "sglang_local", "transformers", "transformers_bnb"],
    )
    model.add_argument("--params-b", default=None, help="Model size in billions (auto-inferred from the id if omitted).")
    model.add_argument("--device", default="auto", help="transformers backend: auto | cuda | cpu.")
    model.add_argument("--torch-dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    model.add_argument("--llm-model-family", default="auto", choices=["auto", "causal_lm", "qwen3_6"])
    model.add_argument("--hf-token-env", default="HF_TOKEN")
    model.add_argument("--auto-allow-sglang", action="store_true")
    model.add_argument("--tensor-parallel-size", type=int, default=0, help="offline vLLM (0 = auto from GPUs).")
    model.add_argument("--dtype", default="auto", help="offline vLLM dtype.")
    model.add_argument("--quantization", default=None, help="offline vLLM quantization (e.g. fp8).")
    model.add_argument("--kv-cache-dtype", default="auto")
    model.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    model.add_argument("--max-model-len", type=int, default=32768)
    model.add_argument("--trust-remote-code", action="store_true")
    model.add_argument("--enforce-eager", action="store_true")
    model.add_argument("--server-host", default="127.0.0.1", help="vllm_local/sglang_local host.")
    model.add_argument("--server-port", type=int, default=8000)
    model.add_argument("--server-api-key", default="EMPTY")

    gen = parser.add_argument_group("generation")
    gen.add_argument("--temperature", type=float, default=0.0)
    gen.add_argument("--top-p", type=float, default=1.0)
    gen.add_argument("--max-new-tokens", type=int, default=768)
    gen.add_argument("--enable-thinking", action="store_true")
    gen.add_argument("--batch-size", type=int, default=8, help="Items per generate() call (offline vLLM only).")
    gen.add_argument("--strict-json-retry-attempts", type=int, default=2)

    out = parser.add_argument_group("output")
    out.add_argument("--output-dir", help="Override output dir (default: dataset validation dir).")
    out.add_argument("--output-name", default="llm_ratings.jsonl")
    out.add_argument("--annotator-id", default=None, help="Override annotator id (default: llm:<model_id>).")
    out.add_argument(
        "--emit-validated",
        action="store_true",
        help="After judging, also write validated_instructions.jsonl (the instruction-validity gate "
        "consumed by relevance_pool) from the graded ratings.",
    )
    out.add_argument("--accept-threshold", type=float, default=4.0, help="--emit-validated: overall_validity >= accepts.")
    out.add_argument("--contextual-policy", default="truncate", choices=["truncate", "drop", "per_step"])
    out.add_argument("--chain-aggregate", default="min", choices=["min", "mean"])
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    instruction_field = str(args.instruction_field)
    annotator_id = str(args.annotator_id) if args.annotator_id else f"llm:{args.model_id}"

    dataset, samples = load_samples(args)
    if not samples:
        print("[llm-judge] no instruction variants found for the selected slice/field.", flush=True)
        return

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else _validation_output_dir(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / str(args.output_name)

    already_done: set = set()
    if not args.overwrite:
        already_done = _answered_rating_keys(_read_jsonl_records(output_path), annotator_id)

    pending: List[Dict[str, Any]] = []
    for sample in samples:
        item_key = _rating_item_key(
            {
                "chain_id": sample["chain"].chain_id,
                "turn_index": sample["step"].turn_index,
                "variant_index": _record_variant_index(sample["record"]),
                "instruction_field": instruction_field,
            }
        )
        if item_key in already_done:
            continue
        pending.append(sample)
    if int(args.limit) > 0:
        pending = pending[: int(args.limit)]

    print(
        f"[llm-judge] items: {len(samples)} total, {len(already_done)} already rated by {annotator_id}, "
        f"{len(pending)} to judge -> {output_path}",
        flush=True,
    )
    if not pending:
        return

    judge = build_judge(args)
    batch_size = max(1, int(args.batch_size)) if judge.kind == "vllm" else 1
    retries = max(1, int(args.strict_json_retry_attempts))

    written = 0
    parse_failures = 0
    with output_path.open("a", encoding="utf-8") as out_f:
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            messages_batch: List[List[Dict[str, str]]] = []
            evidences: List[tuple[Dict[str, Any], Dict[str, Any]]] = []
            for sample in batch:
                step: StepView = sample["step"]
                source = _clip_evidence(dataset, step.source_clip_id)
                target = _clip_evidence(dataset, step.target_clip_id)
                evidences.append((source, target))
                messages_batch.append(build_rubric_prompt(sample["instruction"], source, target))

            raws = decode_batch(judge, args, messages_batch)

            for position, (sample, raw, (source, target)) in enumerate(zip(batch, raws, evidences)):
                chain: ChainView = sample["chain"]
                step = sample["step"]
                record = sample["record"]

                parsed: Dict[str, Any] | None = None
                last_error = ""
                for attempt in range(retries):
                    try:
                        parsed = _extract_json_object(raw)
                        break
                    except Exception as exc:  # noqa: BLE001
                        last_error = f"attempt={attempt + 1}: {exc}"
                        if attempt + 1 < retries:
                            raw = decode_batch(judge, args, [messages_batch[position]])[0]

                if parsed is None:
                    parse_failures += 1
                    scored_answers: Dict[str, Any] = {
                        str(q["id"]): {
                            "label": None,
                            "score": None,
                            "cannot_judge": False,
                            "not_applicable": False,
                            "polarity": q["polarity"],
                            "parsed": False,
                        }
                        for q in RATING_QUESTIONS
                    }
                    issue_tags: List[str] = []
                    notes = ""
                    unparsed = len(RATING_QUESTIONS)
                    parse_ok = False
                else:
                    scored_answers, issue_tags, unparsed = _score_answers(parsed)
                    notes = str(parsed.get("notes", "") or "")
                    parse_ok = True
                    if unparsed:
                        parse_failures += 1

                source_row = dataset.manifest_by_clip.get(step.source_clip_id)
                target_row = dataset.manifest_by_clip.get(step.target_clip_id)
                out_f.write(
                    json.dumps(
                        {
                            "annotation_type": "llm_single_variant_rating",
                            "annotated_at_utc": datetime.now(timezone.utc).isoformat(),
                            "annotator_id": annotator_id,
                            "judge_model_id": str(args.model_id),
                            "judge_backend": str(judge.backend),
                            "modality": "text_only",
                            "audio_available": False,
                            "instruction_field": instruction_field,
                            **_sample_identity(chain, step, record),
                            "source_label": _clip_label(source_row, step.source_clip_id),
                            "target_label": _clip_label(target_row, step.target_clip_id),
                            "source_metadata": _metadata_view(source_row),
                            "target_metadata": _metadata_view(target_row),
                            "source_caption": source["caption"],
                            "target_caption": target["caption"],
                            "assignment": sample.get("assignment"),
                            "instruction": sample["instruction"],
                            "answers": scored_answers,
                            "issue_tags": issue_tags,
                            "notes": notes,
                            "parse_ok": parse_ok,
                            "unparsed_questions": unparsed,
                            "parse_error": last_error if not parse_ok else "",
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
                out_f.flush()
                written += 1

            print(f"[llm-judge] {written}/{len(pending)} judged", flush=True)

    print(
        f"[llm-judge] done. wrote {written} ratings to {output_path} "
        f"({parse_failures} items with parse issues).",
        flush=True,
    )

    if args.emit_validated:
        from jamendo_instruct.validation_gate import GateConfig, grade_records, select_chain_variants

        all_ratings = _read_jsonl_records(output_path)
        instruction_records = [sample["record"] for sample in samples if sample.get("record")]
        config = GateConfig(
            accept_threshold=float(args.accept_threshold),
            contextual_policy=str(args.contextual_policy),
            chain_aggregate=str(args.chain_aggregate),
            instruction_field=instruction_field,
        )
        graded, _ = grade_records(instruction_records, all_ratings, config)
        selected, report = select_chain_variants(graded, config)
        with (output_dir / "instruction_grades.jsonl").open("w", encoding="utf-8") as gf:
            for record in graded:
                gf.write(json.dumps(record, ensure_ascii=True) + "\n")
        with (output_dir / "validated_instructions.jsonl").open("w", encoding="utf-8") as vf:
            for record in selected:
                vf.write(json.dumps(record, ensure_ascii=True) + "\n")
        (output_dir / "validation_gate_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        c = report["counts"]
        print(
            f"[llm-judge] wrote grades + validated_instructions — accepted {c['accepted']}/{c['steps']} steps "
            f"(threshold={args.accept_threshold}, fallback_used={c['fallback_used']}).",
            flush=True,
        )


if __name__ == "__main__":
    main()
