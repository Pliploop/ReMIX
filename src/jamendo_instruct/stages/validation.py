from __future__ import annotations

import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Tuple

from jamendo_instruct.llm_backends import (
    OPENAI_COMPAT_BACKENDS,
    build_openai_chat_client,
    build_vllm_offline_chat_model,
    decode_openai_chat_completion,
    decode_vllm_chat_completion,
    load_chat_processor_and_model,
    resolve_backend_name,
)
from jamendo_instruct.progress import StageTracker, rich_tqdm
from jamendo_instruct.semantic_delta import build_typed_semantic_delta, typed_item_texts

if TYPE_CHECKING:
    from omegaconf import DictConfig
else:
    DictConfig = Any

CONF_DIR = str(Path(__file__).resolve().parents[3] / "conf")

_RELATIVE_CUES = [
    "keep",
    "still",
    "but",
    "more",
    "less",
    "without",
    "instead",
    "same",
    "bring back",
    "again",
    "like",
]

_FAIL_FORMAT = "failed:format_error"
_FAIL_NO_GENUINE_CHANGE = "failed:no_genuine_change"
_FAIL_CONTRADICTION = "failed:contradiction"
_FAIL_METADATA_INVENTION = "failed:metadata_invention"
_FAIL_REQUIRES_HISTORY = "failed:requires_history"
_FAIL_HISTORY_INCOHERENT = "failed:history_incoherent"
_FAIL_CAPTION_ONLY = "failed:caption_only_verbalization_missing"


def _log(cfg: DictConfig, message: str) -> None:
    if bool(cfg.stage.progress.enabled):
        print(f"[validation] {message}", flush=True)


def _cfg_section_to_plain(obj: Any) -> Any:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(obj):
            return OmegaConf.to_container(obj, resolve=True)
    except Exception:
        pass
    if hasattr(obj, "items"):
        return {str(k): _cfg_section_to_plain(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {str(k): _cfg_section_to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_cfg_section_to_plain(v) for v in obj]
    return obj


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _key(record: Dict[str, Any]) -> Tuple[str, int]:
    return str(record.get("chain_id", "") or ""), int(record.get("turn_index", 0) or 0)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", str(text or "").lower())


def _normalize_constraint_term(value: str) -> str:
    text = str(value or "").strip().lower()
    if ":" in text:
        return text.split(":", 1)[1].strip()
    return text


def _collect_caption_terms(payload: Dict[str, Any]) -> List[str]:
    terms: List[str] = []
    for scope in ("from_seed", "from_previous"):
        fuzzy = dict(payload.get("caption_differences_fuzzy", {}).get(scope, {}) or {})
        for key in ("added_terms", "removed_terms", "shared_terms", "added_phrases", "removed_phrases", "shared_phrases"):
            values = fuzzy.get(key, [])
            if isinstance(values, list):
                for value in values:
                    value = str(value).strip().lower()
                    if value:
                        terms.append(value)
        raw = dict(payload.get("caption_differences_raw", {}).get(scope, {}) or {})
        for key in ("source_caption", "target_caption"):
            for token in _tokenize(str(raw.get(key, "") or "")):
                if len(token) >= 4:
                    terms.append(token)
    out: List[str] = []
    for term in terms:
        if term not in out:
            out.append(term)
    return out


def _collect_lyric_terms(payload: Dict[str, Any]) -> List[str]:
    terms: List[str] = []
    for scope in ("from_seed", "from_previous"):
        fuzzy = dict(payload.get("lyric_differences_fuzzy", {}).get(scope, {}) or {})
        for key in ("added_terms", "removed_terms", "shared_terms", "added_phrases", "removed_phrases", "shared_phrases"):
            values = fuzzy.get(key, [])
            if isinstance(values, list):
                for value in values:
                    value = str(value).strip().lower()
                    if value:
                        terms.append(value)
        raw = dict(payload.get("lyric_differences_raw", {}).get(scope, {}) or {})
        for key in ("source_lyrics", "target_lyrics"):
            for token in _tokenize(str(raw.get(key, "") or "")):
                if len(token) >= 4:
                    terms.append(token)
    out: List[str] = []
    for term in terms:
        if term not in out:
            out.append(term)
    return out


def _collect_constraint_terms(payload: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    for bucket_name in ("persistent_constraints", "new_constraints", "removed_constraints"):
        bucket = dict(payload.get(bucket_name, {}) or {})
        for scope in ("from_seed", "from_previous"):
            raw_values = bucket.get(scope, [])
            if isinstance(raw_values, list):
                for value in raw_values:
                    term = _normalize_constraint_term(str(value))
                    if term:
                        values.append(term)
    out: List[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _dedupe_list(values: List[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _fallback_semantic_delta(payload: Dict[str, Any]) -> Dict[str, Any]:
    delta = dict(payload.get("delta_from_previous", {}) or {})
    new_items = [str(x) for x in payload.get("new_constraints", {}).get("from_previous", []) or [] if str(x).strip()]
    lost_items = [str(x) for x in payload.get("removed_constraints", {}).get("from_previous", []) or [] if str(x).strip()]
    preserved_items = [str(x) for x in payload.get("persistent_constraints", {}).get("from_previous", []) or [] if str(x).strip()]
    caption_only_change = (
        not delta.get("tags_added")
        and not delta.get("tags_removed")
        and str(delta.get("source_vocals", "") or "").strip() == str(delta.get("target_vocals", "") or "").strip()
        and str(delta.get("source_speed", "") or "").strip() == str(delta.get("target_speed", "") or "").strip()
        and (
            str(delta.get("source_caption", "") or "").strip() != str(delta.get("target_caption", "") or "").strip()
            or str(delta.get("source_lyrics", "") or "").strip() != str(delta.get("target_lyrics", "") or "").strip()
        )
    )
    if new_items:
        primary_edit = f"introduces {new_items[0]}"
    elif lost_items:
        primary_edit = f"moves away from {lost_items[0]}"
    else:
        primary_edit = "refines the current sound"
    return {
        "preserved": _dedupe_list(preserved_items),
        "new": _dedupe_list(new_items),
        "lost": _dedupe_list(lost_items),
        "primary_edit": primary_edit,
        "caption_only_change": bool(caption_only_change),
    }


def _semantic_delta(record: Dict[str, Any], payload: Dict[str, Any], *, field_name: str) -> Dict[str, Any]:
    value = record.get(field_name)
    if value is None and field_name == "semantic_delta_full":
        value = record.get("semantic_constraints")
    if isinstance(value, dict):
        preserved = _dedupe_list([str(x) for x in value.get("preserved", [])]) if isinstance(value.get("preserved", []), list) else []
        new = _dedupe_list([str(x) for x in value.get("new", [])]) if isinstance(value.get("new", []), list) else []
        lost = _dedupe_list([str(x) for x in value.get("lost", [])]) if isinstance(value.get("lost", []), list) else []
        primary_edit = str(value.get("primary_edit", "") or "").strip()
        caption_only_change = bool(value.get("caption_only_change", False))
        if primary_edit:
            return {
                "preserved": preserved,
                "new": new,
                "lost": lost,
                "primary_edit": primary_edit,
                "caption_only_change": caption_only_change,
            }
    return _fallback_semantic_delta(payload)


def _semantic_delta_pair(record: Dict[str, Any], payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    full = _semantic_delta(record, payload, field_name="semantic_delta_full")
    verbalized = _semantic_delta(record, payload, field_name="semantic_delta_verbalized")
    if not verbalized.get("primary_edit"):
        verbalized = full
    return full, verbalized


def _semantic_delta_typed(record: Dict[str, Any], payload: Dict[str, Any], *, field_name: str, delta: Dict[str, Any]) -> Dict[str, Any]:
    value = record.get(field_name)
    if isinstance(value, dict):
        return value
    return build_typed_semantic_delta(payload, delta)


def _semantic_delta_typed_pair(record: Dict[str, Any], payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    full, verbalized = _semantic_delta_pair(record, payload)
    full_typed = _semantic_delta_typed(record, payload, field_name="semantic_delta_full_typed", delta=full)
    verbalized_typed = _semantic_delta_typed(record, payload, field_name="semantic_delta_verbalized_typed", delta=verbalized)
    return full_typed, verbalized_typed


def _instruction_plan(record: Dict[str, Any]) -> Dict[str, Any]:
    plan = record.get("instruction_plan")
    return plan if isinstance(plan, dict) else {}


def _plan_evidence_terms(plan: Dict[str, Any], key: str) -> List[str]:
    values = plan.get(key, [])
    if not isinstance(values, list):
        return []
    out: List[str] = []
    for item in values:
        if not isinstance(item, dict):
            continue
        text = str(item.get("evidence", "") or "").strip()
        if text:
            out.append(text)
    return _dedupe_list(out)


def _contains_any(text: str, terms: List[str]) -> bool:
    lowered = str(text or "").lower()
    for term in terms:
        normalized = _normalize_constraint_term(term)
        if normalized and normalized in lowered:
            return True
    return False


def _payload_metadata_values(payload: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    for key in ("seed_view", "previous_view", "target_view"):
        view = dict(payload.get(key, {}) or {})
        metadata = dict(view.get("metadata", {}) or {})
        for meta_key in ("artist_name", "title", "release_date"):
            text = str(metadata.get(meta_key, "") or "").strip().lower()
            if text:
                values.append(text)
    return _dedupe_list(values)


def _has_relative_language(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(cue in lowered for cue in _RELATIVE_CUES)


def _count_term_hits(text: str, terms: List[str]) -> int:
    lowered = str(text or "").lower()
    hits = 0
    for term in terms:
        if term and term in lowered:
            hits += 1
    return hits


def _validate_variant(cfg: DictConfig, payload: Dict[str, Any], record: Dict[str, Any], instruction: str, *, variant: str) -> Dict[str, Any]:
    text = str(instruction or "").strip()
    reasons: List[str] = []
    min_chars = max(1, int(cfg.stage.checks.min_instruction_chars))
    caption_terms = _collect_caption_terms(payload)
    lyric_terms = _collect_lyric_terms(payload)
    semantic_full, semantic_verbalized = _semantic_delta_pair(record, payload)
    semantic_full_typed, semantic_verbalized_typed = _semantic_delta_typed_pair(record, payload)
    plan = _instruction_plan(record)
    plan_change_terms = _plan_evidence_terms(plan, "selected_changes")
    plan_preservation_terms = _plan_evidence_terms(plan, "selected_preservations")
    genuine_change_terms = plan_change_terms or _dedupe_list(list(semantic_verbalized.get("new", [])) + list(semantic_verbalized.get("lost", [])))
    explicit_preserved_terms = plan_preservation_terms or _dedupe_list(list(semantic_verbalized.get("preserved", [])))
    full_preserved_terms = _dedupe_list(list(semantic_full.get("preserved", [])))
    caption_semantic_terms = typed_item_texts(
        semantic_verbalized_typed if bool(semantic_full.get("caption_only_change", False)) else semantic_full_typed,
        source="caption",
    )
    lyric_semantic_terms = typed_item_texts(
        semantic_verbalized_typed if bool(semantic_full.get("caption_only_change", False)) else semantic_full_typed,
        source="lyrics",
    )
    caption_hits = _count_term_hits(text, caption_terms)
    lyric_hits = _count_term_hits(text, lyric_terms)
    caption_semantic_hits = _count_term_hits(text, [str(x).lower() for x in caption_semantic_terms])
    lyric_semantic_hits = _count_term_hits(text, [str(x).lower() for x in lyric_semantic_terms])
    constraint_hits = _count_term_hits(text, genuine_change_terms)

    if len(text) < min_chars:
        reasons.append(_FAIL_FORMAT)
    if bool(cfg.stage.checks.require_relative_language) and not _has_relative_language(text):
        reasons.append(_FAIL_FORMAT)
    if bool(getattr(cfg.stage.checks, "require_genuine_change", True)) and genuine_change_terms and constraint_hits <= 0:
        reasons.append(_FAIL_NO_GENUINE_CHANGE)
    if bool(getattr(cfg.stage.checks, "require_caption_only_grounding", True)) and bool(semantic_full.get("caption_only_change", False)):
        semantic_grounded = (caption_semantic_hits + lyric_semantic_hits) > 0
        raw_grounded = (caption_hits + lyric_hits) > 0
        if ((caption_semantic_terms or lyric_semantic_terms) and not semantic_grounded) or (
            not (caption_semantic_terms or lyric_semantic_terms) and (caption_terms or lyric_terms) and not raw_grounded
        ):
            reasons.append(_FAIL_CAPTION_ONLY)
    metadata_values = _payload_metadata_values(payload)
    if not metadata_values:
        if re.search(r"\b(?:artist|band|singer|album|\d{4})\b", text.lower()):
            reasons.append(_FAIL_METADATA_INVENTION)
    history_callback_present = False
    if variant == "history_aware":
        history_candidates = payload.get("history_reference_candidates", [])
        history_callback_present = any(
            cue in text.lower() for cue in ("earlier", "before", "back", "again", "turn ", "step ")
        )
        if bool(getattr(cfg.stage.checks, "require_history_callback", False)) and history_candidates and not history_callback_present:
            reasons.append(_FAIL_HISTORY_INCOHERENT)
    elif variant == "history_unaware" and bool(getattr(cfg.stage.checks, "require_seed_solvable", True)):
        if "as before" in text.lower() or "same as before" in text.lower():
            reasons.append(_FAIL_REQUIRES_HISTORY)

    return {
        "passed": len(_dedupe_list(reasons)) == 0,
        "reasons": _dedupe_list(reasons),
        "caption_hits": caption_hits,
        "constraint_hits": constraint_hits,
        "char_len": len(text),
        "history_callback_present": history_callback_present,
        "semantic_delta_full_used": semantic_full,
        "semantic_delta_verbalized_used": semantic_verbalized,
        "semantic_delta_full_typed_used": semantic_full_typed,
        "semantic_delta_verbalized_typed_used": semantic_verbalized_typed,
        "instruction_plan_used": plan or None,
        "caption_semantic_hits": caption_semantic_hits,
        "lyric_semantic_hits": lyric_semantic_hits,
        "preserved_term_count": len(explicit_preserved_terms),
        "explicit_preservation_term_count": len(explicit_preserved_terms),
        "full_preservation_term_count": len(full_preserved_terms),
    }


def _prompt_header() -> str:
    return (
        "You are a strict judge for compositional music-retrieval instructions.\n"
        "Return exactly one JSON object and nothing else.\n"
        "Do not use markdown fences.\n"
        "Do not add commentary outside the JSON."
    )


def _judge_prompt(payload: Dict[str, Any], record: Dict[str, Any], heuristic: Dict[str, Any]) -> List[Dict[str, str]]:
    user_content = (
        f"{_prompt_header()}\n\n"
        "Task:\n"
        "Evaluate whether both generated instructions satisfy the dataset requirements.\n"
        "Be strict but fair. Use the payload as the source of truth.\n\n"
        "Judging rules:\n"
        "1. Each instruction must be relative rather than fully self-contained.\n"
        "2. PASS if the instruction reflects at least one genuine change from `semantic_delta_verbalized.new` or `semantic_delta_verbalized.lost`.\n"
        f'   FAIL label: "{_FAIL_NO_GENUINE_CHANGE}"\n'
        "3. PASS if the instruction does not contradict explicit preservation constraints from `instruction_plan.selected_preservations` or `semantic_delta_verbalized.preserved`.\n"
        f'   FAIL label: "{_FAIL_CONTRADICTION}"\n'
        "4. Treat `semantic_delta_full.preserved` as diagnostic source-affinity context, not as hidden hard conservation requirements.\n"
        "5. PASS if the instruction does not invent unsupported metadata.\n"
        f'   FAIL label: "{_FAIL_METADATA_INVENTION}"\n'
        "6. For caption-only turns, PASS only if the instruction uses caption-derived and/or lyric-derived semantic content.\n"
        f'   FAIL label: "{_FAIL_CAPTION_ONLY}"\n'
        "7. `history_unaware` must be understandable from the seed plus the current request.\n"
        f'   FAIL label: "{_FAIL_REQUIRES_HISTORY}"\n'
        "8. `history_aware` may use broader chain history and should remain coherent if it references earlier turns.\n"
        f'   FAIL label: "{_FAIL_HISTORY_INCOHERENT}"\n'
        "9. Treat the heuristic precheck as advisory, not authoritative.\n\n"
        "Output format:\n"
        "{"
        '"accepted": true, '
        '"history_unaware": {"passed": true, "reasons": [], "checks": {"relative": true, "genuine_change": true, "no_invention": true, "constraint_consistency": true, "seed_solvable": true}}, '
        '"history_aware": {"passed": true, "reasons": [], "checks": {"relative": true, "genuine_change": true, "no_invention": true, "constraint_consistency": true, "history_coherence": true}}'
        "}\n\n"
        f"Payload:\n{json.dumps(payload, ensure_ascii=True, indent=2)}\n\n"
        "Generated instructions:\n"
        f"{json.dumps({'semantic_delta_full': record.get('semantic_delta_full') or record.get('semantic_constraints'), 'semantic_delta_verbalized': record.get('semantic_delta_verbalized'), 'instruction_plan': record.get('instruction_plan'), 'history_unaware_instruction': record.get('history_unaware_instruction', ''), 'history_aware_instruction': record.get('history_aware_instruction', '')}, ensure_ascii=True, indent=2)}\n\n"
        f"Heuristic precheck:\n{json.dumps(heuristic, ensure_ascii=True, indent=2)}"
    )
    return [
        {"role": "system", "content": "You are a rigorous validator that outputs strict JSON only."},
        {"role": "user", "content": user_content},
    ]


def _resolve_torch_device(cfg: DictConfig):
    import torch

    requested = str(cfg.stage.runtime.device)
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("stage.runtime.device requests CUDA, but torch.cuda.is_available() is false.")
    return SimpleNamespace(torch=torch, device=torch.device(requested))


def _resolve_torch_dtype(cfg: DictConfig, torch_module: Any) -> Any:
    value = str(cfg.stage.runtime.torch_dtype)
    if value == "auto":
        if torch_module.cuda.is_available():
            return torch_module.bfloat16
        return "auto"
    mapping = {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported stage.runtime.torch_dtype: {value}")
    return mapping[value]


def _build_judge(cfg: DictConfig) -> Any:
    model_id = str(cfg.stage.models.model_id)
    resolved = resolve_backend_name(
        configured_backend=str(getattr(cfg.stage.runtime, "backend", "transformers")),
        model_id=model_id,
        model_params_b=getattr(cfg.stage.models, "params_b", None),
        allow_sglang=bool(getattr(cfg.stage.runtime, "auto_allow_sglang", False)),
    )
    backend = str(resolved["backend"])
    if str(getattr(cfg.stage.runtime, "backend", "transformers")) == "auto":
        _log(
            cfg,
            "Auto-selected LLM backend "
            f"{backend} ({resolved.get('reason', 'unknown')}; GPUs={resolved.get('gpu_names', [])})",
        )
    if backend == "vllm_local":
        _log(cfg, f"Using OpenAI-compatible vLLM validation judge {model_id}")
        return build_openai_chat_client(
            model_id=model_id,
            host=str(getattr(cfg.stage.runtime, "vllm_host", "127.0.0.1")),
            port=int(getattr(cfg.stage.runtime, "vllm_port", 8000)),
            api_key=str(getattr(cfg.stage.runtime, "vllm_api_key", "EMPTY")),
            backend="vllm_local",
        )
    if backend == "sglang_local":
        _log(cfg, f"Using OpenAI-compatible SGLang validation judge {model_id}")
        return build_openai_chat_client(
            model_id=model_id,
            host=str(getattr(cfg.stage.runtime, "sglang_host", getattr(cfg.stage.runtime, "vllm_host", "127.0.0.1"))),
            port=int(getattr(cfg.stage.runtime, "sglang_port", getattr(cfg.stage.runtime, "vllm_port", 8000))),
            api_key=str(getattr(cfg.stage.runtime, "sglang_api_key", getattr(cfg.stage.runtime, "vllm_api_key", "EMPTY"))),
            backend="sglang_local",
        )
    if backend == "vllm":
        tensor_parallel_size = int(getattr(cfg.stage.runtime, "vllm_tensor_parallel_size", 0) or 0)
        if tensor_parallel_size <= 0:
            tensor_parallel_size = int(resolved.get("tensor_parallel_size", 1) or 1)
        quantization = getattr(cfg.stage.runtime, "vllm_quantization", None)
        if quantization is None:
            quantization = resolved.get("quantization")
        kv_cache_dtype = str(getattr(cfg.stage.runtime, "vllm_kv_cache_dtype", resolved.get("kv_cache_dtype", "auto")))
        _log(
            cfg,
            f"Loading offline vLLM validation model {model_id} "
            f"(tp={tensor_parallel_size}, quantization={quantization}, kv_cache_dtype={kv_cache_dtype})",
        )
        return build_vllm_offline_chat_model(
            model_id=model_id,
            tensor_parallel_size=tensor_parallel_size,
            dtype=str(getattr(cfg.stage.runtime, "vllm_dtype", "auto")),
            quantization=quantization,
            kv_cache_dtype=kv_cache_dtype,
            gpu_memory_utilization=float(getattr(cfg.stage.runtime, "vllm_gpu_memory_utilization", 0.9)),
            max_model_len=int(getattr(cfg.stage.runtime, "vllm_max_model_len", 0) or 0),
            trust_remote_code=bool(getattr(cfg.stage.runtime, "vllm_trust_remote_code", False)),
            enforce_eager=bool(getattr(cfg.stage.runtime, "vllm_enforce_eager", False)),
        )
    if backend not in {"transformers", "transformers_bnb"}:
        raise ValueError(f"Unsupported stage.runtime.backend: {backend}")

    runtime = _resolve_torch_device(cfg)
    torch = runtime.torch
    device = runtime.device
    token_env = str(getattr(cfg.stage.auth, "hf_token_env", "HF_TOKEN"))
    token = os.environ.get(token_env, "").strip() or None
    if token is None:
        _log(cfg, f"No Hugging Face token found in ${token_env}; gated model downloads may fail.")
    dtype = _resolve_torch_dtype(cfg, torch)
    quantization = "nf4" if backend == "transformers_bnb" else None
    _log(cfg, f"Loading validation model {model_id} on {device} with backend={backend}")
    processor, model, model_family = load_chat_processor_and_model(
        model_id=model_id,
        token=token,
        torch_dtype=dtype,
        device=device,
        model_family=str(getattr(cfg.stage.runtime, "llm_model_family", "auto")),
        quantization=quantization,
    )
    return SimpleNamespace(model=model, processor=processor, torch=torch, device=device, backend=backend, model_family=model_family)


def _decode_response_text(ctx: Any, messages: List[Dict[str, str]], cfg: DictConfig) -> str:
    backend = str(getattr(ctx, "backend", "transformers"))
    if backend in OPENAI_COMPAT_BACKENDS:
        return decode_openai_chat_completion(
            ctx,
            messages=messages,
            max_tokens=int(cfg.stage.judge.max_new_tokens),
            temperature=float(cfg.stage.judge.temperature),
            top_p=float(cfg.stage.judge.top_p),
        )
    if backend == "vllm":
        return decode_vllm_chat_completion(
            ctx,
            messages=messages,
            max_tokens=int(cfg.stage.judge.max_new_tokens),
            temperature=float(cfg.stage.judge.temperature),
            top_p=float(cfg.stage.judge.top_p),
            enable_thinking=bool(getattr(cfg.stage.runtime, "enable_thinking", False)),
        )

    processor = ctx.processor
    model = ctx.model
    torch = ctx.torch
    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=bool(getattr(cfg.stage.runtime, "enable_thinking", False)),
    )
    inputs = processor(text=chat_text, return_tensors="pt")
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]
    gen_kwargs = {
        "max_new_tokens": int(cfg.stage.judge.max_new_tokens),
        "do_sample": float(cfg.stage.judge.temperature) > 0.0,
        "temperature": float(cfg.stage.judge.temperature),
        "top_p": float(cfg.stage.judge.top_p),
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    return str(processor.decode(outputs[0][input_len:], skip_special_tokens=True) or "").strip()


def _strip_code_fences(text: str) -> str:
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


def _run_llm_judge(ctx: Any, cfg: DictConfig, payload: Dict[str, Any], record: Dict[str, Any], heuristic: Dict[str, Any]) -> Dict[str, Any]:
    messages = _judge_prompt(payload, record, heuristic)
    retries = max(1, int(cfg.stage.behavior.strict_json_retry_attempts))
    last_error = ""
    for attempt in range(retries):
        raw = _decode_response_text(ctx, messages, cfg)
        try:
            return _extract_json_object(raw)
        except Exception as exc:
            last_error = f"attempt={attempt + 1}: {exc}"
    raise ValueError(f"Judge output parse failure: {last_error or 'unknown_error'}")


def _judge_variant_section(section: Any) -> Dict[str, Any]:
    if not isinstance(section, dict):
        return {"passed": False, "reasons": ["missing_variant_section"], "checks": {}}
    reasons_raw = section.get("reasons", [])
    if isinstance(reasons_raw, list):
        reasons = [str(x) for x in reasons_raw if str(x).strip()]
    else:
        reasons = [str(reasons_raw)] if str(reasons_raw).strip() else []
    checks_raw = section.get("checks", {})
    checks = {}
    if isinstance(checks_raw, dict):
        checks = {str(k): bool(v) for k, v in checks_raw.items()}
    return {
        "passed": bool(section.get("passed", False)),
        "reasons": reasons,
        "checks": checks,
    }


def run_validation(cfg: DictConfig) -> Dict[str, object]:
    instructions_path = Path(str(cfg.stage.io.input_instructions_jsonl))
    prepared_path = Path(str(cfg.stage.io.input_prepared_jsonl))
    out_dir = Path(str(cfg.stage.io.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / str(cfg.stage.io.output_validated_jsonl)
    report_path = out_dir / str(cfg.stage.io.report_file)
    tracker = StageTracker(
        cfg,
        "validation",
        title="Validate Generated Instructions",
        subtitle=f"instructions={instructions_path}",
        total_steps=4,
    )

    for path in (instructions_path, prepared_path):
        if not path.exists():
            raise FileNotFoundError(f"Required validation input not found: {path}")

    tracker.step("Load prepared prompts and instructions", detail=f"prepared={prepared_path.name}")
    prepared_by_key = {_key(record): record for record in _read_jsonl(prepared_path)}
    max_steps = cfg.stage.behavior.max_steps
    discard_failed = bool(cfg.stage.behavior.discard_failed)
    require_both = bool(cfg.stage.behavior.require_both_variants)
    use_llm_judge = bool(cfg.stage.behavior.use_llm_judge)
    skip_llm_on_precheck_failure = bool(cfg.stage.behavior.skip_llm_on_precheck_failure)
    every_n = max(1, int(cfg.stage.progress.every_n_rows))
    tracker.step(
        "Resolve validation judge",
        detail=str(cfg.stage.models.model_id) if use_llm_judge else "heuristic-only mode",
    )
    judge_ctx = _build_judge(cfg) if use_llm_judge else None

    counts = {
        "steps_seen": 0,
        "steps_written": 0,
        "steps_failed": 0,
        "history_unaware_passed": 0,
        "history_aware_passed": 0,
        "missing_prepared_records": 0,
        "heuristic_failures": 0,
        "llm_judge_failures": 0,
        "judge_parse_failures": 0,
        "failure_reasons": {},
    }

    tracker.step("Validate instruction variants", detail=f"max_steps={max_steps if max_steps is not None else 'all'}")
    with out_jsonl.open("w", encoding="utf-8") as out_f:
        records = list(_read_jsonl(instructions_path))
        if max_steps is not None:
            records = records[: max(0, int(max_steps))]
        with rich_tqdm(cfg, total=len(records), desc="Validate steps", unit="step") as progress:
            for record in records:
                counts["steps_seen"] += 1
                prepared = prepared_by_key.get(_key(record))
                if prepared is None:
                    counts["missing_prepared_records"] += 1
                    counts["steps_failed"] += 1
                    counts["failure_reasons"]["missing_prepared_record"] = counts["failure_reasons"].get("missing_prepared_record", 0) + 1
                    progress.update(1)
                    continue

                unaware = _validate_variant(
                    cfg,
                    prepared,
                    record,
                    str(record.get("history_unaware_instruction", "") or ""),
                    variant="history_unaware",
                )
                aware = _validate_variant(
                    cfg,
                    prepared,
                    record,
                    str(record.get("history_aware_instruction", "") or ""),
                    variant="history_aware",
                )
                heuristic = {
                    "history_unaware": unaware,
                    "history_aware": aware,
                }
                heuristic_accepted = unaware["passed"] and aware["passed"] if require_both else (unaware["passed"] or aware["passed"])
                if not heuristic_accepted:
                    counts["heuristic_failures"] += 1

                final_unaware = unaware
                final_aware = aware
                judge_result = None
                if judge_ctx is not None and not (skip_llm_on_precheck_failure and not heuristic_accepted):
                    try:
                        judge_result = _run_llm_judge(judge_ctx, cfg, prepared, record, heuristic)
                        final_unaware = _judge_variant_section(judge_result.get("history_unaware"))
                        final_aware = _judge_variant_section(judge_result.get("history_aware"))
                    except Exception as exc:
                        counts["judge_parse_failures"] += 1
                        counts["failure_reasons"]["judge_parse_failure"] = counts["failure_reasons"].get("judge_parse_failure", 0) + 1
                        final_unaware = {"passed": False, "reasons": [f"judge_error:{exc.__class__.__name__}"]}
                        final_aware = {"passed": False, "reasons": [f"judge_error:{exc.__class__.__name__}"]}

                if final_unaware["passed"]:
                    counts["history_unaware_passed"] += 1
                if final_aware["passed"]:
                    counts["history_aware_passed"] += 1

                accepted = final_unaware["passed"] and final_aware["passed"] if require_both else (final_unaware["passed"] or final_aware["passed"])
                if not accepted:
                    counts["steps_failed"] += 1
                    if judge_ctx is not None:
                        counts["llm_judge_failures"] += 1
                    for reason in list(final_unaware["reasons"]) + list(final_aware["reasons"]):
                        counts["failure_reasons"][reason] = counts["failure_reasons"].get(reason, 0) + 1
                    if discard_failed:
                        progress.update(1)
                        continue

                output = dict(record)
                output["validation"] = {
                    "accepted": accepted,
                    "history_unaware": final_unaware,
                    "history_aware": final_aware,
                    "heuristic_precheck": heuristic,
                    "judge_result": judge_result,
                }
                out_f.write(json.dumps(output, ensure_ascii=True) + "\n")
                counts["steps_written"] += 1
                if counts["steps_seen"] % every_n == 0:
                    _log(cfg, f"Validation steps processed: {counts['steps_seen']:,}")
                progress.update(1)

    report = {
        "stage": "validation",
        "input": {
            "input_instructions_jsonl": str(instructions_path),
            "input_prepared_jsonl": str(prepared_path),
        },
        "counts": counts,
        "config": {
            "behavior": _cfg_section_to_plain(cfg.stage.behavior),
            "checks": _cfg_section_to_plain(cfg.stage.checks),
            "judge": _cfg_section_to_plain(cfg.stage.judge),
        },
        "outputs": {
            "output_validated_jsonl": str(out_jsonl),
            "report": str(report_path),
        },
    }

    tracker.step("Write report", detail=report_path.name)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)

    tracker.finish(
        f"wrote {counts['steps_written']:,}/{counts['steps_seen']:,} records"
    )
    _log(cfg, f"Validation complete. Wrote {counts['steps_written']:,} / {counts['steps_seen']:,} records")
    return report


def _main_impl(cfg: DictConfig) -> None:
    report = run_validation(cfg)
    print(json.dumps({"status": "ok", "stage": "validation", "outputs": report["outputs"]}, indent=2))


def main() -> None:
    import hydra

    @hydra.main(version_base=None, config_path=CONF_DIR, config_name="config")
    def _wrapped(cfg: DictConfig) -> None:
        _main_impl(cfg)

    _wrapped()


if __name__ == "__main__":
    main()
