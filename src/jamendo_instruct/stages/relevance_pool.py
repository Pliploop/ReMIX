from __future__ import annotations

import csv
import json
import math
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Sequence, Tuple

from jamendo_instruct.progress import StageTracker, rich_tqdm
from jamendo_instruct.semantic_delta import build_typed_semantic_delta, typed_item_texts

if TYPE_CHECKING:
    from omegaconf import DictConfig
else:
    DictConfig = Any

CONF_DIR = str(Path(__file__).resolve().parents[3] / "conf")


def _np():
    import numpy as np

    return np


def _log(cfg: DictConfig, message: str) -> None:
    if bool(cfg.stage.progress.enabled):
        print(f"[relevance_pool] {message}", flush=True)


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


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _key(record: Dict[str, Any]) -> Tuple[str, int]:
    return str(record.get("chain_id", "") or ""), int(record.get("turn_index", 0) or 0)


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
    return [str(x).strip() for x in data if str(x).strip()]


def _structured_index(path: Path) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for row in _read_csv_rows(path):
        clip_id = str(row.get("clip_id", "") or "").strip()
        if clip_id:
            out[clip_id] = row
    return out


def _node_index(path: Path) -> Dict[int, Dict[str, str]]:
    return {int(row["node_idx"]): row for row in _read_csv_rows(path)}


def _clip_to_node_index(nodes_by_idx: Dict[int, Dict[str, str]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for node_idx, row in nodes_by_idx.items():
        clip_id = str(row.get("clip_id", "") or "").strip()
        if clip_id:
            out[clip_id] = node_idx
    return out


def _lookup_manifest_index(path: Path) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for row in _read_csv_rows(path):
        clip_id = str(row.get("clip_id", "") or "").strip()
        if clip_id:
            out[clip_id] = row
    return out


def _normalize_np(matrix: Any) -> Any:
    np = _np()
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def _resolve_torch_device(cfg: DictConfig):
    import torch

    requested = str(getattr(cfg.stage.runtime, "device", "auto"))
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("stage.runtime.device requests CUDA, but torch.cuda.is_available() is false.")
    return SimpleNamespace(torch=torch, device=torch.device(requested))


def _build_text_embedder(cfg: DictConfig):
    from transformers import AutoModel, AutoTokenizer

    runtime = _resolve_torch_device(cfg)
    torch = runtime.torch
    device = runtime.device
    token_env = str(getattr(cfg.stage.auth, "hf_token_env", "HF_TOKEN"))
    token = os.environ.get(token_env, "").strip() or None
    model_id = str(cfg.stage.models.text_model_id)
    _log(cfg, f"Loading relevance-pool text model {model_id} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModel.from_pretrained(model_id, token=token)
    model = model.to(device)
    model.eval()
    return SimpleNamespace(model=model, tokenizer=tokenizer, torch=torch, device=device)


def _encode_texts(texts: List[str], cfg: DictConfig, ctx: Any) -> Any:
    np = _np()
    torch = ctx.torch
    device = ctx.device
    model = ctx.model
    tokenizer = ctx.tokenizer
    max_length = int(getattr(cfg.stage.text, "max_length", 512))
    with torch.no_grad():
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = model(**encoded)
        token_embeddings = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1)
        masked = token_embeddings * attention_mask
        summed = masked.sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1)
        batch_embs = summed / counts
        batch_embs = torch.nn.functional.normalize(batch_embs, dim=-1)
        result = batch_embs.detach().cpu().numpy().astype(np.float32)
    return result


def _build_llm_judge(cfg: DictConfig):
    from transformers import AutoModelForCausalLM, AutoProcessor

    runtime = _resolve_torch_device(cfg)
    torch = runtime.torch
    device = runtime.device
    token_env = str(getattr(cfg.stage.auth, "hf_token_env", "HF_TOKEN"))
    token = os.environ.get(token_env, "").strip() or None
    model_id = str(getattr(cfg.stage.models, "judge_model_id", cfg.stage.models.text_model_id))
    _log(cfg, f"Loading relevance-pool judge model {model_id} on {device}")
    processor = AutoProcessor.from_pretrained(model_id, token=token)
    model_kwargs: Dict[str, Any] = {"token": token}
    if str(device).startswith("cuda"):
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if not str(device).startswith("cuda"):
        model = model.to(device)
    model.eval()
    return SimpleNamespace(model=model, processor=processor, torch=torch, device=device)


def _decode_judge_response(ctx: Any, cfg: DictConfig, messages: List[Dict[str, str]]) -> str:
    processor = ctx.processor
    model = ctx.model
    torch = ctx.torch
    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=bool(getattr(cfg.stage.runtime, "enable_thinking", False)),
    )
    encoded = processor(text=chat_text, return_tensors="pt")
    model_device = next(model.parameters()).device
    encoded = {k: v.to(model_device) for k, v in encoded.items()}
    input_len = encoded["input_ids"].shape[-1]
    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=int(getattr(cfg.stage.judge, "max_new_tokens", 256)),
            do_sample=float(getattr(cfg.stage.judge, "temperature", 0.0)) > 0.0,
            temperature=float(getattr(cfg.stage.judge, "temperature", 0.0)),
            top_p=float(getattr(cfg.stage.judge, "top_p", 1.0)),
        )
    return str(processor.decode(outputs[0][input_len:], skip_special_tokens=True) or "").strip()


def _extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
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


def _edges_by_source(path: Path) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    for row in _read_csv_rows(path):
        source = int(row["source_node_idx"])
        parsed = {
            "source_node_idx": source,
            "target_node_idx": int(row["target_node_idx"]),
            "audio_rank": int(row["audio_rank"]),
            "rerank_rank": int(row["rerank_rank"]),
            "audio_similarity": float(row["audio_similarity"]),
            "text_similarity": float(row["text_similarity"]),
            "rerank_score": float(row["rerank_score"]),
        }
        out.setdefault(source, []).append(parsed)
    for rows in out.values():
        rows.sort(key=lambda item: (item["rerank_rank"], -item["rerank_score"]))
    return out


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", str(text or "").lower())


def _tag_set(row: Dict[str, str]) -> List[str]:
    tags = _parse_json_list(row.get("normalized_tags_json", ""))
    if tags:
        return sorted(set(tags))
    raw = str(row.get("tags", "") or "")
    return sorted({part.strip() for part in raw.split(",") if part.strip()})


def _caption_text(row: Dict[str, str]) -> str:
    return str(row.get("normalized_caption", "") or row.get("caption", "") or "").strip()


def _lyrics_text(row: Dict[str, str]) -> str:
    return str(row.get("normalized_lyrics", "") or row.get("lyrics", "") or "").strip()


def _combined_text(row: Dict[str, str]) -> str:
    return " ".join(part for part in (_caption_text(row), _lyrics_text(row)) if part).strip()


def _constraint_terms(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        text = str(value).strip().lower()
        if ":" in text:
            text = text.split(":", 1)[1].strip()
        if text and text not in out:
            out.append(text)
    return out


def _term_value(term: str) -> str:
    text = str(term or "").strip().lower()
    if ":" in text:
        return text.split(":", 1)[1].strip()
    return text


def _dedupe_list(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _constraint_component_scores(candidate_row: Dict[str, str], required: Sequence[str], removed: Sequence[str], persistent: Sequence[str]) -> Dict[str, float]:
    tags = set(_tag_set(candidate_row))
    vocals = str(candidate_row.get("vocals", "") or "").strip().lower()
    speed = str(candidate_row.get("speed", "") or "").strip().lower()

    def _has(term: str) -> bool:
        normalized = str(term).strip().lower()
        if normalized.startswith("vocals:"):
            return vocals == _term_value(normalized)
        if normalized.startswith("speed:"):
            return speed == _term_value(normalized)
        return _term_value(normalized) in tags

    def _missing(term: str) -> bool:
        normalized = str(term).strip().lower()
        if normalized.startswith("vocals:"):
            return vocals != _term_value(normalized)
        if normalized.startswith("speed:"):
            return speed != _term_value(normalized)
        return _term_value(normalized) not in tags

    add_score = 1.0 if not required else sum(1.0 for term in required if _has(term)) / len(required)
    remove_score = 1.0 if not removed else sum(1.0 for term in removed if _missing(term)) / len(removed)
    preserve_score = 1.0 if not persistent else sum(1.0 for term in persistent if _has(term)) / len(persistent)
    mean_score = (add_score + remove_score + preserve_score) / 3.0
    return {
        "add_score": add_score,
        "remove_score": remove_score,
        "preserve_score": preserve_score,
        "mean_score": mean_score,
    }


def _score_constraint_match(candidate_row: Dict[str, str], required: Sequence[str], removed: Sequence[str], persistent: Sequence[str]) -> float:
    return _constraint_component_scores(candidate_row, required, removed, persistent)["mean_score"]


def _caption_overlap(candidate_row: Dict[str, str], target_row: Dict[str, str]) -> float:
    cand = set(_tokenize(_combined_text(candidate_row)))
    target = set(_tokenize(_combined_text(target_row)))
    if not cand or not target:
        return 0.0
    return len(cand & target) / max(1, len(target))


def _caption_precision(candidate_row: Dict[str, str], target_row: Dict[str, str]) -> float:
    cand = set(_tokenize(_combined_text(candidate_row)))
    target = set(_tokenize(_combined_text(target_row)))
    if not cand or not target:
        return 0.0
    return len(cand & target) / max(1, len(cand))


def _tag_overlap(candidate_row: Dict[str, str], target_row: Dict[str, str]) -> float:
    cand = set(_tag_set(candidate_row))
    target = set(_tag_set(target_row))
    if not target:
        return 0.0
    return len(cand & target) / len(target)


def _tag_precision(candidate_row: Dict[str, str], target_row: Dict[str, str]) -> float:
    cand = set(_tag_set(candidate_row))
    target = set(_tag_set(target_row))
    if not cand or not target:
        return 0.0
    return len(cand & target) / len(cand)


def _grade_label(score: float, cfg: DictConfig) -> Tuple[int, str]:
    if score >= float(cfg.stage.pool.strong_threshold):
        return 4, "strong_match"
    if score >= float(cfg.stage.pool.good_threshold):
        return 3, "good_match"
    if score >= float(cfg.stage.pool.partial_threshold):
        return 2, "partial_match"
    if score >= float(cfg.stage.pool.near_miss_threshold):
        return 1, "near_miss"
    return 0, "miss"


def _validation_status_index(path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    return {_key(record): dict(record.get("validation", {}) or {}) for record in _read_jsonl(path)}


def _validation_record_index(path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    return {_key(record): record for record in _read_jsonl(path)}


def _fallback_semantic_delta(prepared: Dict[str, Any]) -> Dict[str, Any]:
    delta = dict(prepared.get("delta_from_previous", {}) or {})
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
    return {
        "preserved": [str(x) for x in prepared.get("persistent_constraints", {}).get("from_previous", []) or [] if str(x).strip()],
        "new": [str(x) for x in prepared.get("new_constraints", {}).get("from_previous", []) or [] if str(x).strip()],
        "lost": [str(x) for x in prepared.get("removed_constraints", {}).get("from_previous", []) or [] if str(x).strip()],
        "primary_edit": "",
        "caption_only_change": bool(caption_only_change),
    }


def _semantic_delta(prepared: Dict[str, Any], validation_record: Dict[str, Any] | None, *, field_name: str) -> Dict[str, Any]:
    value = None
    if validation_record is not None:
        value = validation_record.get(field_name)
        if value is None and field_name == "semantic_delta_full":
            value = validation_record.get("semantic_constraints")
    if isinstance(value, dict):
        return {
            "preserved": [str(x) for x in value.get("preserved", []) if str(x).strip()] if isinstance(value.get("preserved", []), list) else [],
            "new": [str(x) for x in value.get("new", []) if str(x).strip()] if isinstance(value.get("new", []), list) else [],
            "lost": [str(x) for x in value.get("lost", []) if str(x).strip()] if isinstance(value.get("lost", []), list) else [],
            "primary_edit": str(value.get("primary_edit", "") or "").strip(),
            "caption_only_change": bool(value.get("caption_only_change", False)),
        }
    return _fallback_semantic_delta(prepared)


def _semantic_delta_pair(prepared: Dict[str, Any], validation_record: Dict[str, Any] | None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    full = _semantic_delta(prepared, validation_record, field_name="semantic_delta_full")
    verbalized = _semantic_delta(prepared, validation_record, field_name="semantic_delta_verbalized")
    if not verbalized.get("primary_edit"):
        verbalized = full
    return full, verbalized


def _semantic_delta_typed(validation_record: Dict[str, Any] | None, prepared: Dict[str, Any], *, field_name: str, delta: Dict[str, Any]) -> Dict[str, Any]:
    value = None
    if validation_record is not None:
        value = validation_record.get(field_name)
    if isinstance(value, dict):
        return value
    return build_typed_semantic_delta(prepared, delta)


def _semantic_delta_typed_pair(prepared: Dict[str, Any], validation_record: Dict[str, Any] | None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    full, verbalized = _semantic_delta_pair(prepared, validation_record)
    full_typed = _semantic_delta_typed(validation_record, prepared, field_name="semantic_delta_full_typed", delta=full)
    verbalized_typed = _semantic_delta_typed(validation_record, prepared, field_name="semantic_delta_verbalized_typed", delta=verbalized)
    return full_typed, verbalized_typed


def _candidate_has_term(candidate_row: Dict[str, str], term: str) -> bool:
    normalized = str(term).strip().lower()
    if not normalized:
        return False
    tags = set(_tag_set(candidate_row))
    caption = _combined_text(candidate_row).lower()
    vocals = str(candidate_row.get("vocals", "") or "").strip().lower()
    speed = str(candidate_row.get("speed", "") or "").strip().lower()
    if normalized.startswith("vocals:"):
        return vocals == _term_value(normalized)
    if normalized.startswith("speed:"):
        return speed == _term_value(normalized)
    term_value = _term_value(normalized)
    return term_value in tags or term_value in caption


def _candidate_lacks_term(candidate_row: Dict[str, str], term: str) -> bool:
    normalized = str(term).strip().lower()
    if not normalized:
        return False
    tags = set(_tag_set(candidate_row))
    caption = _combined_text(candidate_row).lower()
    vocals = str(candidate_row.get("vocals", "") or "").strip().lower()
    speed = str(candidate_row.get("speed", "") or "").strip().lower()
    if normalized.startswith("vocals:"):
        return vocals != _term_value(normalized)
    if normalized.startswith("speed:"):
        return speed != _term_value(normalized)
    term_value = _term_value(normalized)
    return term_value not in tags and term_value not in caption


def _candidate_caption_has_term(candidate_row: Dict[str, str], term: str) -> bool:
    normalized = _term_value(str(term).strip().lower())
    if not normalized:
        return False
    return normalized in _combined_text(candidate_row).lower()


def _candidate_caption_lacks_term(candidate_row: Dict[str, str], term: str) -> bool:
    normalized = _term_value(str(term).strip().lower())
    if not normalized:
        return False
    return normalized not in _combined_text(candidate_row).lower()


def _constraint_status_lists(candidate_row: Dict[str, str], required: Sequence[str], removed: Sequence[str], persistent: Sequence[str]) -> Tuple[List[str], List[str]]:
    satisfied: List[str] = []
    failed: List[str] = []
    for term in required:
        (satisfied if _candidate_has_term(candidate_row, term) else failed).append(str(term))
    for term in persistent:
        (satisfied if _candidate_has_term(candidate_row, term) else failed).append(str(term))
    for term in removed:
        (satisfied if _candidate_lacks_term(candidate_row, term) else failed).append(str(term))
    return satisfied, failed


def _caption_similarity(candidate_row: Dict[str, str], reference_row: Dict[str, str]) -> float:
    return max(_caption_overlap(candidate_row, reference_row), _caption_precision(candidate_row, reference_row))


def _caption_constraint_component_scores(candidate_row: Dict[str, str], required: Sequence[str], removed: Sequence[str], persistent: Sequence[str]) -> Dict[str, float]:
    add_score = 1.0 if not required else sum(1.0 for term in required if _candidate_caption_has_term(candidate_row, term)) / len(required)
    remove_score = 1.0 if not removed else sum(1.0 for term in removed if _candidate_caption_lacks_term(candidate_row, term)) / len(removed)
    preserve_score = 1.0 if not persistent else sum(1.0 for term in persistent if _candidate_caption_has_term(candidate_row, term)) / len(persistent)
    mean_score = (add_score + remove_score + preserve_score) / 3.0
    return {
        "add_score": add_score,
        "remove_score": remove_score,
        "preserve_score": preserve_score,
        "mean_score": mean_score,
    }


def _caption_constraint_status_lists(candidate_row: Dict[str, str], required: Sequence[str], removed: Sequence[str], persistent: Sequence[str]) -> Tuple[List[str], List[str]]:
    satisfied: List[str] = []
    failed: List[str] = []
    for term in required:
        (satisfied if _candidate_caption_has_term(candidate_row, term) else failed).append(str(term))
    for term in persistent:
        (satisfied if _candidate_caption_has_term(candidate_row, term) else failed).append(str(term))
    for term in removed:
        (satisfied if _candidate_caption_lacks_term(candidate_row, term) else failed).append(str(term))
    return satisfied, failed


def _candidate_query_score(instruction: str, candidate_row: Dict[str, str]) -> float:
    query_tokens = [tok for tok in _tokenize(instruction) if len(tok) >= 4]
    if not query_tokens:
        return 0.0
    candidate_tokens = set(_tokenize(_caption_text(candidate_row)))
    for tag in _tag_set(candidate_row):
        candidate_tokens.update(_tokenize(tag))
    candidate_tokens.update(_tokenize(str(candidate_row.get("vocals", "") or "")))
    candidate_tokens.update(_tokenize(str(candidate_row.get("speed", "") or "")))
    hits = sum(1 for token in query_tokens if token in candidate_tokens)
    return hits / max(1, len(query_tokens))


def _load_text_embedding(path_str: str, cache: Dict[str, Any]) -> Any:
    np = _np()
    path = str(path_str or "").strip()
    if not path:
        return None
    cached = cache.get(path)
    if cached is not None:
        return cached
    arr = np.load(path).astype(np.float32, copy=False).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm > 1e-12:
        arr = arr / norm
    cache[path] = arr
    return arr


def _embedding_similarity(a: Any, b: Any) -> float:
    np = _np()
    if a is None or b is None:
        return 0.0
    return float(np.dot(a.reshape(-1), b.reshape(-1)))


def _candidate_llm_judge_prompt(
    *,
    candidate_clip_id: str,
    semantic_delta_verbalized: Dict[str, Any],
    semantic_delta_verbalized_typed: Dict[str, Any],
    semantic_delta_full: Dict[str, Any],
    heuristic_candidate: Dict[str, Any],
    source_row: Dict[str, str],
    target_row: Dict[str, str],
    candidate_row: Dict[str, str],
) -> List[Dict[str, str]]:
    payload = {
        "candidate_clip_id": candidate_clip_id,
        "semantic_delta_full": semantic_delta_full,
        "semantic_delta_verbalized": semantic_delta_verbalized,
        "semantic_delta_verbalized_typed": semantic_delta_verbalized_typed,
        "source_caption": _caption_text(source_row),
        "source_tags": _tag_set(source_row),
        "target_caption": _caption_text(target_row),
        "target_tags": _tag_set(target_row),
        "candidate_caption": _caption_text(candidate_row),
        "candidate_tags": _tag_set(candidate_row),
        "candidate_vocals": str(candidate_row.get("vocals", "") or ""),
        "candidate_speed": str(candidate_row.get("speed", "") or ""),
        "heuristic_candidate": heuristic_candidate,
    }
    user_content = (
        "You are judging the evaluation failure mode for one candidate in a compositional music-retrieval benchmark.\n"
        "Return exactly one JSON object and nothing else.\n"
        "Output format:\n"
        '{"pool_type": "Type_T", "grade": 2, "failure_category": "caption_miss", "confidence": 0.0, "reason": "...", "satisfied_constraints": ["..."], "failed_constraints": ["..."]}\n\n'
        "Rules:\n"
        "1. Use the semantic deltas and the candidate content as the source of truth. Heuristic metadata is advisory only.\n"
        "2. Choose exactly one pool type from: Type_TARGET, Type_STRONG, Type_H, Type_T, Type_PARTIAL, Type_HARD_NEG.\n"
        "3. Type_TARGET is only for the exact target.\n"
        "4. Type_STRONG means the candidate satisfies the requested verbalized semantics and preserves the needed prior intent.\n"
        "5. Type_H means it fits the latest edit but violates earlier preserved intent, i.e. a history shortcut.\n"
        "6. Type_T means deterministic or tag-level constraints mostly fit, but caption semantics miss.\n"
        "7. Type_PARTIAL means the candidate is close but misses part of the requested semantics.\n"
        "8. Type_HARD_NEG means it is clearly wrong overall.\n"
        "9. Provide the grade that matches your pool type and concise satisfied/failed constraint lists.\n\n"
        f"Payload:\n{json.dumps(payload, ensure_ascii=True, indent=2)}"
    )
    return [
        {"role": "system", "content": "You are a strict semantic matching judge and output JSON only."},
        {"role": "user", "content": user_content},
    ]


def _run_candidate_llm_judge(ctx: Any, cfg: DictConfig, **kwargs: Any) -> Dict[str, Any]:
    raw = _decode_judge_response(ctx, cfg, _candidate_llm_judge_prompt(**kwargs))
    parsed = _extract_json_object(raw)
    pool_type = str(parsed.get("pool_type", "") or "").strip()
    grade = int(parsed.get("grade", 0) or 0)
    satisfied = parsed.get("satisfied_constraints", [])
    failed = parsed.get("failed_constraints", [])
    return {
        "pool_type": pool_type,
        "grade": grade,
        "failure_category": str(parsed.get("failure_category", "") or "").strip(),
        "confidence": float(parsed.get("confidence", 0.0) or 0.0),
        "reason": str(parsed.get("reason", "") or "").strip(),
        "satisfied_constraints": [str(x) for x in satisfied if str(x).strip()] if isinstance(satisfied, list) else [],
        "failed_constraints": [str(x) for x in failed if str(x).strip()] if isinstance(failed, list) else [],
    }


def _label_from_grade(grade: int) -> str:
    if grade >= 4:
        return "strong_match"
    if grade == 3:
        return "good_match"
    if grade == 2:
        return "partial_match"
    if grade == 1:
        return "near_miss"
    return "miss"


def _reason_code_from_candidate(item: Dict[str, Any]) -> str:
    pool_type = str(item.get("pool_type", "") or "")
    failure_category = str(item.get("failure_category", "") or "")
    if pool_type == "Type_TARGET":
        return "success:target"
    if pool_type == "Type_STRONG":
        return "success:strong"
    if failure_category == "history_shortcut":
        return "failed:history_shortcut"
    if failure_category == "caption_miss":
        return "failed:caption_miss"
    if failure_category in {"partial_constraint_miss", "soft_semantic_partial"}:
        return "failed:partial_constraint_miss"
    if failure_category == "hard_negative":
        return "failed:hard_negative"
    if failure_category:
        return f"failed:{failure_category}"
    return "failed:unclassified"


def _pool_type_defaults(pool_type: str) -> Tuple[int, str]:
    mapping = {
        "Type_TARGET": (4, "target"),
        "Type_STRONG": (3, "strong_positive"),
        "Type_H": (1, "history_shortcut"),
        "Type_T": (2, "caption_miss"),
        "Type_PARTIAL": (1, "partial_constraint_miss"),
        "Type_HARD_NEG": (0, "hard_negative"),
    }
    return mapping.get(str(pool_type), (0, "hard_negative"))


def _desired_pool_counts(max_candidates: int) -> Dict[str, int]:
    if max_candidates <= 32:
        return {
            "Type_TARGET": 1,
            "Type_STRONG": 3,
            "Type_H": 4,
            "Type_T": 3,
            "Type_PARTIAL": 3,
            "Type_HARD_NEG": max(2, max_candidates - 14),
        }
    return {
        "Type_TARGET": 1,
        "Type_STRONG": 10,
        "Type_H": 12,
        "Type_T": 8,
        "Type_PARTIAL": 6,
        "Type_HARD_NEG": max(8, max_candidates - 37),
    }


def _constructive_candidate_metadata(
    *,
    cand: Dict[str, Any],
    candidate_row: Dict[str, str],
    target_row: Dict[str, str],
    source_row: Dict[str, str],
    prepared: Dict[str, Any],
    semantic_full: Dict[str, Any],
    semantic_verbalized: Dict[str, Any],
    semantic_full_typed: Dict[str, Any],
    semantic_verbalized_typed: Dict[str, Any],
    candidate_text_embedding: Any,
    target_text_embedding: Any,
    cfg: DictConfig,
) -> Dict[str, Any]:
    full_required = list(semantic_full.get("new", []) or [])
    full_removed = list(semantic_full.get("lost", []) or [])
    full_persistent = list(semantic_full.get("preserved", []) or [])
    verbalized_required = list(semantic_verbalized.get("new", []) or [])
    verbalized_removed = list(semantic_verbalized.get("lost", []) or [])
    verbalized_persistent = list(semantic_verbalized.get("preserved", []) or [])
    seed_required = list(prepared.get("new_constraints", {}).get("from_seed", []) or [])
    seed_removed = list(prepared.get("removed_constraints", {}).get("from_seed", []) or [])
    seed_persistent = list(prepared.get("persistent_constraints", {}).get("from_seed", []) or [])

    full_components = _constraint_component_scores(candidate_row, full_required, full_removed, full_persistent)
    verbalized_components = _constraint_component_scores(candidate_row, verbalized_required, verbalized_removed, verbalized_persistent)
    seed_components = _constraint_component_scores(candidate_row, seed_required, seed_removed, seed_persistent)
    full_satisfied, full_failed = _constraint_status_lists(candidate_row, full_required, full_removed, full_persistent)
    verbalized_satisfied, verbalized_failed = _constraint_status_lists(candidate_row, verbalized_required, verbalized_removed, verbalized_persistent)
    seed_satisfied, seed_failed = _constraint_status_lists(candidate_row, seed_required, seed_removed, seed_persistent)
    full_caption_required = typed_item_texts(semantic_full_typed, source="caption", buckets=("new_items",)) + typed_item_texts(semantic_full_typed, source="lyrics", buckets=("new_items",))
    full_caption_removed = typed_item_texts(semantic_full_typed, source="caption", buckets=("lost_items",)) + typed_item_texts(semantic_full_typed, source="lyrics", buckets=("lost_items",))
    full_caption_persistent = typed_item_texts(semantic_full_typed, source="caption", buckets=("preserved_items",)) + typed_item_texts(semantic_full_typed, source="lyrics", buckets=("preserved_items",))
    verbalized_caption_required = typed_item_texts(semantic_verbalized_typed, source="caption", buckets=("new_items",)) + typed_item_texts(semantic_verbalized_typed, source="lyrics", buckets=("new_items",))
    verbalized_caption_removed = typed_item_texts(semantic_verbalized_typed, source="caption", buckets=("lost_items",)) + typed_item_texts(semantic_verbalized_typed, source="lyrics", buckets=("lost_items",))
    verbalized_caption_persistent = typed_item_texts(semantic_verbalized_typed, source="caption", buckets=("preserved_items",)) + typed_item_texts(semantic_verbalized_typed, source="lyrics", buckets=("preserved_items",))
    full_caption_components = _caption_constraint_component_scores(candidate_row, full_caption_required, full_caption_removed, full_caption_persistent)
    verbalized_caption_components = _caption_constraint_component_scores(candidate_row, verbalized_caption_required, verbalized_caption_removed, verbalized_caption_persistent)
    full_caption_satisfied, full_caption_failed = _caption_constraint_status_lists(candidate_row, full_caption_required, full_caption_removed, full_caption_persistent)
    verbalized_caption_satisfied, verbalized_caption_failed = _caption_constraint_status_lists(candidate_row, verbalized_caption_required, verbalized_caption_removed, verbalized_caption_persistent)

    tag_overlap = _tag_overlap(candidate_row, target_row)
    tag_precision = _tag_precision(candidate_row, target_row)
    caption_overlap = _caption_overlap(candidate_row, target_row)
    caption_precision = _caption_precision(candidate_row, target_row)
    caption_sim_to_target = _caption_similarity(candidate_row, target_row)
    caption_sim_to_source = _caption_similarity(candidate_row, source_row)
    caption_embedding_sim_to_target = _embedding_similarity(candidate_text_embedding, target_text_embedding)
    semantic_score = max(float(cand.get("rerank_score", 0.0)), float(cand.get("text_similarity", 0.0)))
    weight_total = (
        float(cfg.stage.pool.seed_constraint_weight)
        + float(cfg.stage.pool.prev_constraint_weight)
        + float(cfg.stage.pool.target_tag_recall_weight)
        + float(cfg.stage.pool.target_tag_precision_weight)
        + float(cfg.stage.pool.target_caption_recall_weight)
        + float(cfg.stage.pool.target_caption_precision_weight)
        + float(cfg.stage.pool.semantic_score_weight)
    )
    final_score = (
        float(cfg.stage.pool.seed_constraint_weight) * seed_components["mean_score"]
        + float(cfg.stage.pool.prev_constraint_weight) * verbalized_components["mean_score"]
        + float(cfg.stage.pool.target_tag_recall_weight) * tag_overlap
        + float(cfg.stage.pool.target_tag_precision_weight) * tag_precision
        + float(cfg.stage.pool.target_caption_recall_weight) * caption_overlap
        + float(cfg.stage.pool.target_caption_precision_weight) * caption_precision
        + float(cfg.stage.pool.semantic_score_weight) * semantic_score
    ) / max(weight_total, 1e-12)

    satisfies_new_edit = len(verbalized_failed) == 0
    preserves_accumulated_tags = len(seed_failed) == 0
    preserves_accumulated_caption_constraints = True
    if full_caption_persistent:
        preserves_accumulated_caption_constraints = max(
            float(full_caption_components["preserve_score"]),
            float(caption_sim_to_target),
        ) >= float(getattr(cfg.stage.pool, "caption_moderate_threshold", 0.35))
    caption_alignment = max(
        float(caption_sim_to_target),
        float(verbalized_caption_components["mean_score"]),
        float(caption_embedding_sim_to_target),
    )
    matches_target_caption_semantics = caption_alignment >= float(getattr(cfg.stage.pool, "caption_strong_threshold", 0.55))
    target_vocals = str(target_row.get("vocals", "") or "").strip().lower()
    target_speed = str(target_row.get("speed", "") or "").strip().lower()
    cand_vocals = str(candidate_row.get("vocals", "") or "").strip().lower()
    cand_speed = str(candidate_row.get("speed", "") or "").strip().lower()
    satisfies_vocal_status = True if not target_vocals else cand_vocals == target_vocals
    satisfies_speed_constraint = True if not target_speed else cand_speed == target_speed
    history_shortcut_detected = bool(satisfies_new_edit and not preserves_accumulated_tags)
    missing_count = len(verbalized_failed)
    is_exact_target = bool(cand.get("is_exact_target", False))

    pool_type = "Type_HARD_NEG"
    grade = 0
    failure_category = "hard_negative"
    failed_constraints = list(verbalized_failed)
    satisfied_constraints = _dedupe_list(verbalized_satisfied)
    if not failed_constraints and not matches_target_caption_semantics:
        failed_constraints.append("target caption semantics")

    if is_exact_target:
        pool_type = "Type_TARGET"
        grade = 4
        failure_category = "target"
        failed_constraints = []
    elif history_shortcut_detected:
        pool_type = "Type_H"
        grade = 1
        failure_category = "history_shortcut"
    elif preserves_accumulated_tags and satisfies_new_edit and satisfies_vocal_status and satisfies_speed_constraint and caption_alignment >= float(getattr(cfg.stage.pool, "caption_moderate_threshold", 0.35)):
        pool_type = "Type_STRONG"
        grade = 4 if matches_target_caption_semantics else 3
        failure_category = "strong_positive"
        failed_constraints = [] if grade >= 4 else ["moderate caption alignment"]
    elif preserves_accumulated_tags and satisfies_new_edit and caption_alignment < float(getattr(cfg.stage.pool, "caption_miss_threshold", 0.2)):
        pool_type = "Type_T"
        grade = 2
        failure_category = "caption_miss"
        failed_constraints = failed_constraints or _dedupe_list(verbalized_caption_failed) or ["target caption semantics"]
    elif missing_count == 1:
        pool_type = "Type_PARTIAL"
        grade = 1
        failure_category = "partial_constraint_miss"
    elif preserves_accumulated_tags and satisfies_new_edit and not matches_target_caption_semantics:
        pool_type = "Type_PARTIAL"
        grade = 2
        failure_category = "soft_semantic_partial"

    return {
        "pool_type": pool_type,
        "grade": grade,
        "label": _label_from_grade(grade),
        "failure_category": failure_category,
        "failed_constraints": failed_constraints,
        "satisfied_constraints": satisfied_constraints,
        "constraint_satisfaction": {
            "satisfies_new_edit": bool(satisfies_new_edit),
            "preserves_accumulated_tags": bool(preserves_accumulated_tags),
            "preserves_accumulated_caption_constraints": bool(preserves_accumulated_caption_constraints),
            "matches_target_caption_semantics": bool(matches_target_caption_semantics),
            "satisfies_vocal_status": bool(satisfies_vocal_status),
            "satisfies_speed_constraint": bool(satisfies_speed_constraint),
            "history_shortcut_detected": bool(history_shortcut_detected),
        },
        "verbalized_constraint_satisfaction": {
            "satisfied": _dedupe_list(verbalized_satisfied),
            "failed": _dedupe_list(verbalized_failed),
            "score": round(float(verbalized_components["mean_score"]), 6),
        },
        "full_constraint_satisfaction": {
            "satisfied": _dedupe_list(full_satisfied),
            "failed": _dedupe_list(full_failed),
            "score": round(float(full_components["mean_score"]), 6),
        },
        "caption_constraint_satisfaction": {
            "verbalized_satisfied": _dedupe_list(verbalized_caption_satisfied),
            "verbalized_failed": _dedupe_list(verbalized_caption_failed),
            "verbalized_score": round(float(verbalized_caption_components["mean_score"]), 6),
            "full_satisfied": _dedupe_list(full_caption_satisfied),
            "full_failed": _dedupe_list(full_caption_failed),
            "full_score": round(float(full_caption_components["mean_score"]), 6),
        },
        "seed_constraint_components": {k: round(float(v), 6) for k, v in seed_components.items()},
        "prev_constraint_components": {k: round(float(v), 6) for k, v in verbalized_components.items()},
        "seed_constraint_score": round(float(seed_components["mean_score"]), 6),
        "prev_constraint_score": round(float(verbalized_components["mean_score"]), 6),
        "tag_overlap_to_target": round(float(tag_overlap), 6),
        "tag_precision_to_target": round(float(tag_precision), 6),
        "caption_overlap_to_target": round(float(caption_overlap), 6),
        "caption_precision_to_target": round(float(caption_precision), 6),
        "caption_sim_to_target": round(float(caption_sim_to_target), 6),
        "caption_embedding_sim_to_target": round(float(caption_embedding_sim_to_target), 6),
        "caption_alignment_score": round(float(caption_alignment), 6),
        "caption_sim_to_source": round(float(caption_sim_to_source), 6),
        "semantic_score": round(float(semantic_score), 6),
        "final_score": round(float(final_score), 6),
        "history_shortcut": bool(history_shortcut_detected),
        "label_source": "deterministic",
        "constructive_rank_score": round(float(final_score), 6),
    }


def _candidate_source_rows(
    *,
    source_node_idx: int,
    target_node_idx: int | None,
    seed_node_idx: int | None,
    history_node_indices: Sequence[int],
    edges_by_source: Dict[int, List[Dict[str, Any]]],
    cfg: DictConfig,
) -> List[Tuple[str, Dict[str, Any]]]:
    sources: List[Tuple[str, Dict[str, Any]]] = []

    def _extend(label: str, node_idx: int | None) -> None:
        if node_idx is None or node_idx < 0:
            return
        for edge in edges_by_source.get(node_idx, []):
            sources.append((label, edge))

    _extend("source_neighborhood", source_node_idx)
    if bool(getattr(cfg.stage.pool, "include_target_neighborhood", True)):
        _extend("target_neighborhood", target_node_idx)
    if bool(cfg.stage.pool.include_seed_neighborhood):
        _extend("seed_neighborhood", seed_node_idx)
    if bool(cfg.stage.pool.include_history_reference_neighborhood):
        for node_idx in history_node_indices:
            _extend("history_reference_neighborhood", node_idx)
    return sources


def run_relevance_pool(cfg: DictConfig) -> Dict[str, object]:
    manifest_path = Path(str(cfg.stage.io.input_manifest_csv))
    lookup_manifest_path = Path(str(cfg.stage.io.input_lookup_manifest_csv))
    nodes_path = Path(str(cfg.stage.io.input_nodes_csv))
    edges_path = Path(str(cfg.stage.io.input_edges_csv))
    chains_path = Path(str(cfg.stage.io.input_chains_jsonl))
    prepared_path = Path(str(cfg.stage.io.input_prepared_jsonl))
    validation_path = Path(str(cfg.stage.io.input_validation_jsonl))
    out_dir = Path(str(cfg.stage.io.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / str(cfg.stage.io.output_pools_jsonl)
    reason_labels_jsonl = out_dir / str(getattr(cfg.stage.io, "output_reason_labels_jsonl", "chain_step_candidate_reason_labels.jsonl"))
    report_path = out_dir / str(cfg.stage.io.report_file)
    tracker = StageTracker(
        cfg,
        "relevance_pool",
        title="Assemble Relevance Pools",
        subtitle=f"chains={chains_path}",
        total_steps=4,
    )

    use_text_encoder_audit = bool(getattr(cfg.stage.behavior, "use_text_encoder_audit", True))
    use_candidate_llm_judge = bool(
        getattr(
            cfg.stage.behavior,
            "use_candidate_llm_judge",
            getattr(cfg.stage.behavior, "use_caption_llm_judge", True),
        )
    )
    required = [manifest_path, nodes_path, edges_path, chains_path, prepared_path]
    if use_text_encoder_audit:
        required.append(lookup_manifest_path)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required relevance-pool input not found: {path}")
    if bool(cfg.stage.behavior.require_validated_instructions) and not validation_path.exists():
        raise FileNotFoundError(f"Validation input required but not found: {validation_path}")

    tracker.step("Load manifests, graph edges, and prepared records", detail=f"manifest={manifest_path.name}")
    structured_by_clip = _structured_index(manifest_path)
    clip_ids_by_split: Dict[str, List[str]] = {}
    for clip_id, row in structured_by_clip.items():
        split = str(row.get("split", "") or "")
        clip_ids_by_split.setdefault(split, []).append(clip_id)
    nodes_by_idx = _node_index(nodes_path)
    clip_to_node = _clip_to_node_index(nodes_by_idx)
    edges_by_source = _edges_by_source(edges_path)
    prepared_by_key = {_key(record): record for record in _read_jsonl(prepared_path)}
    validation_by_key = _validation_record_index(validation_path) if validation_path.exists() else {}
    lookup_by_clip = _lookup_manifest_index(lookup_manifest_path) if lookup_manifest_path.exists() else {}
    text_embedding_cache: Dict[str, Any] = {}
    chains = list(_read_jsonl(chains_path))
    max_steps = cfg.stage.behavior.max_steps
    max_candidates = max(1, int(cfg.stage.pool.max_candidates_per_step))
    min_rerank = cfg.stage.pool.min_rerank_score
    min_rerank_score = None if min_rerank in (None, "") else float(min_rerank)
    only_keep_passed = bool(cfg.stage.behavior.only_keep_validation_passed)
    write_full_dataset_labels = bool(getattr(cfg.stage.behavior, "write_full_dataset_labels", False))
    label_splits = {str(x) for x in getattr(cfg.stage.behavior, "full_dataset_label_splits", []) or []}
    run_solvability_audit = bool(getattr(cfg.stage.behavior, "run_solvability_audit", True))
    solvability_top_k = max(1, int(getattr(cfg.stage.behavior, "solvability_top_k", 3)))
    solvability_threshold = float(getattr(cfg.stage.behavior, "solvability_score_threshold", 0.6))
    candidate_judge_splits = {
        str(x)
        for x in getattr(
            cfg.stage.behavior,
            "candidate_llm_judge_splits",
            getattr(cfg.stage.behavior, "caption_llm_judge_splits", []),
        )
        or []
    }
    tracker.step(
        "Resolve optional audit models",
        detail=f"text_encoder={use_text_encoder_audit}, llm_judge={use_candidate_llm_judge}",
    )
    text_ctx = _build_text_embedder(cfg) if use_text_encoder_audit else None
    llm_judge_ctx = _build_llm_judge(cfg) if use_candidate_llm_judge else None

    counts: Dict[str, Any] = {
        "steps_seen": 0,
        "steps_written": 0,
        "steps_skipped_missing_prepared": 0,
        "steps_skipped_validation": 0,
        "candidate_rows_written": 0,
        "category_counts": {
            "strong_match": 0,
            "good_match": 0,
            "partial_match": 0,
            "near_miss": 0,
            "miss": 0,
        },
        "pool_type_counts": {
            "Type_TARGET": 0,
            "Type_STRONG": 0,
            "Type_H": 0,
            "Type_T": 0,
            "Type_PARTIAL": 0,
            "Type_HARD_NEG": 0,
        },
        "history_shortcut_negatives": 0,
        "steps_with_history_shortcut_candidates": 0,
        "steps_with_caption_only_change": 0,
        "full_dataset_labels_written": 0,
        "solvability_flagged": 0,
        "candidate_llm_judge_calls": 0,
        "candidate_llm_judge_label_changes": 0,
    }

    tracker.step("Construct candidate pools", detail=f"max_steps={max_steps if max_steps is not None else 'all'}")
    with out_jsonl.open("w", encoding="utf-8") as out_f:
        reason_f = reason_labels_jsonl.open("w", encoding="utf-8") if write_full_dataset_labels else None
        try:
            total_hint = None if max_steps is None else int(max_steps)
            with rich_tqdm(cfg, total=total_hint, desc="Relevance pools", unit="step") as progress:
                stop = False
                for chain in chains:
                    chain_id = str(chain.get("chain_id", "") or "")
                    seed_clip_id = str(chain.get("seed", {}).get("clip_id", "") or "")
                    seed_node_idx = clip_to_node.get(seed_clip_id)
                    for turn_index, step in enumerate(list(chain.get("steps", [])), start=1):
                        if max_steps is not None and counts["steps_seen"] >= int(max_steps):
                            stop = True
                            break
                        counts["steps_seen"] += 1
                        key = (chain_id, turn_index)
                        prepared = prepared_by_key.get(key)
                        if prepared is None:
                            counts["steps_skipped_missing_prepared"] += 1
                            progress.update(1)
                            continue
                        validation_record = validation_by_key.get(key)
                        validation = dict(validation_record.get("validation", {}) or {}) if validation_record is not None else {}
                        if only_keep_passed and validation and not bool(validation.get("accepted", False)):
                            counts["steps_skipped_validation"] += 1
                            progress.update(1)
                            continue

                        source_node_idx = int(step.get("source_node_idx", -1) or -1)
                        target_node_idx = int(step.get("target_node_idx", -1) or -1)
                        target_clip_id = str(step.get("target_clip_id", "") or "")
                        source_clip_id = str(step.get("source_clip_id", "") or "")
                        target_row = structured_by_clip.get(target_clip_id)
                        source_row = structured_by_clip.get(source_clip_id)
                        if target_row is None or source_row is None:
                            progress.update(1)
                            continue
                        target_lookup = lookup_by_clip.get(target_clip_id, {})
                        target_text_embedding = _load_text_embedding(str(target_lookup.get("text_embedding_path", "") or ""), text_embedding_cache)

                        semantic_full, semantic_verbalized = _semantic_delta_pair(prepared, validation_record)
                        semantic_full_typed, semantic_verbalized_typed = _semantic_delta_typed_pair(prepared, validation_record)
                        if bool(semantic_full.get("caption_only_change", False)):
                            counts["steps_with_caption_only_change"] += 1
                        history_node_indices: List[int] = []
                        if bool(cfg.stage.pool.include_history_reference_neighborhood):
                            for ref in list(prepared.get("history_reference_candidates", []) or []):
                                prior_clip_id = str(ref.get("prior_clip_id", "") or "")
                                node_idx = clip_to_node.get(prior_clip_id)
                                if node_idx is not None and node_idx not in history_node_indices:
                                    history_node_indices.append(node_idx)

                        candidate_edge_sources = _candidate_source_rows(
                            source_node_idx=source_node_idx,
                            target_node_idx=target_node_idx,
                            seed_node_idx=seed_node_idx,
                            history_node_indices=history_node_indices,
                            edges_by_source=edges_by_source,
                            cfg=cfg,
                        )
                        candidate_rows_by_clip: Dict[str, Dict[str, Any]] = {}

                        def _upsert_candidate(base: Dict[str, Any]) -> None:
                            clip_id = str(base.get("clip_id", "") or "")
                            if not clip_id:
                                return
                            existing = candidate_rows_by_clip.get(clip_id)
                            if existing is None:
                                candidate_rows_by_clip[clip_id] = {
                                    "clip_id": clip_id,
                                    "track_id": str(base.get("track_id", "") or ""),
                                    "node_idx": int(base.get("node_idx", -1) or -1),
                                    "audio_rank": int(base.get("audio_rank", 999999) or 999999),
                                    "rerank_rank": int(base.get("rerank_rank", 999999) or 999999),
                                    "audio_similarity": float(base.get("audio_similarity", 0.0) or 0.0),
                                    "text_similarity": float(base.get("text_similarity", 0.0) or 0.0),
                                    "rerank_score": float(base.get("rerank_score", 0.0) or 0.0),
                                    "candidate_sources": list(base.get("candidate_sources", []) or []),
                                    "audio_sim_to_target": float(base.get("audio_sim_to_target", 0.0) or 0.0),
                                    "audio_sim_to_seed": float(base.get("audio_sim_to_seed", 0.0) or 0.0),
                                    "audio_sim_to_source": float(base.get("audio_sim_to_source", 0.0) or 0.0),
                                    "is_exact_target": bool(base.get("is_exact_target", False)),
                                }
                                return
                            existing["audio_rank"] = min(int(existing["audio_rank"]), int(base.get("audio_rank", 999999) or 999999))
                            existing["rerank_rank"] = min(int(existing["rerank_rank"]), int(base.get("rerank_rank", 999999) or 999999))
                            existing["audio_similarity"] = max(float(existing["audio_similarity"]), float(base.get("audio_similarity", 0.0) or 0.0))
                            existing["text_similarity"] = max(float(existing["text_similarity"]), float(base.get("text_similarity", 0.0) or 0.0))
                            existing["rerank_score"] = max(float(existing["rerank_score"]), float(base.get("rerank_score", 0.0) or 0.0))
                            existing["audio_sim_to_target"] = max(float(existing["audio_sim_to_target"]), float(base.get("audio_sim_to_target", 0.0) or 0.0))
                            existing["audio_sim_to_seed"] = max(float(existing["audio_sim_to_seed"]), float(base.get("audio_sim_to_seed", 0.0) or 0.0))
                            existing["audio_sim_to_source"] = max(float(existing["audio_sim_to_source"]), float(base.get("audio_sim_to_source", 0.0) or 0.0))
                            existing["is_exact_target"] = bool(existing["is_exact_target"] or base.get("is_exact_target", False))
                            for source in list(base.get("candidate_sources", []) or []):
                                if source not in existing["candidate_sources"]:
                                    existing["candidate_sources"].append(source)

                        if bool(cfg.stage.pool.include_exact_target):
                            target_node = nodes_by_idx.get(target_node_idx)
                            if target_node is not None:
                                _upsert_candidate(
                                    {
                                        "clip_id": target_clip_id,
                                        "track_id": str(target_node.get("track_id", "") or ""),
                                        "node_idx": target_node_idx,
                                        "audio_rank": 0,
                                        "rerank_rank": 0,
                                        "audio_similarity": 1.0,
                                        "text_similarity": 1.0,
                                        "rerank_score": 1.0,
                                        "candidate_sources": ["exact_target"],
                                        "audio_sim_to_target": 1.0,
                                        "is_exact_target": True,
                                    }
                                )

                        if bool(cfg.stage.pool.include_source_clip):
                            source_node = nodes_by_idx.get(source_node_idx)
                            if source_clip_id and source_node is not None:
                                _upsert_candidate(
                                    {
                                        "clip_id": source_clip_id,
                                        "track_id": str(source_node.get("track_id", "") or ""),
                                        "node_idx": source_node_idx,
                                        "audio_rank": 0,
                                        "rerank_rank": 0,
                                        "audio_similarity": 1.0,
                                        "text_similarity": 1.0,
                                        "rerank_score": 1.0,
                                        "candidate_sources": ["source_clip"],
                                        "audio_sim_to_source": 1.0,
                                    }
                                )

                        if bool(cfg.stage.pool.include_history_targets):
                            for prior_step in list(chain.get("steps", []))[: max(0, turn_index - 1)]:
                                prior_clip_id = str(prior_step.get("target_clip_id", "") or "")
                                node_idx = clip_to_node.get(prior_clip_id)
                                if not prior_clip_id or node_idx is None:
                                    continue
                                node = nodes_by_idx.get(node_idx)
                                if node is None:
                                    continue
                                _upsert_candidate(
                                    {
                                        "clip_id": prior_clip_id,
                                        "track_id": str(node.get("track_id", "") or ""),
                                        "node_idx": node_idx,
                                        "audio_rank": 0,
                                        "rerank_rank": 0,
                                        "audio_similarity": 0.0,
                                        "text_similarity": 0.0,
                                        "rerank_score": 0.0,
                                        "candidate_sources": ["chain_history_target"],
                                    }
                                )

                        filtered_candidate_edge_sources: List[Tuple[str, Dict[str, Any]]] = []
                        for source_label, edge in candidate_edge_sources:
                            if min_rerank_score is not None and float(edge["rerank_score"]) < min_rerank_score:
                                continue
                            filtered_candidate_edge_sources.append((source_label, edge))
                            if len(filtered_candidate_edge_sources) >= max_candidates * 4:
                                break

                        for source_label, edge in filtered_candidate_edge_sources:
                            node = nodes_by_idx.get(int(edge["target_node_idx"]))
                            if node is None:
                                continue
                            clip_id = str(node.get("clip_id", "") or "")
                            if not clip_id:
                                continue
                            _upsert_candidate(
                                {
                                    "clip_id": clip_id,
                                    "track_id": str(node.get("track_id", "") or ""),
                                    "node_idx": int(edge["target_node_idx"]),
                                    "audio_rank": int(edge["audio_rank"]),
                                    "rerank_rank": int(edge["rerank_rank"]),
                                    "audio_similarity": float(edge["audio_similarity"]),
                                    "text_similarity": float(edge["text_similarity"]),
                                    "rerank_score": float(edge["rerank_score"]),
                                    "candidate_sources": [source_label],
                                    "audio_sim_to_target": float(edge["audio_similarity"]) if source_label == "target_neighborhood" else 0.0,
                                    "audio_sim_to_seed": float(edge["audio_similarity"]) if source_label == "seed_neighborhood" else 0.0,
                                    "audio_sim_to_source": float(edge["audio_similarity"]) if source_label == "source_neighborhood" else 0.0,
                                }
                            )
                            if len(candidate_rows_by_clip) >= max_candidates * 4:
                                break

                        all_scored_candidates: List[Dict[str, Any]] = []
                        for cand in candidate_rows_by_clip.values():
                            candidate_row = structured_by_clip.get(str(cand["clip_id"]))
                            if candidate_row is None:
                                continue
                            candidate_lookup = lookup_by_clip.get(str(cand["clip_id"]), {})
                            candidate_text_embedding = _load_text_embedding(
                                str(candidate_lookup.get("text_embedding_path", "") or ""),
                                text_embedding_cache,
                            )
                            scored = {
                                **cand,
                                **_constructive_candidate_metadata(
                                    cand=cand,
                                    candidate_row=candidate_row,
                                    target_row=target_row,
                                    source_row=source_row,
                                    prepared=prepared,
                                    semantic_full=semantic_full,
                                    semantic_verbalized=semantic_verbalized,
                                    semantic_full_typed=semantic_full_typed,
                                    semantic_verbalized_typed=semantic_verbalized_typed,
                                    candidate_text_embedding=candidate_text_embedding,
                                    target_text_embedding=target_text_embedding,
                                    cfg=cfg,
                                ),
                            }
                            scored["candidate_text_embedding_available"] = candidate_text_embedding is not None
                            all_scored_candidates.append(scored)
                            if bool(scored.get("history_shortcut", False)):
                                counts["history_shortcut_negatives"] += 1

                        target_split = str(target_row.get("split", "") or "")
                        if llm_judge_ctx is not None and (not candidate_judge_splits or target_split in candidate_judge_splits):
                            for item in all_scored_candidates:
                                if bool(item.get("is_exact_target", False)):
                                    continue
                                candidate_row = structured_by_clip.get(str(item["clip_id"]))
                                if candidate_row is None:
                                    continue
                                original_pool_type = str(item.get("pool_type", "") or "")
                                original_grade = int(item.get("grade", 0) or 0)
                                judge = _run_candidate_llm_judge(
                                    llm_judge_ctx,
                                    cfg,
                                    candidate_clip_id=str(item["clip_id"]),
                                    semantic_delta_verbalized=semantic_verbalized,
                                    semantic_delta_verbalized_typed=semantic_verbalized_typed,
                                    semantic_delta_full=semantic_full,
                                    heuristic_candidate=item,
                                    source_row=source_row,
                                    target_row=target_row,
                                    candidate_row=candidate_row,
                                )
                                counts["candidate_llm_judge_calls"] += 1
                                judged_pool_type = str(judge.get("pool_type", "") or "").strip() or original_pool_type
                                default_grade, default_failure_category = _pool_type_defaults(judged_pool_type)
                                judged_grade = int(judge.get("grade", default_grade) or default_grade)
                                item["candidate_llm_judge"] = judge
                                item["label_source"] = "llm_judge"
                                item["pool_type"] = judged_pool_type
                                item["grade"] = judged_grade
                                item["label"] = _label_from_grade(judged_grade)
                                item["failure_category"] = str(judge.get("failure_category", "") or "").strip() or default_failure_category
                                if judge.get("satisfied_constraints"):
                                    item["satisfied_constraints"] = _dedupe_list(list(judge.get("satisfied_constraints", []) or []))
                                if judge.get("failed_constraints"):
                                    item["failed_constraints"] = _dedupe_list(list(judge.get("failed_constraints", []) or []))
                                item["judge_reason"] = str(judge.get("reason", "") or "").strip()
                                item["judge_confidence"] = float(judge.get("confidence", 0.0) or 0.0)
                                if judged_pool_type != original_pool_type or judged_grade != original_grade:
                                    counts["candidate_llm_judge_label_changes"] += 1

                        desired_counts = _desired_pool_counts(max_candidates)
                        if any(str(item.get("pool_type", "")) == "Type_H" for item in all_scored_candidates):
                            counts["steps_with_history_shortcut_candidates"] += 1
                        by_type: Dict[str, List[Dict[str, Any]]] = {}
                        for item in all_scored_candidates:
                            by_type.setdefault(str(item["pool_type"]), []).append(item)
                        for items in by_type.values():
                            items.sort(
                                key=lambda item: (
                                    -int(item["grade"]),
                                    -float(item.get("constructive_rank_score", item.get("final_score", 0.0))),
                                    -float(item.get("audio_sim_to_target", 0.0)),
                                    -float(item.get("audio_sim_to_source", 0.0)),
                                )
                            )

                        selected_candidates: List[Dict[str, Any]] = []
                        leftovers: List[Dict[str, Any]] = []
                        for pool_type in ("Type_TARGET", "Type_STRONG", "Type_H", "Type_T", "Type_PARTIAL", "Type_HARD_NEG"):
                            items = by_type.get(pool_type, [])
                            keep_n = desired_counts.get(pool_type, 0)
                            selected_candidates.extend(items[:keep_n])
                            leftovers.extend(items[keep_n:])
                        if len(selected_candidates) < max_candidates:
                            leftovers.sort(
                                key=lambda item: (
                                    -int(item["grade"]),
                                    -float(item.get("constructive_rank_score", item.get("final_score", 0.0))),
                                )
                            )
                            seen_ids = {str(item["clip_id"]) for item in selected_candidates}
                            for item in leftovers:
                                clip_id = str(item["clip_id"])
                                if clip_id in seen_ids:
                                    continue
                                selected_candidates.append(item)
                                seen_ids.add(clip_id)
                                if len(selected_candidates) >= max_candidates:
                                    break
                        scored_candidates = selected_candidates[:max_candidates]

                        for item in scored_candidates:
                            counts["category_counts"][str(item["label"])] += 1
                            counts["pool_type_counts"][str(item["pool_type"])] += 1

                        pool_summary = {
                            "strong_match": sum(1 for item in scored_candidates if item["label"] == "strong_match"),
                            "good_match": sum(1 for item in scored_candidates if item["label"] == "good_match"),
                            "partial_match": sum(1 for item in scored_candidates if item["label"] == "partial_match"),
                            "near_miss": sum(1 for item in scored_candidates if item["label"] == "near_miss"),
                            "miss": sum(1 for item in scored_candidates if item["label"] == "miss"),
                        }
                        pool_type_summary = {
                            "Type_TARGET": sum(1 for item in scored_candidates if item["pool_type"] == "Type_TARGET"),
                            "Type_STRONG": sum(1 for item in scored_candidates if item["pool_type"] == "Type_STRONG"),
                            "Type_H": sum(1 for item in scored_candidates if item["pool_type"] == "Type_H"),
                            "Type_T": sum(1 for item in scored_candidates if item["pool_type"] == "Type_T"),
                            "Type_PARTIAL": sum(1 for item in scored_candidates if item["pool_type"] == "Type_PARTIAL"),
                            "Type_HARD_NEG": sum(1 for item in scored_candidates if item["pool_type"] == "Type_HARD_NEG"),
                        }
                        solvability_audit = None
                        if run_solvability_audit and validation_record is not None:
                            instruction = str(validation_record.get("history_unaware_instruction", "") or "").strip()
                            if instruction:
                                scored_for_audit: List[Tuple[str, float, float]] = []
                                instruction_emb = _encode_texts([instruction], cfg, text_ctx)[0] if text_ctx is not None else None
                                for item in scored_candidates:
                                    candidate_row = structured_by_clip.get(str(item["clip_id"]))
                                    if candidate_row is None:
                                        continue
                                    candidate_lookup = lookup_by_clip.get(str(item["clip_id"]), {})
                                    candidate_emb = _load_text_embedding(
                                        str(candidate_lookup.get("text_embedding_path", "") or ""),
                                        text_embedding_cache,
                                    )
                                    encoder_score = _embedding_similarity(instruction_emb, candidate_emb)
                                    lexical_score = _candidate_query_score(instruction, candidate_row)
                                    score = max(float(encoder_score), float(lexical_score))
                                    scored_for_audit.append((str(item["clip_id"]), float(score), float(encoder_score)))
                                scored_for_audit.sort(key=lambda pair: (-float(pair[1]), pair[0]))
                                target_rank = None
                                target_score = 0.0
                                target_encoder_score = 0.0
                                for rank, (clip_id, score, encoder_score) in enumerate(scored_for_audit, start=1):
                                    if clip_id == target_clip_id:
                                        target_rank = rank
                                        target_score = float(score)
                                        target_encoder_score = float(encoder_score)
                                        break
                                flagged = bool(target_rank is not None and target_rank <= solvability_top_k and target_score >= solvability_threshold)
                                if flagged:
                                    counts["solvability_flagged"] += 1
                                top_matches = [
                                    {"clip_id": clip_id, "score": round(float(score), 6), "encoder_score": round(float(encoder_score), 6)}
                                    for clip_id, score, encoder_score in scored_for_audit[:solvability_top_k]
                                ]
                                solvability_audit = {
                                    "instruction": instruction,
                                    "flagged": flagged,
                                    "target_rank": target_rank,
                                    "target_score": round(float(target_score), 6),
                                    "target_encoder_score": round(float(target_encoder_score), 6),
                                    "top_k": solvability_top_k,
                                    "score_threshold": round(float(solvability_threshold), 6),
                                    "top_matches": top_matches,
                                    "baseline": "text_encoder_plus_lexical_overlap",
                                }
                        output = {
                            "chain_id": chain_id,
                            "turn_index": turn_index,
                            "seed_clip_id": seed_clip_id,
                            "source_clip_id": source_clip_id,
                            "target_clip_id": target_clip_id,
                            "semantic_delta_full": semantic_full,
                            "semantic_delta_verbalized": semantic_verbalized,
                            "semantic_delta_full_typed": semantic_full_typed,
                            "semantic_delta_verbalized_typed": semantic_verbalized_typed,
                            "semantic_constraints": semantic_full,
                            "source_node_idx": source_node_idx,
                            "target_node_idx": target_node_idx,
                            "validation": validation if validation else None,
                            "solvability_audit": solvability_audit,
                            "candidate_count": len(scored_candidates),
                            "candidate_frontier_count": len(all_scored_candidates),
                            "pool_summary": pool_summary,
                            "pool_type_summary": pool_type_summary,
                            "candidates": scored_candidates,
                        }
                        out_f.write(json.dumps(output, ensure_ascii=True) + "\n")
                        counts["steps_written"] += 1
                        counts["candidate_rows_written"] += len(scored_candidates)

                        if reason_f is not None:
                            should_label_split = not label_splits or target_split in label_splits
                            if should_label_split:
                                frontier_by_clip = {str(item["clip_id"]): item for item in all_scored_candidates}
                                selected_ids = {str(item["clip_id"]) for item in scored_candidates}
                                for clip_id in clip_ids_by_split.get(target_split, []):
                                    frontier_item = frontier_by_clip.get(str(clip_id))
                                    if frontier_item is not None:
                                        reason_record = {
                                            "chain_id": chain_id,
                                            "turn_index": turn_index,
                                            "split": target_split,
                                            "clip_id": clip_id,
                                            "seed_clip_id": seed_clip_id,
                                            "source_clip_id": source_clip_id,
                                            "target_clip_id": target_clip_id,
                                            "semantic_delta_full": semantic_full,
                                            "semantic_delta_verbalized": semantic_verbalized,
                                            "semantic_delta_full_typed": semantic_full_typed,
                                            "semantic_delta_verbalized_typed": semantic_verbalized_typed,
                                            "reason_code": _reason_code_from_candidate(frontier_item),
                                            "selected_for_pool": clip_id in selected_ids,
                                            "pool_type": frontier_item.get("pool_type"),
                                            "failure_category": frontier_item.get("failure_category"),
                                            "grade": frontier_item.get("grade"),
                                            "label": frontier_item.get("label"),
                                        }
                                    else:
                                        reason_record = {
                                            "chain_id": chain_id,
                                            "turn_index": turn_index,
                                            "split": target_split,
                                            "clip_id": clip_id,
                                            "seed_clip_id": seed_clip_id,
                                            "source_clip_id": source_clip_id,
                                            "target_clip_id": target_clip_id,
                                            "semantic_delta_full": semantic_full,
                                            "semantic_delta_verbalized": semantic_verbalized,
                                            "semantic_delta_full_typed": semantic_full_typed,
                                            "semantic_delta_verbalized_typed": semantic_verbalized_typed,
                                            "reason_code": "failed:too_semantically_far_from_target",
                                            "selected_for_pool": False,
                                            "pool_type": None,
                                            "failure_category": "too_semantically_far_from_target",
                                            "grade": 0,
                                            "label": "miss",
                                        }
                                    reason_f.write(json.dumps(reason_record, ensure_ascii=True) + "\n")
                                    counts["full_dataset_labels_written"] += 1

                        if counts["steps_seen"] % max(1, int(cfg.stage.progress.every_n_rows)) == 0:
                            _log(cfg, f"Relevance pools processed: {counts['steps_seen']:,}")
                        progress.update(1)

                    if stop:
                        break
        finally:
            if reason_f is not None:
                reason_f.close()

    report = {
        "stage": "relevance_pool",
        "input": {
            "input_manifest_csv": str(manifest_path),
            "input_lookup_manifest_csv": str(lookup_manifest_path) if lookup_manifest_path.exists() else None,
            "input_nodes_csv": str(nodes_path),
            "input_edges_csv": str(edges_path),
            "input_chains_jsonl": str(chains_path),
            "input_prepared_jsonl": str(prepared_path),
            "input_validation_jsonl": str(validation_path) if validation_path.exists() else None,
        },
        "counts": counts,
        "config": {
            "behavior": _cfg_section_to_plain(cfg.stage.behavior),
            "models": _cfg_section_to_plain(cfg.stage.models),
            "runtime": _cfg_section_to_plain(cfg.stage.runtime),
            "judge": _cfg_section_to_plain(cfg.stage.judge),
            "pool": _cfg_section_to_plain(cfg.stage.pool),
        },
        "outputs": {
            "output_pools_jsonl": str(out_jsonl),
            "output_reason_labels_jsonl": str(reason_labels_jsonl) if write_full_dataset_labels else None,
            "report": str(report_path),
        },
    }

    tracker.step("Write report", detail=report_path.name)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)

    tracker.finish(f"wrote {counts['steps_written']:,} step pools")
    _log(cfg, f"Relevance pool complete. Wrote {counts['steps_written']:,} step pools")
    return report


def _main_impl(cfg: DictConfig) -> None:
    report = run_relevance_pool(cfg)
    print(json.dumps({"status": "ok", "stage": "relevance_pool", "outputs": report["outputs"]}, indent=2))


def main() -> None:
    import hydra

    @hydra.main(version_base=None, config_path=CONF_DIR, config_name="config")
    def _wrapped(cfg: DictConfig) -> None:
        _main_impl(cfg)

    _wrapped()


if __name__ == "__main__":
    main()
