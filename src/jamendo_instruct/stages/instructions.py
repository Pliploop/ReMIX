from __future__ import annotations

import csv
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple

from jamendo_instruct.progress import StageTracker, rich_tqdm
from jamendo_instruct.semantic_delta import build_typed_semantic_delta

if TYPE_CHECKING:
    from omegaconf import DictConfig
else:
    DictConfig = Any

CONF_DIR = str(Path(__file__).resolve().parents[3] / "conf")

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "up",
    "with",
}


def _log(cfg: DictConfig, message: str) -> None:
    if bool(cfg.stage.progress.enabled):
        print(f"[instructions] {message}", flush=True)


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
            if not line:
                continue
            yield json.loads(line)


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


def _metadata_view(row: Dict[str, str]) -> Dict[str, str]:
    keys = [
        "clip_id",
        "track_id",
        "split",
        "title",
        "artist_id",
        "artist_name",
        "vocals",
        "speed",
        "lyrics_status",
        "lyrics_language",
        "start_time",
        "end_time",
    ]
    return {key: str(row.get(key, "") or "") for key in keys}


def _structured_delta(source_row: Dict[str, str], target_row: Dict[str, str]) -> Dict[str, Any]:
    source_tags = set(_tag_set(source_row))
    target_tags = set(_tag_set(target_row))
    return {
        "tags_added": sorted(target_tags - source_tags),
        "tags_removed": sorted(source_tags - target_tags),
        "tags_preserved": sorted(source_tags & target_tags),
        "source_vocals": str(source_row.get("vocals", "") or ""),
        "target_vocals": str(target_row.get("vocals", "") or ""),
        "source_speed": str(source_row.get("speed", "") or ""),
        "target_speed": str(target_row.get("speed", "") or ""),
        "source_caption": _caption_text(source_row),
        "target_caption": _caption_text(target_row),
        "source_lyrics": _lyrics_text(source_row),
        "target_lyrics": _lyrics_text(target_row),
    }


def _delta_semantic_buckets(delta: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    persistent = [str(x) for x in delta.get("tags_preserved", []) if str(x).strip()]
    new_constraints = [str(x) for x in delta.get("tags_added", []) if str(x).strip()]
    removed_constraints = [str(x) for x in delta.get("tags_removed", []) if str(x).strip()]

    source_vocals = str(delta.get("source_vocals", "") or "").strip()
    target_vocals = str(delta.get("target_vocals", "") or "").strip()
    if source_vocals and source_vocals == target_vocals:
        persistent.append(f"vocals:{target_vocals}")
    elif target_vocals:
        new_constraints.append(f"vocals:{target_vocals}")
        if source_vocals:
            removed_constraints.append(f"vocals:{source_vocals}")

    source_speed = str(delta.get("source_speed", "") or "").strip()
    target_speed = str(delta.get("target_speed", "") or "").strip()
    if source_speed and source_speed == target_speed:
        persistent.append(f"speed:{target_speed}")
    elif target_speed:
        new_constraints.append(f"speed:{target_speed}")
        if source_speed:
            removed_constraints.append(f"speed:{source_speed}")

    return sorted(set(persistent)), sorted(set(new_constraints)), sorted(set(removed_constraints))


def _tokenize_text(text: str, cfg: DictConfig) -> List[str]:
    min_len = max(1, int(cfg.stage.caption.min_token_len))
    return [
        token
        for token in re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", str(text or "").lower())
        if len(token) >= min_len and token not in _STOPWORDS
    ]


def _phrase_candidates(text: str) -> List[str]:
    chunks = re.split(r"[,:;.!?]|(?:\s+-\s+)|(?:\s+and\s+)|(?:\s+with\s+)", str(text or "").lower())
    out: List[str] = []
    for chunk in chunks:
        phrase = re.sub(r"\s+", " ", chunk).strip(" '\"()[]{}")
        if not phrase:
            continue
        if len(phrase.split()) > 8:
            phrase = " ".join(phrase.split()[:8])
        if phrase and phrase not in out:
            out.append(phrase)
    return out


def _limited(values: Sequence[str], limit: int) -> List[str]:
    out: List[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _dedupe_str_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _text_differences_fuzzy(source_text: str, target_text: str, cfg: DictConfig) -> Dict[str, Any]:
    max_terms = max(1, int(cfg.stage.caption.max_terms_per_list))
    max_phrases = max(1, int(cfg.stage.caption.max_phrases_per_list))
    source_tokens = _tokenize_text(source_text, cfg)
    target_tokens = _tokenize_text(target_text, cfg)
    source_set = set(source_tokens)
    target_set = set(target_tokens)

    source_phrases = _phrase_candidates(source_text)
    target_phrases = _phrase_candidates(target_text)
    source_phrase_set = set(source_phrases)
    target_phrase_set = set(target_phrases)

    return {
        "added_terms": _limited([token for token in target_tokens if token not in source_set], max_terms),
        "removed_terms": _limited([token for token in source_tokens if token not in target_set], max_terms),
        "shared_terms": _limited([token for token in target_tokens if token in source_set], max_terms),
        "added_phrases": _limited([phrase for phrase in target_phrases if phrase not in source_phrase_set], max_phrases),
        "removed_phrases": _limited([phrase for phrase in source_phrases if phrase not in target_phrase_set], max_phrases),
        "shared_phrases": _limited([phrase for phrase in target_phrases if phrase in source_phrase_set], max_phrases),
        "same_tags_caption_shift": False,
    }


def _caption_differences_fuzzy(source_caption: str, target_caption: str, cfg: DictConfig) -> Dict[str, Any]:
    return _text_differences_fuzzy(source_caption, target_caption, cfg)


def _caption_differences_raw(source_caption: str, target_caption: str) -> Dict[str, str]:
    return {
        "source_caption": str(source_caption or "").strip(),
        "target_caption": str(target_caption or "").strip(),
    }


def _lyrics_differences_fuzzy(source_lyrics: str, target_lyrics: str, cfg: DictConfig) -> Dict[str, Any]:
    diffs = _text_differences_fuzzy(source_lyrics, target_lyrics, cfg)
    diffs["same_tags_lyric_shift"] = diffs.pop("same_tags_caption_shift", False)
    return diffs


def _lyrics_differences_raw(source_lyrics: str, target_lyrics: str) -> Dict[str, str]:
    return {
        "source_lyrics": str(source_lyrics or "").strip(),
        "target_lyrics": str(target_lyrics or "").strip(),
    }


def _apply_caption_signal_mode(
    *,
    cfg: DictConfig,
    caption_fuzzy_seed: Dict[str, Any],
    caption_fuzzy_previous: Dict[str, Any],
    caption_raw_seed: Dict[str, Any],
    caption_raw_previous: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    mode = str(cfg.stage.caption.signal_mode)
    if mode == "fuzzy":
        return (
            {"from_seed": caption_fuzzy_seed, "from_previous": caption_fuzzy_previous},
            {"from_seed": {}, "from_previous": {}},
        )
    if mode == "raw":
        return (
            {"from_seed": {}, "from_previous": {}},
            {"from_seed": caption_raw_seed, "from_previous": caption_raw_previous},
        )
    return (
        {"from_seed": caption_fuzzy_seed, "from_previous": caption_fuzzy_previous},
        {"from_seed": caption_raw_seed, "from_previous": caption_raw_previous},
    )


def _history_reference_candidates(
    *,
    chain: Dict[str, Any],
    current_turn_index: int,
    current_target_row: Dict[str, str],
    previous_target_row: Dict[str, str],
    structured_by_clip: Dict[str, Dict[str, str]],
    cfg: DictConfig,
) -> List[Dict[str, Any]]:
    current_tags = set(_tag_set(current_target_row))
    previous_tags = set(_tag_set(previous_target_row))
    current_text_terms = set(_tokenize_text(_caption_text(current_target_row), cfg)) | set(_tokenize_text(_lyrics_text(current_target_row), cfg))
    candidates: List[Dict[str, Any]] = []
    steps = list(chain.get("steps", []))

    for prior_turn_idx in range(0, current_turn_index - 1):
        step = steps[prior_turn_idx]
        prior_clip_id = str(step.get("target_clip_id", "") or "")
        prior_row = structured_by_clip.get(prior_clip_id)
        if prior_row is None:
            continue
        prior_tags = set(_tag_set(prior_row))
        prior_text_terms = set(_tokenize_text(_caption_text(prior_row), cfg)) | set(_tokenize_text(_lyrics_text(prior_row), cfg))

        shared_tags = sorted(current_tags & prior_tags)
        reintroduced_tags = sorted((current_tags & prior_tags) - previous_tags)
        shared_text_terms = sorted(current_text_terms & prior_text_terms)

        score = (3 * len(reintroduced_tags)) + (2 * len(shared_tags)) + len(shared_text_terms)
        if score <= 0:
            continue

        reasons: List[str] = []
        if reintroduced_tags:
            reasons.append(f"reintroduces earlier tags: {', '.join(reintroduced_tags[:3])}")
        if shared_tags:
            reasons.append(f"shares target tags with turn {prior_turn_idx + 1}: {', '.join(shared_tags[:3])}")
        if shared_text_terms:
            reasons.append(f"echoes earlier text cues: {', '.join(shared_text_terms[:4])}")

        candidates.append(
            {
                "turn_index": prior_turn_idx + 1,
                "prior_clip_id": prior_clip_id,
                "shared_tags": shared_tags[:6],
                "reintroduced_tags": reintroduced_tags[:6],
                "shared_text_terms": shared_text_terms[:8],
                "why_useful": "; ".join(reasons),
                "score": score,
            }
        )

    candidates.sort(key=lambda item: (-int(item["score"]), int(item["turn_index"])))
    return candidates[:2]


def _choose_verbosity(cfg: DictConfig, rng: random.Random) -> str:
    short_p = max(0.0, float(cfg.stage.generation.short_probability))
    medium_p = max(0.0, float(cfg.stage.generation.medium_probability))
    long_p = max(0.0, float(cfg.stage.generation.long_probability))
    total = short_p + medium_p + long_p
    if total <= 0:
        return "medium"
    pick = rng.random() * total
    if pick < short_p:
        return "short"
    if pick < short_p + medium_p:
        return "medium"
    return "long"


def _build_step_payload(
    *,
    chain: Dict[str, Any],
    step: Dict[str, Any],
    turn_index: int,
    structured_by_clip: Dict[str, Dict[str, str]],
    cfg: DictConfig,
    rng: random.Random,
) -> Dict[str, Any]:
    seed_clip_id = str(chain.get("seed", {}).get("clip_id", "") or "")
    source_clip_id = str(step.get("source_clip_id", "") or "")
    target_clip_id = str(step.get("target_clip_id", "") or "")
    seed_row = structured_by_clip.get(seed_clip_id)
    source_row = structured_by_clip.get(source_clip_id)
    target_row = structured_by_clip.get(target_clip_id)
    if seed_row is None or source_row is None or target_row is None:
        missing = [clip_id for clip_id, row in ((seed_clip_id, seed_row), (source_clip_id, source_row), (target_clip_id, target_row)) if row is None]
        raise KeyError(f"Missing structured rows for clip ids: {missing}")

    delta_from_previous = dict(step.get("structured_delta", {}) or {})
    if not delta_from_previous:
        delta_from_previous = _structured_delta(source_row, target_row)
    delta_from_seed = _structured_delta(seed_row, target_row)

    prev_persistent, prev_new, prev_removed = _delta_semantic_buckets(delta_from_previous)
    seed_persistent, seed_new, seed_removed = _delta_semantic_buckets(delta_from_seed)

    source_caption = str(delta_from_previous.get("source_caption", _caption_text(source_row)) or "")
    target_caption = str(delta_from_previous.get("target_caption", _caption_text(target_row)) or "")
    seed_caption = _caption_text(seed_row)
    source_lyrics = str(delta_from_previous.get("source_lyrics", _lyrics_text(source_row)) or "")
    target_lyrics = str(delta_from_previous.get("target_lyrics", _lyrics_text(target_row)) or "")
    seed_lyrics = _lyrics_text(seed_row)

    caption_fuzzy_previous = _caption_differences_fuzzy(source_caption, target_caption, cfg)
    caption_fuzzy_seed = _caption_differences_fuzzy(seed_caption, target_caption, cfg)
    lyric_fuzzy_previous = _lyrics_differences_fuzzy(source_lyrics, target_lyrics, cfg)
    lyric_fuzzy_seed = _lyrics_differences_fuzzy(seed_lyrics, target_lyrics, cfg)
    same_tags_now = sorted(delta_from_previous.get("tags_added", [])) == [] and sorted(delta_from_previous.get("tags_removed", [])) == []
    caption_fuzzy_previous["same_tags_caption_shift"] = bool(same_tags_now)
    caption_fuzzy_seed["same_tags_caption_shift"] = bool(
        sorted(delta_from_seed.get("tags_added", [])) == [] and sorted(delta_from_seed.get("tags_removed", [])) == []
    )
    lyric_fuzzy_previous["same_tags_lyric_shift"] = bool(same_tags_now)
    lyric_fuzzy_seed["same_tags_lyric_shift"] = bool(
        sorted(delta_from_seed.get("tags_added", [])) == [] and sorted(delta_from_seed.get("tags_removed", [])) == []
    )

    history_candidates = _history_reference_candidates(
        chain=chain,
        current_turn_index=turn_index,
        current_target_row=target_row,
        previous_target_row=source_row,
        structured_by_clip=structured_by_clip,
        cfg=cfg,
    )

    caption_raw_seed = _caption_differences_raw(seed_caption, target_caption)
    caption_raw_previous = _caption_differences_raw(source_caption, target_caption)
    lyric_raw_seed = _lyrics_differences_raw(seed_lyrics, target_lyrics)
    lyric_raw_previous = _lyrics_differences_raw(source_lyrics, target_lyrics)
    active_fuzzy, active_raw = _apply_caption_signal_mode(
        cfg=cfg,
        caption_fuzzy_seed=caption_fuzzy_seed,
        caption_fuzzy_previous=caption_fuzzy_previous,
        caption_raw_seed=caption_raw_seed,
        caption_raw_previous=caption_raw_previous,
    )

    return {
        "chain_id": str(chain.get("chain_id", "") or ""),
        "turn_index": turn_index,
        "seed_clip_id": seed_clip_id,
        "source_clip_id": source_clip_id,
        "target_clip_id": target_clip_id,
        "source_node_idx": int(step.get("source_node_idx", -1) or -1),
        "target_node_idx": int(step.get("target_node_idx", -1) or -1),
        "split": str(step.get("split", "") or target_row.get("split", "") or ""),
        "hardness": str(step.get("hardness", "") or ""),
        "transition_score": float(step.get("transition_score", 0.0) or 0.0),
        "verbosity": _choose_verbosity(cfg, rng),
        "seed_view": {
            "metadata": _metadata_view(seed_row),
            "tags": _tag_set(seed_row),
            "caption": seed_caption,
            "lyrics": seed_lyrics,
        },
        "previous_view": {
            "metadata": _metadata_view(source_row),
            "tags": _tag_set(source_row),
            "caption": source_caption,
            "lyrics": source_lyrics,
        },
        "target_view": {
            "metadata": _metadata_view(target_row),
            "tags": _tag_set(target_row),
            "caption": target_caption,
            "lyrics": target_lyrics,
        },
        "delta_from_seed": delta_from_seed,
        "delta_from_previous": delta_from_previous,
        "persistent_constraints": {
            "from_seed": seed_persistent,
            "from_previous": prev_persistent,
        },
        "new_constraints": {
            "from_seed": seed_new,
            "from_previous": prev_new,
        },
        "removed_constraints": {
            "from_seed": seed_removed,
            "from_previous": prev_removed,
        },
        "caption_signal_mode": str(cfg.stage.caption.signal_mode),
        "caption_differences_fuzzy": active_fuzzy,
        "caption_differences_raw": active_raw,
        "lyric_differences_fuzzy": {
            "from_seed": lyric_fuzzy_seed,
            "from_previous": lyric_fuzzy_previous,
        },
        "lyric_differences_raw": {
            "from_seed": lyric_raw_seed,
            "from_previous": lyric_raw_previous,
        },
        "history_reference_candidates": history_candidates,
        "accumulated_intent_state": dict(step.get("accumulated_intent_state", {}) or {}),
    }


def _prompt_header() -> str:
    return (
        "You are generating colloquial user requests for compositional music retrieval.\n"
        "Return exactly one JSON object and nothing else.\n"
        "Do not wrap the JSON in markdown fences.\n"
        "Do not add explanations, notes, or extra keys."
    )


def _shared_generation_rules() -> str:
    return (
        "Shared rules for both instructions:\n"
        "SEMANTIC DELTA EXTRACTION:\n"
        "1. Fill `semantic_delta_full` exhaustively before writing either instruction.\n"
        "2. `semantic_delta_full.preserved`: list every source quality that should still hold in the target, from tags, vocals, speed, captions, and lyrics when supported.\n"
        "3. `semantic_delta_full.new`: list every new target quality that was absent from the source.\n"
        "4. `semantic_delta_full.lost`: list every source quality that should no longer hold.\n"
        "5. `semantic_delta_full.primary_edit`: one sentence describing the dominant semantic change.\n"
        "6. `semantic_delta_full.caption_only_change`: true only when tags, vocals, and speed do not explain the turn but the captions still shift meaningfully.\n"
        "7. Fill `semantic_delta_verbalized` as the subset of the full semantic delta that both instruction variants are actually allowed to request.\n"
        "8. `semantic_delta_verbalized` must be a subset of `semantic_delta_full`.\n"
        "9. Both instruction variants should verbalize the same requested semantic subset; only the history dependence should differ.\n\n"
        "INSTRUCTION VERBALIZATION:\n"
        "10. Instructions must be faithful to `semantic_delta_full`, but they do NOT need to mention every item.\n"
        "11. Instructions should express the requested content in `semantic_delta_verbalized` and should not silently require extra hidden changes from the full delta.\n"
        "12. Short: verbalize one salient genuine change from `semantic_delta_verbalized.new` or `semantic_delta_verbalized.lost`.\n"
        "13. Medium: verbalize one salient change plus one preservation cue from `semantic_delta_verbalized.preserved` when available.\n"
        "14. Long: verbalize two salient changes or one change with richer context. Stay concise.\n"
        "15. Caption-only turns must surface caption- and/or lyric-derived content such as mood, texture, energy, atmosphere, narrative cues, or emotional register.\n"
        "16. Keep the language colloquial, natural, varied, and concise.\n"
        "17. Write like a real person making a request in everyday language, not like an annotator, rubric, or metadata template.\n"
        "18. Prefer plain spoken phrasing someone would actually type, such as 'make it more dreamy' or 'keep the beat but make it softer'.\n"
        "19. Avoid stiff, overly formal, robotic, or benchmark-style wording.\n"
        "20. Do not sound like you are listing attributes from a schema.\n"
        "21. Contractions, casual phrasing, and natural sentence rhythm are good when they fit.\n"
        "22. The result should read like a believable human request first, while still staying faithful to the payload.\n"
        "23. Do not invent unsupported metadata or contradict preserved constraints.\n"
        "24. Do not mention JSON, metadata fields, or the dataset.\n"
        "25. Prefer short, to-the-point requests over long explanations.\n"
        "26. Rephrase caption wording instead of copying it verbatim whenever possible.\n"
        "27. Use synonyms, reformulations, and natural user language rather than echoing source captions.\n"
        "28. Avoid formulaic rewrite templates such as 'keep X but make it more Y', 'take away X and add Y', or 'make it sound like' when a fresher phrasing would say the same thing.\n"
        "29. Vary sentence openings and verbs across examples; do not default to repeatedly starting with 'keep', 'make', 'take', 'swap', or 'bring back'.\n"
        "30. Prefer the way a person would casually describe the change in one shot, not a mechanical before/after decomposition.\n"
        "31. If the draft sounds like an annotation, checklist, or caption rewrite, silently rewrite it into something more conversational before returning it.\n"
        "32. Good requests often have a little voice or texture, for example: 'lean more into the dreamy side', 'push this toward a softer corporate feel', or 'lose the glitchiness and let it breathe more'.\n"
        "33. Bad requests sound templated or literal, for example: 'remove glitch and add piano', 'keep inspirational and add corporate', or 'change the dreamy Indian fusion to a confusing techno beat'."
    )


def _history_unaware_prompt(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    user_content = (
        f"{_prompt_header()}\n\n"
        "Task:\n"
        "Write the `history_unaware_instruction` as a relative request from the seed item to the current target.\n"
        "It must be understandable without intermediate turns.\n"
        "It should still sound like a natural user request, not a full formal restatement.\n\n"
        f"{_shared_generation_rules()}\n\n"
        "History-unaware specific rules:\n"
        "1. Anchor the edit relative to the seed state.\n"
        "2. Reflect all required changes from seed to target.\n"
        "3. Preserve all seed-to-target constraints that still hold.\n"
        "4. Always use caption and lyric evidence from the payload when available.\n"
        "5. The instruction should be solvable from the seed plus this request alone.\n\n"
        "Output format:\n"
        '{"history_unaware_instruction": "<text>"}\n\n'
        f"Payload:\n{json.dumps(payload, ensure_ascii=True, indent=2)}"
    )
    return [
        {"role": "system", "content": "You follow formatting rules exactly and produce faithful, natural retrieval instructions."},
        {"role": "user", "content": user_content},
    ]


def _history_aware_prompt(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    user_content = (
        f"{_prompt_header()}\n\n"
        "Task:\n"
        "Write the `history_aware_instruction` as a relative request from the seed item to the current target.\n"
        "It may depend on the whole chain and may refer back to earlier turns when helpful.\n"
        "Prefer non-local references when they make the instruction more natural or more faithful, but do not force them if they are unnecessary.\n\n"
        f"{_shared_generation_rules()}\n\n"
        "History-aware specific rules:\n"
        "1. Anchor the request in the chain history, not just the immediately previous step.\n"
        "2. You may reference earlier turns explicitly by turn number or implicitly in natural language.\n"
        "3. If an earlier turn provides a useful preserved vibe, attribute, or caption cue, consider weaving it in.\n"
        "4. Do not collapse into a generic previous-step edit if the payload supports broader history use.\n"
        "5. Still cover the full current-step semantics, not just the referenced callback.\n"
        "6. Always use caption and lyric evidence from the payload when available.\n\n"
        "Output format:\n"
        '{"history_aware_instruction": "<text>"}\n\n'
        f"Payload:\n{json.dumps(payload, ensure_ascii=True, indent=2)}"
    )
    return [
        {"role": "system", "content": "You produce faithful history-aware retrieval instructions and follow JSON output constraints exactly."},
        {"role": "user", "content": user_content},
    ]


def _combined_generation_prompt(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    user_content = (
        f"{_prompt_header()}\n\n"
        "Task:\n"
        "Generate both instruction variants for the same chain step.\n"
        "Return one JSON object with exactly these keys:\n"
        "{\n"
        '  "semantic_delta_full": {\n'
        '    "preserved": ["..."],\n'
        '    "new": ["..."],\n'
        '    "lost": ["..."],\n'
        '    "primary_edit": "...",\n'
        '    "caption_only_change": true\n'
        "  },\n"
        '  "semantic_delta_verbalized": {\n'
        '    "preserved": ["..."],\n'
        '    "new": ["..."],\n'
        '    "lost": ["..."],\n'
        '    "primary_edit": "...",\n'
        '    "caption_only_change": true\n'
        "  },\n"
        '  "history_unaware_instruction": "<text>",\n'
        '  "history_aware_instruction": "<text>"\n'
        "}\n\n"
        f"{_shared_generation_rules()}\n\n"
        "Style target:\n"
        "1. Sound like a real user talking to a retrieval system.\n"
        "2. Be crisp and direct.\n"
        "3. Avoid long multi-clause restatements unless the sampled verbosity requires it.\n"
        "4. When the captions mention a concept directly, try to paraphrase it rather than repeat the exact phrase.\n"
        "5. Prefer compact, natural edits over verbose explanations, but avoid falling into canned sentence templates.\n"
        "6. Use colloquial, everyday phrasing a real human would naturally say.\n"
        "7. Avoid sounding like a taxonomy, checklist, or metadata summary.\n"
        "8. If there is a choice, prefer the more human-sounding phrasing over the more literal caption wording.\n"
        "9. Do not reuse the same sentence skeleton across examples; aim for phrasing variety.\n"
        "10. Favor requests that feel spoken or typed by a listener, not generated from attribute diffs.\n"
        "11. A slightly idiomatic, conversational line is better than a perfectly literal restatement.\n\n"
        "History-unaware rules:\n"
        "1. Anchor the edit relative to the seed state.\n"
        "2. It must be understandable without intermediate turns.\n"
        "3. It must remain faithful to the step semantics even if it verbalizes only a subset.\n"
        "4. Always use caption and lyric evidence from the payload when the semantics depend on it.\n\n"
        "History-aware rules:\n"
        "1. Also anchor the request relative to the seed, but allow dependence on the whole chain.\n"
        "2. Prefer non-local history references when they make the instruction more natural or more faithful.\n"
        "3. You may reference earlier turns explicitly by turn number or implicitly in natural language.\n"
        "4. Do not force a non-local callback if the current turn already naturally carries the needed meaning.\n"
        "5. Still reflect the current step semantics even if you reference earlier turns.\n"
        "6. Always use caption and lyric evidence from the payload when the semantics depend on it.\n\n"
        "Formatting guardrails:\n"
        "1. Output JSON only.\n"
        "2. No markdown fences.\n"
        "3. No extra keys.\n"
        "4. `semantic_delta_full` and `semantic_delta_verbalized` must come before the instruction fields.\n"
        "5. Each instruction value must be a single string.\n"
        "6. Do not leave any required field empty.\n\n"
        "Length guidance by verbosity:\n"
        "1. short: usually 1 short sentence.\n"
        "2. medium: usually 1 sentence, occasionally 2 compact clauses.\n"
        "3. long: still concise, but can include one extra preservation or refinement clause.\n\n"
        f"Payload:\n{json.dumps(payload, ensure_ascii=True, indent=2)}"
    )
    return [
        {
            "role": "system",
            "content": "You generate faithful, diverse retrieval instructions that sound like real human requests, and you obey strict JSON formatting.",
        },
        {"role": "user", "content": user_content},
    ]


def _verification_prompt(payload: Dict[str, Any], generated: Dict[str, str]) -> List[Dict[str, str]]:
    user_content = (
        f"{_prompt_header()}\n\n"
        "Task:\n"
        "Verify the generated instructions against the payload and return strict machine-readable JSON.\n"
        "Judge each variant separately and be strict.\n\n"
        "Output format:\n"
        '{"history_unaware": "pass", "history_aware": "pass", "history_unaware_reason": "...", "history_aware_reason": "..."}\n\n'
        "Validation rules:\n"
        "1. The instruction must be relative.\n"
        "2. PASS if the instruction reflects at least one genuine change from `semantic_delta_verbalized.new` or `semantic_delta_verbalized.lost`.\n"
        '   FAIL label: "failed:no_genuine_change"\n'
        "3. PASS if it does not contradict any preserved constraint from `semantic_delta_full`.\n"
        '   FAIL label: "failed:contradiction"\n'
        "4. PASS if it does not invent unsupported metadata.\n"
        '   FAIL label: "failed:metadata_invention"\n'
        "5. For history-unaware: PASS if it is understandable from seed plus current request.\n"
        '   FAIL label: "failed:requires_history"\n'
        "6. For history-aware: PASS if it remains coherent with the broader chain context.\n"
        '   FAIL label: "failed:history_incoherent"\n'
        "7. For caption-only turns: FAIL if the instruction uses only generic tag language with no caption-derived content.\n"
        '   FAIL label: "failed:caption_only_verbalization_missing"\n'
        f"Payload:\n{json.dumps(payload, ensure_ascii=True, indent=2)}\n\n"
        f"Generated:\n{json.dumps(generated, ensure_ascii=True, indent=2)}"
    )
    return [
        {"role": "system", "content": "You are a strict verifier and output JSON only."},
        {"role": "user", "content": user_content},
    ]


def _normalize_semantic_delta(value: Any, *, field_name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    preserved = _dedupe_str_list(value.get("preserved", []))
    new = _dedupe_str_list(value.get("new", []))
    lost = _dedupe_str_list(value.get("lost", []))
    primary_edit = str(value.get("primary_edit", "") or "").strip()
    caption_only_change = value.get("caption_only_change")
    if not primary_edit:
        raise ValueError(f"{field_name}.primary_edit is required")
    if not isinstance(caption_only_change, bool):
        raise ValueError(f"{field_name}.caption_only_change must be a boolean")
    return {
        "preserved": preserved,
        "new": new,
        "lost": lost,
        "primary_edit": primary_edit,
        "caption_only_change": caption_only_change,
    }


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


def _build_generator(cfg: DictConfig) -> Any:
    backend = str(getattr(cfg.stage.runtime, "backend", "transformers"))
    if backend == "vllm_local":
        return _build_vllm_local_generator(cfg)
    if backend != "transformers":
        raise ValueError(f"Unsupported stage.runtime.backend: {backend}")

    from transformers import AutoModelForCausalLM, AutoProcessor

    runtime = _resolve_torch_device(cfg)
    torch = runtime.torch
    device = runtime.device
    token_env = str(getattr(cfg.stage.auth, "hf_token_env", "HF_TOKEN"))
    token = os.environ.get(token_env, "").strip() or None
    if token is None:
        _log(cfg, f"No Hugging Face token found in ${token_env}; gated model downloads may fail.")
    model_id = str(cfg.stage.models.model_id)
    dtype = _resolve_torch_dtype(cfg, torch)
    _log(cfg, f"Loading instruction model {model_id} on {device}")
    processor = AutoProcessor.from_pretrained(model_id, token=token)
    model_kwargs: Dict[str, Any] = {"token": token, "torch_dtype": dtype}
    if str(device).startswith("cuda"):
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if not str(device).startswith("cuda"):
        model = model.to(device)
    model.eval()
    return SimpleNamespace(model=model, processor=processor, torch=torch, device=device, backend=backend)


def _build_vllm_local_generator(cfg: DictConfig) -> Any:
    try:
        import httpx
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "stage.runtime.backend=vllm_local requires the `httpx` package in the active environment."
        ) from exc
    try:
        import vllm  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "stage.runtime.backend=vllm_local requires the `vllm` package in the active environment."
        ) from exc

    model_id = str(cfg.stage.models.model_id)
    host = str(getattr(cfg.stage.runtime, "vllm_host", "127.0.0.1"))
    port = int(getattr(cfg.stage.runtime, "vllm_port", 8000))
    api_key = str(getattr(cfg.stage.runtime, "vllm_api_key", "EMPTY"))
    tensor_parallel_size = int(getattr(cfg.stage.runtime, "vllm_tensor_parallel_size", 1))
    quantization = getattr(cfg.stage.runtime, "vllm_quantization", None)
    trust_remote_code = bool(getattr(cfg.stage.runtime, "vllm_trust_remote_code", False))
    gpu_mem_util = float(getattr(cfg.stage.runtime, "vllm_gpu_memory_utilization", 0.9))
    max_model_len = int(getattr(cfg.stage.runtime, "vllm_max_model_len", 32768))
    health_timeout_sec = int(getattr(cfg.stage.runtime, "vllm_health_timeout_sec", 300))
    base_url = f"http://{host}:{port}"
    env = os.environ.copy()
    log_dir = Path(str(cfg.stage.io.output_dir))
    log_dir.mkdir(parents=True, exist_ok=True)
    server_log = log_dir / "vllm_server.log"
    client = httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0))

    def _suggest_retry_model_len(log_path: Path, requested_len: int) -> Optional[int]:
        try:
            text = log_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return None
        match = re.search(r"estimated maximum model length is (\d+)", text)
        if not match:
            return None
        estimated_len = int(match.group(1))
        # Leave some headroom because the estimate can be optimistic across runs.
        safe_len = max(1024, (estimated_len * 9) // 10)
        # Round down to a multiple of 256 to keep the retry predictable.
        safe_len -= safe_len % 256
        if safe_len <= 0 or safe_len >= requested_len:
            return None
        return safe_len

    attempted_model_len = max_model_len
    proc = None
    log_handle = None
    healthy = False
    for attempt_index in range(2):
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_id,
            "--host",
            host,
            "--port",
            str(port),
            "--tensor-parallel-size",
            str(tensor_parallel_size),
            "--gpu-memory-utilization",
            str(gpu_mem_util),
            "--max-model-len",
            str(attempted_model_len),
        ]
        if quantization is not None and str(quantization).strip():
            cmd.extend(["--quantization", str(quantization).strip()])
        if trust_remote_code:
            cmd.append("--trust-remote-code")
        _log(
            cfg,
            f"Starting local vLLM server for {model_id} at {base_url} "
            f"(max_model_len={attempted_model_len})",
        )
        log_handle = server_log.open("a", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
        deadline = time.time() + health_timeout_sec
        while time.time() < deadline:
            if proc.poll() is not None:
                log_handle.flush()
                retry_model_len = _suggest_retry_model_len(server_log, attempted_model_len)
                if attempt_index == 0 and retry_model_len is not None:
                    _log(
                        cfg,
                        "vLLM startup failed because the requested max_model_len exceeded "
                        f"available KV cache memory; retrying with max_model_len={retry_model_len}.",
                    )
                    try:
                        log_handle.close()
                    except Exception:
                        pass
                    attempted_model_len = retry_model_len
                    break
                raise RuntimeError(
                    f"vLLM server exited early with code {proc.returncode}. Check {server_log}"
                )
            try:
                resp = client.get(f"{base_url}/health")
                if resp.status_code == 200:
                    healthy = True
                    break
            except Exception:
                pass
            time.sleep(2.0)
        if healthy:
            break
        if proc.poll() is None and time.time() >= deadline:
            proc.terminate()
            raise TimeoutError(f"Timed out waiting for vLLM health endpoint at {base_url}/health")
    if not healthy:
        raise RuntimeError(f"vLLM server did not become healthy. Check {server_log}")
    _log(cfg, f"vLLM server is healthy at {base_url}")

    def _close() -> None:
        try:
            client.close()
        except Exception:
            pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except Exception:
                proc.kill()
        try:
            log_handle.close()
        except Exception:
            pass

    return SimpleNamespace(
        backend="vllm_local",
        base_url=base_url,
        api_key=api_key,
        model_id=model_id,
        client=client,
        process=proc,
        close=_close,
        server_log=str(server_log),
    )


def _decode_response_text(ctx: Any, messages: List[Dict[str, str]], cfg: DictConfig, *, max_new_tokens: int, temperature: float, top_p: float) -> str:
    if str(getattr(ctx, "backend", "transformers")) == "vllm_local":
        headers = {"Authorization": f"Bearer {ctx.api_key}"}
        payload = {
            "model": ctx.model_id,
            "messages": messages,
            "max_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
        }
        resp = ctx.client.post(f"{ctx.base_url}/v1/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("vLLM response did not include choices")
        message = choices[0].get("message", {})
        text = str(message.get("content", "") or "").strip()
        return text

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
        "max_new_tokens": int(max_new_tokens),
        "do_sample": float(temperature) > 0.0,
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    text = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    return str(text or "").strip()


def _decode_response_text_batch(
    ctx: Any,
    messages_batch: List[List[Dict[str, str]]],
    cfg: DictConfig,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    if str(getattr(ctx, "backend", "transformers")) == "vllm_local":
        return [
            _decode_response_text(
                ctx,
                messages,
                cfg,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            for messages in messages_batch
        ]

    processor = ctx.processor
    model = ctx.model
    torch = ctx.torch

    chat_texts = [
        processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(getattr(cfg.stage.runtime, "enable_thinking", False)),
        )
        for messages in messages_batch
    ]
    inputs = processor(text=chat_texts, return_tensors="pt", padding=True)
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        input_lens = [int(inputs["input_ids"].shape[-1])] * int(inputs["input_ids"].shape[0])
    else:
        input_lens = [int(x) for x in attention_mask.sum(dim=-1).tolist()]
    gen_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": float(temperature) > 0.0,
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    texts = []
    for row_idx, input_len in enumerate(input_lens):
        text = processor.decode(outputs[row_idx][input_len:], skip_special_tokens=True)
        texts.append(str(text or "").strip())
    return texts


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
        snippet = cleaned[start : end + 1]
        data = json.loads(snippet)
        if isinstance(data, dict):
            return data
    raise json.JSONDecodeError("Unable to parse JSON object from model output", cleaned, 0)


def _generate_instruction_pair(ctx: Any, cfg: DictConfig, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    messages = _combined_generation_prompt(payload)
    retries = max(1, int(cfg.stage.behavior.strict_json_retry_attempts))
    last_error = ""
    for attempt in range(retries):
        if attempt == 0 and int(payload.get("turn_index", 0)) <= 3:
            _log(
                cfg,
                f"Starting generation for chain={payload.get('chain_id')} turn={payload.get('turn_index')} "
                f"(verbosity={payload.get('verbosity')}, max_new_tokens={int(cfg.stage.generation.max_new_tokens)})",
            )
        elif attempt > 0:
            _log(
                cfg,
                f"Retrying generation for chain={payload.get('chain_id')} turn={payload.get('turn_index')} "
                f"(attempt={attempt + 1}/{retries})",
            )
        raw = _decode_response_text(
            ctx,
            messages,
            cfg,
            max_new_tokens=int(cfg.stage.generation.max_new_tokens),
            temperature=float(cfg.stage.generation.temperature),
            top_p=float(cfg.stage.generation.top_p),
        )
        try:
            parsed = _extract_json_object(raw)
            semantic_delta_full = parsed.get("semantic_delta_full")
            semantic_delta_verbalized = parsed.get("semantic_delta_verbalized")
            if semantic_delta_full is None and parsed.get("semantic_constraints") is not None:
                semantic_delta_full = parsed.get("semantic_constraints")
            if semantic_delta_verbalized is None and semantic_delta_full is not None:
                semantic_delta_verbalized = semantic_delta_full
            full = _normalize_semantic_delta(semantic_delta_full, field_name="semantic_delta_full")
            verbalized = _normalize_semantic_delta(semantic_delta_verbalized, field_name="semantic_delta_verbalized")
            full_typed = build_typed_semantic_delta(payload, full)
            verbalized_typed = build_typed_semantic_delta(payload, verbalized)
            generated = {
                "semantic_delta_full": full,
                "semantic_delta_verbalized": verbalized,
                "semantic_delta_full_typed": full_typed,
                "semantic_delta_verbalized_typed": verbalized_typed,
                "semantic_constraints": full,
                "history_unaware_instruction": str(parsed.get("history_unaware_instruction", "") or "").strip(),
                "history_aware_instruction": str(parsed.get("history_aware_instruction", "") or "").strip(),
            }
            errors = {
                key.replace("_instruction", ""): "missing_instruction"
                for key, value in generated.items()
                if not value
            }
            if errors:
                raise ValueError(f"Missing required instructions: {sorted(errors)}")
            if int(payload.get("turn_index", 0)) <= 3:
                _log(
                    cfg,
                    f"Generation succeeded for chain={payload.get('chain_id')} turn={payload.get('turn_index')} "
                    f"on attempt {attempt + 1}/{retries}",
                )
            return generated, {}
        except Exception as exc:
            last_error = f"attempt={attempt + 1}: {exc}"
            _log(
                cfg,
                f"Generation parse/validation failed for chain={payload.get('chain_id')} "
                f"turn={payload.get('turn_index')} on attempt {attempt + 1}/{retries}: {exc.__class__.__name__}",
            )
    return {}, {"history_unaware": last_error or "unknown_generation_error", "history_aware": last_error or "unknown_generation_error"}


def _parse_generated_instruction_payload(payload: Dict[str, Any], parsed: Dict[str, Any]) -> Dict[str, Any]:
    semantic_delta_full = parsed.get("semantic_delta_full")
    semantic_delta_verbalized = parsed.get("semantic_delta_verbalized")
    if semantic_delta_full is None and parsed.get("semantic_constraints") is not None:
        semantic_delta_full = parsed.get("semantic_constraints")
    if semantic_delta_verbalized is None and semantic_delta_full is not None:
        semantic_delta_verbalized = semantic_delta_full
    full = _normalize_semantic_delta(semantic_delta_full, field_name="semantic_delta_full")
    verbalized = _normalize_semantic_delta(semantic_delta_verbalized, field_name="semantic_delta_verbalized")
    full_typed = build_typed_semantic_delta(payload, full)
    verbalized_typed = build_typed_semantic_delta(payload, verbalized)
    generated = {
        "semantic_delta_full": full,
        "semantic_delta_verbalized": verbalized,
        "semantic_delta_full_typed": full_typed,
        "semantic_delta_verbalized_typed": verbalized_typed,
        "semantic_constraints": full,
        "history_unaware_instruction": str(parsed.get("history_unaware_instruction", "") or "").strip(),
        "history_aware_instruction": str(parsed.get("history_aware_instruction", "") or "").strip(),
    }
    errors = {
        key.replace("_instruction", ""): "missing_instruction"
        for key, value in generated.items()
        if not value
    }
    if errors:
        raise ValueError(f"Missing required instructions: {sorted(errors)}")
    return generated


def _generate_instruction_pairs_batch(
    ctx: Any,
    cfg: DictConfig,
    payloads: List[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], Dict[str, str]]]:
    if not payloads:
        return []
    max_new_tokens = int(cfg.stage.generation.max_new_tokens)
    temperature = float(cfg.stage.generation.temperature)
    top_p = float(cfg.stage.generation.top_p)
    retries = max(1, int(cfg.stage.behavior.strict_json_retry_attempts))
    messages_batch = [_combined_generation_prompt(payload) for payload in payloads]
    for payload in payloads[:3]:
        if int(payload.get("turn_index", 0)) <= 3:
            _log(
                cfg,
                f"Starting generation for chain={payload.get('chain_id')} turn={payload.get('turn_index')} "
                f"(verbosity={payload.get('verbosity')}, max_new_tokens={max_new_tokens})",
            )
    _log(cfg, f"Running batched generation for {len(payloads):,} instruction step(s)")
    raw_texts = _decode_response_text_batch(
        ctx,
        messages_batch,
        cfg,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    results: List[Tuple[Dict[str, Any], Dict[str, str]]] = []
    for payload, raw in zip(payloads, raw_texts):
        try:
            parsed = _extract_json_object(raw)
            generated = _parse_generated_instruction_payload(payload, parsed)
            if int(payload.get("turn_index", 0)) <= 3:
                _log(
                    cfg,
                    f"Generation succeeded for chain={payload.get('chain_id')} turn={payload.get('turn_index')} "
                    f"on attempt 1/{retries}",
                )
            results.append((generated, {}))
        except Exception as exc:
            _log(
                cfg,
                f"Generation parse/validation failed for chain={payload.get('chain_id')} "
                f"turn={payload.get('turn_index')} on attempt 1/{retries}: {exc.__class__.__name__}",
            )
            if retries > 1:
                results.append(_generate_instruction_pair(ctx, cfg, payload))
            else:
                last_error = f"attempt=1: {exc}"
                results.append(({}, {"history_unaware": last_error, "history_aware": last_error}))
    return results


def _run_verifier(ctx: Any, cfg: DictConfig, payload: Dict[str, Any], generated: Dict[str, str]) -> Dict[str, Any]:
    messages = _verification_prompt(payload, generated)
    raw = _decode_response_text(
        ctx,
        messages,
        cfg,
        max_new_tokens=int(cfg.stage.verification.max_new_tokens),
        temperature=float(cfg.stage.verification.temperature),
        top_p=float(cfg.stage.verification.top_p),
    )
    return _extract_json_object(raw)


def _passes_verifier_result(data: Dict[str, Any], key: str) -> Tuple[bool, List[str]]:
    section = data.get(key)
    if isinstance(section, dict):
        passed = bool(section.get("passed", False))
        reasons_raw = section.get("reasons", [])
        if isinstance(reasons_raw, list):
            reasons = [str(x) for x in reasons_raw if str(x).strip()]
        else:
            reasons = [str(reasons_raw)] if str(reasons_raw).strip() else []
        return passed, reasons
    if isinstance(section, str):
        verdict = str(section).strip().lower()
        reason_key = f"{key}_reason"
        reason = str(data.get(reason_key, "") or "").strip()
        if verdict == "pass":
            return True, []
        if verdict == "fail":
            return False, [reason or "unspecified"]
    return False, ["missing_verifier_section"]


def run_instructions(cfg: DictConfig) -> Dict[str, object]:
    structured_path = Path(str(cfg.stage.io.input_manifest_csv))
    chains_path = Path(str(cfg.stage.io.input_chains_jsonl))
    out_dir = Path(str(cfg.stage.io.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / str(cfg.stage.io.output_instructions_jsonl)
    prepared_jsonl = out_dir / str(cfg.stage.io.output_prepared_jsonl)
    report_path = out_dir / str(cfg.stage.io.report_file)
    tracker = StageTracker(
        cfg,
        "instructions",
        title="Generate Instruction Steps",
        subtitle=f"chains={chains_path}",
        total_steps=4,
    )

    for path in (structured_path, chains_path):
        if not path.exists():
            raise FileNotFoundError(f"Required instructions input not found: {path}")

    tracker.step("Load structured clips and sampled chains", detail=f"manifest={structured_path.name}")
    structured_by_clip = _structured_index(structured_path)
    chains = list(_read_jsonl(chains_path))
    max_chains = cfg.stage.behavior.max_chains
    if max_chains is not None:
        chains = chains[: max(0, int(max_chains))]

    rng = random.Random(int(cfg.stage.behavior.random_seed))
    model_load_start = time.perf_counter()
    tracker.step("Load instruction model", detail=str(cfg.stage.models.model_id))
    generator_ctx = None
    try:
        generator_ctx = _build_generator(cfg)
        model_load_elapsed = time.perf_counter() - model_load_start
        verifier_ctx = generator_ctx if bool(cfg.stage.verification.enabled) else None
        max_steps = cfg.stage.behavior.max_steps
        discard_failed = bool(cfg.stage.behavior.discard_failed)
        write_prepared = bool(cfg.stage.behavior.write_prepared_records)
        every_n = max(1, int(cfg.stage.progress.every_n_rows))
        generation_batch_size = max(1, int(getattr(cfg.stage.runtime, "generation_batch_size", 1)))

        counts: Dict[str, Any] = {
            "chains_seen": len(chains),
            "steps_seen": 0,
            "steps_attempted": 0,
            "steps_written": 0,
            "discarded_steps": 0,
            "parser_failures": 0,
            "verifier_failures": 0,
            "generation_failures_by_variant": {"history_unaware": 0, "history_aware": 0},
            "verifier_failures_by_variant": {"history_unaware": 0, "history_aware": 0},
            "accepted_by_verbosity": {"short": 0, "medium": 0, "long": 0},
            "discard_reasons": {},
            "timing_sec": {
                "model_load": round(model_load_elapsed, 6),
                "generation_total": 0.0,
                "verification_total": 0.0,
                "per_step_mean_generation": 0.0,
            },
        }
        verifier_failure_examples: List[Dict[str, Any]] = []
        first_success_logged = False

        def _process_generation_batch(batch_payloads: List[Dict[str, Any]], *, out_f: Any, prepared_f: Any, progress: Any) -> None:
            nonlocal first_success_logged
            if not batch_payloads:
                return
            for payload in batch_payloads:
                if counts["steps_attempted"] <= 3:
                    _log(
                        cfg,
                        f"Prepared payload for chain={payload['chain_id']} turn={payload['turn_index']} "
                        f"(source={payload['source_clip_id']}, target={payload['target_clip_id']}, "
                        f"verbosity={payload['verbosity']})",
                    )
                if prepared_f is not None:
                    prepared_f.write(json.dumps(payload, ensure_ascii=True) + "\n")
                    prepared_f.flush()

            gen_start = time.perf_counter()
            batch_results = _generate_instruction_pairs_batch(generator_ctx, cfg, batch_payloads)
            gen_elapsed_total = time.perf_counter() - gen_start

            for payload, (generated, generation_errors) in zip(batch_payloads, batch_results):
                counts["timing_sec"]["generation_total"] += gen_elapsed_total / max(1, len(batch_payloads))
                for variant in generation_errors:
                    counts["parser_failures"] += 1
                    counts["generation_failures_by_variant"][variant] += 1

                if generation_errors:
                    counts["discarded_steps"] += 1
                    for variant in generation_errors:
                        reason = f"generation_failed:{variant}"
                        counts["discard_reasons"][reason] = counts["discard_reasons"].get(reason, 0) + 1
                    if not discard_failed:
                        record = {
                            "chain_id": payload["chain_id"],
                            "turn_index": payload["turn_index"],
                            "seed_clip_id": payload["seed_clip_id"],
                            "target_clip_id": payload["target_clip_id"],
                            "verbosity": payload["verbosity"],
                            "semantic_delta_full": generated.get("semantic_delta_full"),
                            "semantic_delta_verbalized": generated.get("semantic_delta_verbalized"),
                            "semantic_delta_full_typed": generated.get("semantic_delta_full_typed"),
                            "semantic_delta_verbalized_typed": generated.get("semantic_delta_verbalized_typed"),
                            "semantic_constraints": generated.get("semantic_constraints"),
                            "history_unaware_instruction": generated.get("history_unaware_instruction", ""),
                            "history_aware_instruction": generated.get("history_aware_instruction", ""),
                            "status": "failed_generation",
                            "errors": generation_errors,
                        }
                        out_f.write(json.dumps(record, ensure_ascii=True) + "\n")
                        out_f.flush()
                    progress.update(1)
                    continue

                if verifier_ctx is not None:
                    try:
                        verify_start = time.perf_counter()
                        verdict = _run_verifier(verifier_ctx, cfg, payload, generated)
                        counts["timing_sec"]["verification_total"] += time.perf_counter() - verify_start
                    except Exception as exc:
                        counts["verifier_failures"] += 1
                        counts["discarded_steps"] += 1
                        reason = f"verifier_error:{exc.__class__.__name__}"
                        counts["discard_reasons"][reason] = counts["discard_reasons"].get(reason, 0) + 1
                        progress.update(1)
                        continue

                    step_failed = False
                    for variant in ("history_unaware", "history_aware"):
                        passed, reasons = _passes_verifier_result(verdict, variant)
                        if not passed:
                            step_failed = True
                            counts["verifier_failures"] += 1
                            counts["verifier_failures_by_variant"][variant] += 1
                            for reason in reasons or ["unspecified"]:
                                key = f"verifier_failed:{variant}:{reason}"
                                counts["discard_reasons"][key] = counts["discard_reasons"].get(key, 0) + 1
                            verifier_failure_examples.append(
                                {
                                    "chain_id": payload["chain_id"],
                                    "turn_index": payload["turn_index"],
                                    "variant": variant,
                                    "reasons": reasons,
                                }
                            )
                    if step_failed:
                        counts["discarded_steps"] += 1
                        progress.update(1)
                        continue

                record = {
                    "chain_id": payload["chain_id"],
                    "turn_index": payload["turn_index"],
                    "seed_clip_id": payload["seed_clip_id"],
                    "source_clip_id": payload["source_clip_id"],
                    "target_clip_id": payload["target_clip_id"],
                    "source_node_idx": payload["source_node_idx"],
                    "target_node_idx": payload["target_node_idx"],
                    "split": payload["split"],
                    "hardness": payload["hardness"],
                    "transition_score": payload["transition_score"],
                    "verbosity": payload["verbosity"],
                    "semantic_delta_full": generated["semantic_delta_full"],
                    "semantic_delta_verbalized": generated["semantic_delta_verbalized"],
                    "semantic_delta_full_typed": generated["semantic_delta_full_typed"],
                    "semantic_delta_verbalized_typed": generated["semantic_delta_verbalized_typed"],
                    "semantic_constraints": generated["semantic_constraints"],
                    "history_unaware_instruction": generated["history_unaware_instruction"],
                    "history_aware_instruction": generated["history_aware_instruction"],
                    "model_id": str(cfg.stage.models.model_id),
                    "caption_signal_mode": str(cfg.stage.caption.signal_mode),
                    "prompt_version": "v4_humanized_semantic_deltas",
                    "status": "ok",
                }
                out_f.write(json.dumps(record, ensure_ascii=True) + "\n")
                out_f.flush()
                counts["steps_written"] += 1
                counts["accepted_by_verbosity"][payload["verbosity"]] += 1
                if not first_success_logged:
                    _log(
                        cfg,
                        f"First instruction record written for chain={payload['chain_id']} "
                        f"turn={payload['turn_index']} after {gen_elapsed_total / max(1, len(batch_payloads)):.2f}s average batch generation",
                    )
                    first_success_logged = True
                if counts["steps_seen"] % every_n == 0:
                    mean_gen = counts["timing_sec"]["generation_total"] / max(1, counts["steps_attempted"])
                    _log(cfg, f"Instruction steps processed: {counts['steps_seen']:,} (mean generation {mean_gen:.2f}s/step)")
                    progress.update(1)
        stage_start = time.perf_counter()
        tracker.step("Generate instruction records", detail=f"chains={len(chains):,}, max_steps={max_steps if max_steps is not None else 'all'}")
        with out_jsonl.open("w", encoding="utf-8") as out_f:
            prepared_f = prepared_jsonl.open("w", encoding="utf-8") if write_prepared else None
            try:
                total_steps_hint = None
                if max_steps is not None:
                    total_steps_hint = int(max_steps)
                with rich_tqdm(cfg, total=total_steps_hint, desc="Instruction steps", unit="step") as progress:
                    stop = False
                    batch_payloads: List[Dict[str, Any]] = []
                    for chain in chains:
                        steps = list(chain.get("steps", []))
                        for idx, step in enumerate(steps, start=1):
                            if max_steps is not None and counts["steps_seen"] >= int(max_steps):
                                stop = True
                                break
                            counts["steps_seen"] += 1
                            counts["steps_attempted"] += 1
                            try:
                                payload = _build_step_payload(
                                    chain=chain,
                                    step=step,
                                    turn_index=idx,
                                    structured_by_clip=structured_by_clip,
                                    cfg=cfg,
                                    rng=rng,
                                )
                            except Exception as exc:
                                counts["discarded_steps"] += 1
                                reason = f"payload_error:{exc.__class__.__name__}"
                                counts["discard_reasons"][reason] = counts["discard_reasons"].get(reason, 0) + 1
                                progress.update(1)
                                continue
                            batch_payloads.append(payload)
                            if len(batch_payloads) >= generation_batch_size:
                                _process_generation_batch(batch_payloads, out_f=out_f, prepared_f=prepared_f, progress=progress)
                                batch_payloads = []

                        if stop:
                            break
                    if batch_payloads:
                        _process_generation_batch(batch_payloads, out_f=out_f, prepared_f=prepared_f, progress=progress)
            finally:
                if prepared_f is not None:
                    prepared_f.close()

        counts["timing_sec"]["total_stage_wall"] = round(time.perf_counter() - stage_start, 6)
        counts["timing_sec"]["generation_total"] = round(float(counts["timing_sec"]["generation_total"]), 6)
        counts["timing_sec"]["verification_total"] = round(float(counts["timing_sec"]["verification_total"]), 6)
        counts["timing_sec"]["per_step_mean_generation"] = round(
            float(counts["timing_sec"]["generation_total"]) / max(1, int(counts["steps_attempted"])),
            6,
        )

        report = {
            "stage": "instructions",
            "input": {
                "input_manifest_csv": str(structured_path),
                "input_chains_jsonl": str(chains_path),
            },
            "counts": counts,
            "config": {
                "runtime": _cfg_section_to_plain(cfg.stage.runtime),
                "behavior": _cfg_section_to_plain(cfg.stage.behavior),
                "caption": _cfg_section_to_plain(cfg.stage.caption),
                "generation": _cfg_section_to_plain(cfg.stage.generation),
                "verification": _cfg_section_to_plain(cfg.stage.verification),
            },
            "outputs": {
                "output_instructions_jsonl": str(out_jsonl),
                "output_prepared_jsonl": str(prepared_jsonl) if write_prepared else None,
                "report": str(report_path),
            },
            "verifier_failure_examples": verifier_failure_examples[:25],
        }

        tracker.step("Write report", detail=report_path.name)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=True)

        tracker.finish(
            f"wrote {counts['steps_written']:,}/{counts['steps_attempted']:,} step records"
        )
        _log(
            cfg,
            f"Instruction generation complete. Wrote {counts['steps_written']:,} / {counts['steps_attempted']:,} step records "
            f"in {counts['timing_sec']['total_stage_wall']:.2f}s",
        )
        return report
    finally:
        if generator_ctx is not None and callable(getattr(generator_ctx, "close", None)):
            _log(cfg, "Shutting down instruction generation backend")
            try:
                generator_ctx.close()
            except Exception as exc:
                _log(cfg, f"Backend shutdown raised {exc.__class__.__name__}: {exc}")


def _main_impl(cfg: DictConfig) -> None:
    report = run_instructions(cfg)
    print(json.dumps({"status": "ok", "stage": "instructions", "outputs": report["outputs"]}, indent=2))


def main() -> None:
    import hydra

    @hydra.main(version_base=None, config_path=CONF_DIR, config_name="config")
    def _wrapped(cfg: DictConfig) -> None:
        _main_impl(cfg)

    _wrapped()


if __name__ == "__main__":
    main()
