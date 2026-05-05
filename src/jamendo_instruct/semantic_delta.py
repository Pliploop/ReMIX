from __future__ import annotations

from typing import Any, Dict, List, Sequence


_KIND_HINTS = {
    "mood": {
        "melancholic",
        "sad",
        "happy",
        "triumphant",
        "optimistic",
        "haunting",
        "dark",
        "uplifting",
        "introspective",
        "tense",
        "dramatic",
        "calm",
        "chill",
        "relaxed",
        "aggressive",
        "playful",
        "romantic",
        "nostalgic",
        "brooding",
        "emotional",
    },
    "texture": {
        "texture",
        "reverb",
        "ambient",
        "atmospheric",
        "sparse",
        "dense",
        "lush",
        "gritty",
        "warm",
        "cold",
        "distorted",
        "acoustic",
        "electronic",
        "layered",
        "minimal",
        "cinematic",
    },
    "energy": {
        "energetic",
        "energy",
        "driving",
        "intense",
        "punchy",
        "powerful",
        "quiet",
        "soft",
        "gentle",
        "subtle",
        "explosive",
    },
    "rhythm": {
        "groove",
        "rhythm",
        "beat",
        "tempo",
        "arpeggio",
        "arpeggios",
        "pulse",
        "staccato",
        "percussion",
        "syncopated",
        "swing",
    },
    "instrument": {
        "piano",
        "guitar",
        "bass",
        "drums",
        "strings",
        "synth",
        "synths",
        "violin",
        "cello",
        "brass",
        "flute",
        "organ",
        "pad",
        "pads",
        "horn",
        "horns",
        "vocal",
        "vocals",
    },
    "atmosphere": {
        "atmosphere",
        "atmospheric",
        "space",
        "spacious",
        "dreamy",
        "moody",
        "meditative",
        "ethereal",
        "mysterious",
    },
}


def dedupe_str_list(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def normalize_term_text(term: str) -> str:
    text = str(term or "").strip().lower()
    if ":" in text:
        return text.split(":", 1)[1].strip()
    return text


def _tag_set_from_view(view: Dict[str, Any]) -> set[str]:
    return {str(x or "").strip().lower() for x in list(view.get("tags", []) or []) if str(x or "").strip()}


def _caption_texts_from_payload(payload: Dict[str, Any]) -> List[str]:
    texts: List[str] = []
    for key in ("seed_view", "previous_view", "target_view"):
        view = dict(payload.get(key, {}) or {})
        caption = str(view.get("caption", "") or "").strip().lower()
        if caption:
            texts.append(caption)
    return texts


def _lyric_texts_from_payload(payload: Dict[str, Any]) -> List[str]:
    texts: List[str] = []
    for key in ("seed_view", "previous_view", "target_view"):
        view = dict(payload.get(key, {}) or {})
        lyrics = str(view.get("lyrics", "") or "").strip().lower()
        if lyrics:
            texts.append(lyrics)
    return texts


def _fuzzy_caption_terms_from_payload(payload: Dict[str, Any]) -> List[str]:
    terms: List[str] = []
    for scope in ("from_seed", "from_previous"):
        fuzzy = dict(payload.get("caption_differences_fuzzy", {}).get(scope, {}) or {})
        for key in ("added_terms", "removed_terms", "shared_terms", "added_phrases", "removed_phrases", "shared_phrases"):
            values = fuzzy.get(key, [])
            if isinstance(values, list):
                for value in values:
                    text = str(value or "").strip().lower()
                    if text:
                        terms.append(text)
    return dedupe_str_list(terms)


def _fuzzy_lyric_terms_from_payload(payload: Dict[str, Any]) -> List[str]:
    terms: List[str] = []
    for scope in ("from_seed", "from_previous"):
        fuzzy = dict(payload.get("lyric_differences_fuzzy", {}).get(scope, {}) or {})
        for key in ("added_terms", "removed_terms", "shared_terms", "added_phrases", "removed_phrases", "shared_phrases"):
            values = fuzzy.get(key, [])
            if isinstance(values, list):
                for value in values:
                    text = str(value or "").strip().lower()
                    if text:
                        terms.append(text)
    return dedupe_str_list(terms)


def infer_semantic_item(term: str, payload: Dict[str, Any]) -> Dict[str, str]:
    text = str(term or "").strip()
    normalized = normalize_term_text(text)
    if not text:
        return {"text": "", "source": "semantic", "kind": "semantic"}
    if str(text).lower().startswith("vocals:"):
        return {"text": text, "source": "vocals", "kind": "vocal_status"}
    if str(text).lower().startswith("speed:"):
        return {"text": text, "source": "speed", "kind": "speed"}

    previous_view = dict(payload.get("previous_view", {}) or {})
    target_view = dict(payload.get("target_view", {}) or {})
    seed_view = dict(payload.get("seed_view", {}) or {})
    tag_sets = (
        _tag_set_from_view(previous_view),
        _tag_set_from_view(target_view),
        _tag_set_from_view(seed_view),
    )
    caption_texts = _caption_texts_from_payload(payload)
    lyric_texts = _lyric_texts_from_payload(payload)
    fuzzy_terms = _fuzzy_caption_terms_from_payload(payload)
    lyric_fuzzy_terms = _fuzzy_lyric_terms_from_payload(payload)

    if any(normalized == tag or normalized in tag for tags in tag_sets for tag in tags):
        source = "tag"
    elif any(normalized in caption for caption in caption_texts) or any(normalized in term for term in fuzzy_terms):
        source = "caption"
    elif any(normalized in lyrics for lyrics in lyric_texts) or any(normalized in term for term in lyric_fuzzy_terms):
        source = "lyrics"
    else:
        source = "caption" if " " in normalized else "semantic"

    kind = "semantic"
    tokens = set(normalized.replace("-", " ").split())
    if source == "tag":
        kind = "tag"
    for candidate_kind, hints in _KIND_HINTS.items():
        if tokens & hints:
            kind = candidate_kind
            break
    if source == "caption" and kind == "tag":
        kind = "semantic"
    return {"text": text, "source": source, "kind": kind}


def build_typed_semantic_delta(payload: Dict[str, Any], semantic_delta: Dict[str, Any]) -> Dict[str, Any]:
    preserved = [infer_semantic_item(term, payload) for term in list(semantic_delta.get("preserved", []) or []) if str(term).strip()]
    new = [infer_semantic_item(term, payload) for term in list(semantic_delta.get("new", []) or []) if str(term).strip()]
    lost = [infer_semantic_item(term, payload) for term in list(semantic_delta.get("lost", []) or []) if str(term).strip()]
    return {
        "preserved_items": preserved,
        "new_items": new,
        "lost_items": lost,
        "primary_edit": str(semantic_delta.get("primary_edit", "") or "").strip(),
        "caption_only_change": bool(semantic_delta.get("caption_only_change", False)),
    }


def typed_item_texts(
    typed_delta: Dict[str, Any],
    *,
    source: str | None = None,
    buckets: Sequence[str] = ("preserved_items", "new_items", "lost_items"),
) -> List[str]:
    out: List[str] = []
    for bucket in buckets:
        values = typed_delta.get(bucket, [])
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            if source is not None and str(item.get("source", "") or "") != source:
                continue
            text = str(item.get("text", "") or "").strip()
            if text and text not in out:
                out.append(text)
    return out
