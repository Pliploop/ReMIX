from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from jamendo_instruct.demo.chains_demo import (
    ChainView,
    DemoDataset,
    StepView,
    _append_validation_record,
    _audio_note_html,
    _audio_preview,
    _format_caption,
    _format_tags,
    _html,
    _instruction_folder_options,
    _instruction_text,
    _load_dataset_for_streamlit,
    _render_clip_panel,
    _streamlit_css,
    _streamlit_runtime_active,
)
from jamendo_instruct.demo.validation_rubric import (
    DEGREE_OPTIONS,
    INSTRUCTION_FIELD_LABELS,
    ISSUE_TAGS,
    LIKERT_OPTIONS,
    PAIRWISE_OPTIONS,
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


FAVICON_PATH = Path(__file__).resolve().parents[3] / "assets" / "favicon.png"
SESSION_BATCH_SIZE = 10
DEFAULT_INSTRUCTION_FOLDER = os.environ.get("INSTRUCTION_NAME", "instructions_axis_focused_5")
SENTINEL_FRACTION = 0.5


def _annotator_id(st: Any) -> str:
    for key in ("annotator", "rater", "token"):
        value = st.query_params.get(key, "")
        if isinstance(value, list):
            value = value[0] if value else ""
        value = str(value or "").strip()
        if value:
            return value
    headers = getattr(getattr(st, "context", None), "headers", {}) or {}
    ip = str(headers.get("cf-connecting-ip") or headers.get("x-forwarded-for") or "").split(",", 1)[0].strip()
    if ip:
        return "ip_" + hashlib.sha1(("jamendo-human-validation:" + ip).encode("utf-8")).hexdigest()[:12]
    if "anonymous_rater_id" not in st.session_state:
        st.session_state.anonymous_rater_id = "session_" + hashlib.sha1(os.urandom(16)).hexdigest()[:12]
    return str(st.session_state.anonymous_rater_id)


def _property_tokens(row: Dict[str, str] | None) -> set[str]:
    row = dict(row or {})
    tokens = {f"tag:{tag}" for tag in _format_tags(row)}
    for field in ("vocals", "speed", "genre", "mood", "instrumentation"):
        value = str(row.get(field, "") or "").strip()
        if value:
            tokens.add(f"{field}:{value}")
    return tokens


def _render_token_list(st: Any, title: str, values: Sequence[str]) -> None:
    st.caption(title)
    if not values:
        st.write("None found")
        return
    st.markdown(
        " ".join(f'<span class="ji-token">{_html(value)}</span>' for value in values),
        unsafe_allow_html=True,
    )


def _render_sticky_instruction_header(
    st: Any,
    *,
    instruction: str,
    source_label: str,
    target_label: str,
    chain_id: str,
    turn_index: int,
    variant_index: int,
    session_done: int,
    session_goal: int,
) -> None:
    progress_pct = 100.0 * min(session_done, session_goal) / max(1, session_goal)
    st.markdown(
        (
            '<section class="ji-sticky-instruction">'
            '<div class="ji-sticky-meta">'
            f'<span>Chain {_html(chain_id)}</span>'
            f'<span>Turn {_html(str(turn_index))}</span>'
            f'<span>Variant {_html(str(variant_index))}</span>'
            f'<span>Session {_html(str(min(session_done, session_goal)))} / {_html(str(session_goal))}</span>'
            "</div>"
            f'<div class="ji-sticky-text">{_html(instruction)}</div>'
            '<div class="ji-progress-track">'
            f'<div class="ji-progress-bar" style="width:{progress_pct:.1f}%"></div>'
            "</div>"
            f'<div class="ji-sticky-clips"><span>Source: {_html(source_label)}</span><span>Target: {_html(target_label)}</span></div>'
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def _render_instruction_card(st: Any, title: str, text: str, *, variant_index: int | None = None) -> None:
    variant = "" if variant_index is None else f"Variant {variant_index}"
    st.markdown(
        (
            '<section class="ji-card ji-instruction-card">'
            f'<div class="ji-section-label">{_html(title)}</div>'
            f'<div class="ji-eyebrow">{_html(variant)}</div>'
            f'<p>{_html(text)}</p>'
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def _render_compact_audio(st: Any, title: str, row: Dict[str, str] | None, *, cache_dir: Path) -> None:
    st.markdown(f"**{title}**")
    audio_path, note = _audio_preview(row, cache_dir=cache_dir)
    if audio_path:
        st.audio(audio_path)
    st.markdown(_audio_note_html(note), unsafe_allow_html=True)


def _render_audio_pair(st: Any, dataset: DemoDataset, step: StepView, *, cache_dir: Path) -> None:
    source_row = dataset.manifest_by_clip.get(step.source_clip_id)
    target_row = dataset.manifest_by_clip.get(step.target_clip_id)

    audio_left, audio_right = st.columns(2, gap="large")
    with audio_left:
        _render_compact_audio(st, "Source", source_row, cache_dir=cache_dir)
    with audio_right:
        _render_compact_audio(st, "Target", target_row, cache_dir=cache_dir)


def _render_evidence_details(st: Any, dataset: DemoDataset, step: StepView, *, cache_dir: Path) -> None:
    source_row = dataset.manifest_by_clip.get(step.source_clip_id)
    target_row = dataset.manifest_by_clip.get(step.target_clip_id)

    with st.expander("Full clip cards", expanded=False):
        source_col, target_col = st.columns(2, gap="large")
        with source_col:
            _render_clip_panel(st, "Source Clip", source_row, cache_dir=cache_dir)
        with target_col:
            _render_clip_panel(st, "Target Clip", target_row, cache_dir=cache_dir)


def _render_intro_tab(st: Any) -> None:
    st.markdown(
        """
        <section class="ji-card">
          <div class="ji-section-label">Study Overview</div>
          <h2>Human validation for music-instruction judging</h2>
          <p>
            This interface collects human ratings for source-target music retrieval items and their generated
            instructions. The main research question is whether LLM judges agree with human judgement on
            instruction faithfulness, contradiction, edit specificity, and overall validity.
          </p>
        </section>
        <section class="ji-card">
          <div class="ji-section-label">Rater Instructions</div>
          <h2>What you are judging</h2>
          <p>
            Each instruction asks for a meaningful musical change from the source item toward the target item.
            It may also include a conservation clause, such as keeping the vocal presence, tempo feel, mood,
            instrumentation, or another salient property. The target does not need to preserve every other detail
            of the source. Your job is to judge whether the instruction is sensible, supported by the evidence,
            captures a real change, and does not make claims contradicted by the source, target, caption,
            metadata, or audio.
          </p>
          <p>
            Use audio, captions, tags, and metadata as evidence. The instruction may include both change
            requests and explicit keep/preserve/maintain clauses. If the available evidence is not enough to
            decide, choose Cannot judge. Absence of evidence is not the same as contradiction: contradiction
            means the available evidence indicates that a factual claim is false.
          </p>
          <p>
            Edit specificity matters. A good edit instruction describes a change from source to target, such as
            "Replace the rock drums with a danceable electronic groove." A weaker target-only instruction only
            describes the target, such as "Make it an upbeat synth-pop track."
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.info("Use the Rate tab for absolute rubric labels. Use Compare when variants are available for the same source-target pair.")


def _render_rater_instruction_block(st: Any) -> None:
    st.info(
        "**How to rate this item.** The instruction asks to move from the source track toward the target track. "
        "It should ask for one or more real changes, and it may ask to keep an important property. "
        "The target is allowed to change in other ways too. Use **Cannot judge** when you do not have enough "
        "evidence to answer; not being able to tell is not the same as the instruction being wrong."
    )


def _validation_css() -> str:
    return """
    <style>
    div[data-testid="stRadio"] div[role="radiogroup"] {
      display: flex;
      flex-wrap: wrap;
      gap: 0.45rem;
      align-items: center;
      margin-bottom: 0.55rem;
    }
    div[data-testid="stRadio"] label {
      background: #ffffff;
      border: 1px solid #dfe6df;
      border-radius: 8px;
      padding: 0.35rem 0.55rem;
      min-height: 2.2rem;
      box-shadow: 0 4px 12px rgba(20, 28, 22, 0.04);
    }
    div[data-testid="stRadio"] label:nth-of-type(1) { background: #f8e7e2; border-color: #d99a8c; }
    div[data-testid="stRadio"] label:nth-of-type(2) { background: #fbefe0; border-color: #dfbc86; }
    div[data-testid="stRadio"] label:nth-of-type(3) { background: #f7f3d6; border-color: #d4c568; }
    div[data-testid="stRadio"] label:nth-of-type(4) { background: #e8f3e7; border-color: #9fca9a; }
    div[data-testid="stRadio"] label:nth-of-type(5) { background: #dcefe4; border-color: #78b48e; }
    div[data-testid="stRadio"] label:hover {
      border-color: #286145;
      background: #f9fbf8;
    }
    div[data-testid="stForm"] {
      border: 1px solid #dfe6df;
      border-radius: 8px;
      padding: 1rem;
      background: #f9fbf8;
    }
    .ji-sticky-instruction {
      position: sticky;
      top: 0.35rem;
      z-index: 999;
      background: rgba(255, 255, 255, 0.97);
      border: 1px solid #cdd9cd;
      border-radius: 8px;
      padding: 0.7rem 0.85rem;
      box-shadow: 0 8px 22px rgba(20, 28, 22, 0.10);
      margin-bottom: 0.85rem;
    }
    .ji-sticky-meta, .ji-sticky-clips {
      display: flex;
      flex-wrap: wrap;
      gap: 0.7rem;
      color: #59645a;
      font-size: 0.82rem;
    }
    .ji-sticky-text {
      color: #151915;
      font-weight: 650;
      margin: 0.25rem 0;
      line-height: 1.35;
    }
    .ji-progress-track {
      height: 8px;
      border-radius: 999px;
      background: #e8eee8;
      overflow: hidden;
      margin: 0.45rem 0;
    }
    .ji-progress-bar {
      height: 100%;
      border-radius: 999px;
      background: #286145;
    }
    .ji-next-button button {
      min-height: 3.4rem;
      font-size: 1.1rem;
      font-weight: 750;
    }
    div[data-testid="stButton"] button[kind="primary"] {
      min-height: 3.5rem;
      border: 0 !important;
      border-radius: 8px !important;
      background: #286145 !important;
      color: #ffffff !important;
      font-size: 1.12rem !important;
      font-weight: 800 !important;
      box-shadow: 0 12px 24px rgba(40, 97, 69, 0.24) !important;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
      background: #1f5138 !important;
      box-shadow: 0 14px 28px rgba(40, 97, 69, 0.32) !important;
    }
    div[data-testid="stButton"] button[kind="primary"]:disabled {
      background: #d6ded6 !important;
      color: #6b766c !important;
      box-shadow: none !important;
    }
    .ji-token {
      display: inline-block;
      margin: 0 0.25rem 0.35rem 0;
      padding: 0.18rem 0.42rem;
      border-radius: 999px;
      background: #edf4ee;
      border: 1px solid #ccdbcf;
      font-size: 0.84rem;
    }
    .ji-validation-title {
      margin: 0 0 0.75rem 0;
      padding: 0.15rem 0;
    }
    .ji-validation-title h1 {
      margin: 0;
      font-size: 1.45rem;
      line-height: 1.2;
      letter-spacing: 0;
      color: #171917;
    }
    </style>
    """


def _is_sentinel_item(item: Dict[str, Any]) -> bool:
    assignment = dict(item.get("assignment", {}) or {})
    return bool(assignment.get("is_sentinel", False) or assignment.get("bucket") == "sentinel")


def _build_item_order(items: Sequence[Dict[str, Any]], *, key: str) -> List[int]:
    rng = random.Random(f"human-validation:{key}:13")
    sentinel_indices = [idx for idx, item in enumerate(items) if _is_sentinel_item(item)]
    regular_indices = [idx for idx, item in enumerate(items) if not _is_sentinel_item(item)]
    rng.shuffle(sentinel_indices)
    rng.shuffle(regular_indices)

    if not sentinel_indices or not regular_indices:
        order = list(range(len(items)))
        rng.shuffle(order)
        return order

    sentinel_per_batch = max(1, min(SESSION_BATCH_SIZE, round(SESSION_BATCH_SIZE * SENTINEL_FRACTION)))
    regular_per_batch = max(1, SESSION_BATCH_SIZE - sentinel_per_batch)
    order: List[int] = []
    while sentinel_indices or regular_indices:
        batch: List[int] = []
        for _ in range(sentinel_per_batch):
            if sentinel_indices:
                batch.append(sentinel_indices.pop())
        for _ in range(regular_per_batch):
            if regular_indices:
                batch.append(regular_indices.pop())
        while len(batch) < SESSION_BATCH_SIZE and sentinel_indices:
            batch.append(sentinel_indices.pop())
        while len(batch) < SESSION_BATCH_SIZE and regular_indices:
            batch.append(regular_indices.pop())
        rng.shuffle(batch)
        order.extend(batch)
    return order


def _ensure_order(key: str, items_or_count: Sequence[Dict[str, Any]] | int) -> List[int]:
    count = len(items_or_count) if not isinstance(items_or_count, int) else int(items_or_count)
    state = f"{key}_order"
    count_state = f"{key}_count"
    if st.session_state.get(count_state) != count or state not in st.session_state:
        if isinstance(items_or_count, int):
            order = list(range(count))
            random.Random(13).shuffle(order)
        else:
            order = _build_item_order(items_or_count, key=key)
        st.session_state[state] = order
        st.session_state[count_state] = count
        st.session_state[f"{key}_pos"] = 0
    return list(st.session_state[state])


def _current_ordered_item(key: str, items: Sequence[Dict[str, Any]]) -> tuple[int, Dict[str, Any]]:
    order = _ensure_order(key, items)
    pos_key = f"{key}_pos"
    pos = max(0, min(int(st.session_state.get(pos_key, 0) or 0), len(order) - 1))
    st.session_state[pos_key] = pos
    item_index = order[pos]
    return item_index, items[item_index]


def _current_ordered_position(key: str, items_or_count: Sequence[Dict[str, Any]] | int) -> int:
    count = len(items_or_count) if not isinstance(items_or_count, int) else int(items_or_count)
    _ensure_order(key, items_or_count)
    pos_key = f"{key}_pos"
    pos = max(0, min(int(st.session_state.get(pos_key, 0) or 0), max(0, count - 1)))
    st.session_state[pos_key] = pos
    return pos


def _advance(key: str, count: int) -> None:
    pos_key = f"{key}_pos"
    st.session_state[pos_key] = min(count - 1, int(st.session_state.get(pos_key, 0) or 0) + 1)


def _score_bucket_counts(records: Sequence[Dict[str, Any]], question_id: str) -> Dict[int, int]:
    counts = {score: 0 for score in range(1, 6)}
    for record in records:
        answer = dict((record.get("answers", {}) or {}).get(question_id, {}) or {})
        score = answer.get("score")
        if isinstance(score, int) and score in counts:
            counts[score] += 1
    return counts


def _admin_question_rows(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for question in RATING_QUESTIONS:
        question_id = str(question["id"])
        scored: List[int] = []
        cannot_judge = 0
        not_applicable = 0
        for record in records:
            answer = dict((record.get("answers", {}) or {}).get(question_id, {}) or {})
            if bool(answer.get("cannot_judge", False)):
                cannot_judge += 1
                continue
            if bool(answer.get("not_applicable", False)):
                not_applicable += 1
                continue
            score = answer.get("score")
            if isinstance(score, int):
                scored.append(score)
        buckets = _score_bucket_counts(records, question_id)
        agreement_count = sum(1 for score in scored if score >= 4)
        rows.append(
            {
                "question_id": question_id,
                "question": question["statement"],
                "polarity": question["polarity"],
                "n_scored": len(scored),
                "cannot_judge": cannot_judge,
                "not_applicable": not_applicable,
                "mean_score": round(sum(scored) / len(scored), 3) if scored else None,
                "agree_or_strongly_agree_rate": round(agreement_count / len(scored), 3) if scored else None,
                "score_1": buckets[1],
                "score_2": buckets[2],
                "score_3": buckets[3],
                "score_4": buckets[4],
                "score_5": buckets[5],
            }
        )
    return rows


def _pairwise_item_key(record: Dict[str, Any]) -> tuple[str, int, int, int, str]:
    left = int(record.get("variant_a_index", 0) or 0)
    right = int(record.get("variant_b_index", 0) or 0)
    return (
        str(record.get("chain_id", "") or ""),
        int(record.get("turn_index", 0) or 0),
        min(left, right),
        max(left, right),
        str(record.get("instruction_field", "") or ""),
    )


def _preference_rows(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    for record in records:
        preference = str(record.get("preference", "") or "Unknown")
        counts[preference] = counts.get(preference, 0) + 1
    return [{"preference": key, "count": value} for key, value in sorted(counts.items())]


def _issue_tag_rows(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {tag: 0 for tag in ISSUE_TAGS}
    for record in records:
        for tag in record.get("issue_tags", []) or []:
            text = str(tag)
            counts[text] = counts.get(text, 0) + 1
    return [{"issue_tag": key, "count": value} for key, value in sorted(counts.items()) if value]


def _flatten_rating_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        row = {
            "annotated_at_utc": record.get("annotated_at_utc", ""),
            "annotator_id": record.get("annotator_id", ""),
            "annotation_type": record.get("annotation_type", ""),
            "instruction_field": record.get("instruction_field", ""),
            "chain_id": record.get("chain_id", ""),
            "turn_index": record.get("turn_index", ""),
            "variant_index": record.get("variant_index", ""),
            "split": record.get("split", ""),
            "source_clip_id": record.get("source_clip_id", ""),
            "target_clip_id": record.get("target_clip_id", ""),
            "source_label": record.get("source_label", ""),
            "target_label": record.get("target_label", ""),
            "instruction": record.get("instruction", ""),
            "issue_tags": ", ".join(str(tag) for tag in record.get("issue_tags", []) or []),
            "notes": record.get("notes", ""),
        }
        audio = dict(record.get("audio_interaction", {}) or {})
        for key, value in audio.items():
            row[f"audio_{key}"] = value
        answers = dict(record.get("answers", {}) or {})
        for question in RATING_QUESTIONS:
            qid = str(question["id"])
            answer = dict(answers.get(qid, {}) or {})
            row[qid] = answer.get("score")
            row[f"{qid}_label"] = answer.get("label", "")
            row[f"{qid}_cannot_judge"] = bool(answer.get("cannot_judge", False))
            row[f"{qid}_not_applicable"] = bool(answer.get("not_applicable", False))
        rows.append(row)
    return rows


def _recent_rows(records: Sequence[Dict[str, Any]], *, limit: int = 20) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in sorted(records, key=lambda item: str(item.get("annotated_at_utc", "") or ""), reverse=True)[:limit]:
        rows.append(
            {
                "time": record.get("annotated_at_utc", ""),
                "annotator": record.get("annotator_id", ""),
                "type": record.get("annotation_type", ""),
                "chain_id": record.get("chain_id", ""),
                "turn_index": record.get("turn_index", ""),
                "variant": record.get("variant_index", record.get("variant_a_index", "")),
                "preference": record.get("preference", ""),
            }
        )
    return rows


def _discover_llm_rating_files(output_dir: Path) -> List[Path]:
    if not output_dir.exists():
        return []
    return sorted(p for p in output_dir.glob("llm_ratings*.jsonl") if p.is_file())


def _llm_rating_records(output_dir: Path) -> List[Dict[str, Any]]:
    """All llm_ratings*.jsonl records, deduped per (model, item) keeping the last seen.

    Files are read in sorted order, so a later/superset run (e.g. a full-validation
    file) overrides an earlier partial run for overlapping items, and re-runs do not
    inflate the aggregates.
    """
    order: List[Dict[str, Any]] = []
    index: Dict[tuple[str, tuple[str, int, int, str]], int] = {}
    for path in _discover_llm_rating_files(output_dir):
        for record in _read_jsonl_records(path):
            key = (str(record.get("annotator_id", "") or ""), _rating_item_key(record))
            if key in index:
                order[index[key]] = record
            else:
                index[key] = len(order)
                order.append(record)
    return order


def _llm_model_ids(records: Sequence[Dict[str, Any]]) -> List[str]:
    ids = {
        str(record.get("annotator_id", "") or "").strip()
        for record in records
        if str(record.get("annotator_id", "") or "").strip()
    }
    return sorted(ids)


def _item_mean_scores(records: Sequence[Dict[str, Any]], question_id: str) -> Dict[tuple[str, int, int, str], float]:
    """Mean numeric score per item for one question, skipping cannot-judge / N-A / unparsed."""
    per_item: Dict[tuple[str, int, int, str], List[int]] = {}
    for record in records:
        answer = dict((record.get("answers", {}) or {}).get(question_id, {}) or {})
        if bool(answer.get("cannot_judge", False)) or bool(answer.get("not_applicable", False)):
            continue
        score = answer.get("score")
        if isinstance(score, int):
            per_item.setdefault(_rating_item_key(record), []).append(score)
    return {key: sum(values) / len(values) for key, values in per_item.items()}


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / ((sxx * syy) ** 0.5)


def _accept_kappa(hs: Sequence[float], ls: Sequence[float]) -> float | None:
    """Cohen's kappa on the binary accept decision (score >= 4)."""
    n = len(hs)
    if n == 0:
        return None
    ha = [1 if x >= 4 else 0 for x in hs]
    la = [1 if y >= 4 else 0 for y in ls]
    po = sum(1 for a, b in zip(ha, la) if a == b) / n
    p1h = sum(ha) / n
    p1l = sum(la) / n
    pe = p1h * p1l + (1 - p1h) * (1 - p1l)
    if pe >= 1:
        return None
    return (po - pe) / (1 - pe)


def _accept_ac1(hs: Sequence[float], ls: Sequence[float]) -> float | None:
    """Gwet's AC1 on the binary accept decision. Robust to prevalence, so it
    does not collapse when almost everything is accepted (unlike Cohen's kappa)."""
    n = len(hs)
    if n == 0:
        return None
    ha = [1 if x >= 4 else 0 for x in hs]
    la = [1 if y >= 4 else 0 for y in ls]
    po = sum(1 for a, b in zip(ha, la) if a == b) / n
    pi = (sum(ha) + sum(la)) / (2 * n)     # mean prevalence of "accept"
    pe = 2 * pi * (1 - pi)
    if pe >= 1:
        return None
    return (po - pe) / (1 - pe)


def _quadratic_kappa(hs: Sequence[float], ls: Sequence[float], *, k: int = 5) -> float | None:
    """Quadratic-weighted kappa over the 1..k scale (scores rounded to categories).
    Near-misses get partial credit, so it is the right agreement metric for the
    ordinal 1-5 rubric."""
    n = len(hs)
    if n == 0:
        return None
    def cat(x: float) -> int:
        return min(k, max(1, int(round(x))))
    a = [cat(x) for x in hs]
    b = [cat(y) for y in ls]
    row = [0.0] * (k + 1)
    col = [0.0] * (k + 1)
    obs = [[0.0] * (k + 1) for _ in range(k + 1)]
    for x, y in zip(a, b):
        obs[x][y] += 1
        row[x] += 1
        col[y] += 1
    num = den = 0.0
    for i in range(1, k + 1):
        for j in range(1, k + 1):
            w = ((i - j) ** 2) / ((k - 1) ** 2)   # disagreement weight
            expected = row[i] * col[j] / n
            num += w * obs[i][j]
            den += w * expected
    if den == 0:
        return None
    return 1.0 - num / den


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Rank correlation (Pearson on average ranks); better than Pearson for ordinal."""
    n = len(xs)
    if n < 3:
        return None

    def ranks(values: Sequence[float]) -> List[float]:
        order = sorted(range(n), key=lambda i: values[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for t in range(i, j + 1):
                out[order[t]] = avg
            i = j + 1
        return out

    return _pearson(ranks(xs), ranks(ys))


def _agreement_rows(
    left_records: Sequence[Dict[str, Any]],
    right_records: Sequence[Dict[str, Any]],
    *,
    left_label: str = "human",
    right_label: str = "llm",
) -> List[Dict[str, Any]]:
    """Per-question agreement between two rater sets (human-vs-LLM or LLM-vs-LLM)."""
    rows: List[Dict[str, Any]] = []
    for question in RATING_QUESTIONS:
        qid = str(question["id"])
        left_map = _item_mean_scores(left_records, qid)
        right_map = _item_mean_scores(right_records, qid)
        keys = sorted(set(left_map) & set(right_map))
        if not keys:
            rows.append({"question_id": qid, "question": question["statement"], "n_items": 0})
            continue
        hs = [left_map[k] for k in keys]
        ls = [right_map[k] for k in keys]
        n = len(keys)
        left_mean = sum(hs) / n
        right_mean = sum(ls) / n
        mae = sum(abs(h - l) for h, l in zip(hs, ls)) / n
        within1 = sum(1 for h, l in zip(hs, ls) if abs(h - l) <= 1) / n
        accept = sum(1 for h, l in zip(hs, ls) if (h >= 4) == (l >= 4)) / n

        def _r(value: float | None) -> float | None:
            return round(value, 3) if value is not None else None

        rows.append(
            {
                "question_id": qid,
                "question": question["statement"],
                "n_items": n,
                f"{left_label}_mean": round(left_mean, 3),
                f"{right_label}_mean": round(right_mean, 3),
                "mean_diff": round(right_mean - left_mean, 3),   # + => right rater more lenient
                "mae": round(mae, 3),
                "within1_rate": round(within1, 3),
                "accept_agree_rate": round(accept, 3),
                "accept_kappa": _r(_accept_kappa(hs, ls)),
                "accept_ac1": _r(_accept_ac1(hs, ls)),
                "quadratic_kappa": _r(_quadratic_kappa(hs, ls)),
                "pearson_r": _r(_pearson(hs, ls)),
                "spearman_r": _r(_spearman(hs, ls)),
            }
        )
    return rows


def _short_model(annotator_id: str) -> str:
    """`llm:google/gemma-2-27b-it` -> `gemma-2-27b-it` for compact labels."""
    text = str(annotator_id or "").split(":", 1)[-1]
    return text.rsplit("/", 1)[-1] or text


def _render_cross_llm_agreement_section(st: Any, output_dir: Path) -> None:
    llm_records = _llm_rating_records(output_dir)
    st.markdown("**Cross-LLM agreement**")
    models = _llm_model_ids(llm_records)
    if len(models) < 2:
        st.info("Need ratings from at least two judge models. Run the sidecar judge with a second `--model-id`.")
        return
    labels = {m: _short_model(m) for m in models}
    col_a, col_b = st.columns(2)
    model_a = col_a.selectbox("Model A", options=models, index=0, format_func=labels.get, key="cross_llm_a")
    default_b = 1 if models[0] == model_a else 0
    model_b = col_b.selectbox("Model B", options=models, index=default_b, format_func=labels.get, key="cross_llm_b")
    if model_a == model_b:
        st.info("Pick two different models to compare.")
        return
    recs_a = [r for r in llm_records if str(r.get("annotator_id", "") or "") == model_a]
    recs_b = [r for r in llm_records if str(r.get("annotator_id", "") or "") == model_b]
    st.dataframe(
        _agreement_rows(recs_a, recs_b, left_label=labels[model_a], right_label=labels[model_b]),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Over items rated by both models. mean_diff = B - A (+ = B more lenient); within1 = share within 1 point; "
        "accept = score >= 4. accept_kappa (Cohen) is deflated under high accept rates; "
        "accept_ac1 (Gwet) is prevalence-robust; quadratic_kappa weights near-misses; "
        "pearson_r / spearman_r over shared items."
    )


def _llm_per_model_question_means(records: Sequence[Dict[str, Any]], models: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model in models:
        model_records = [record for record in records if str(record.get("annotator_id", "") or "") == model]
        row: Dict[str, Any] = {"model": model, "n_ratings": len(model_records)}
        for question in RATING_QUESTIONS:
            qid = str(question["id"])
            scores = [
                answer.get("score")
                for record in model_records
                for answer in [dict((record.get("answers", {}) or {}).get(qid, {}) or {})]
                if isinstance(answer.get("score"), int)
                and not bool(answer.get("cannot_judge", False))
                and not bool(answer.get("not_applicable", False))
            ]
            row[qid] = round(sum(scores) / len(scores), 2) if scores else None
        rows.append(row)
    return rows


def _llm_notes_rows(records: Sequence[Dict[str, Any]], *, limit: int = 25) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        note = str(record.get("notes", "") or "").strip()
        if not note:
            continue
        overall = dict((record.get("answers", {}) or {}).get("overall_validity", {}) or {})
        rows.append(
            {
                "chain_id": record.get("chain_id", ""),
                "turn_index": record.get("turn_index", ""),
                "variant": record.get("variant_index", ""),
                "overall": overall.get("label", ""),
                "instruction": str(record.get("instruction", "") or "")[:120],
                "notes": note[:300],
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _render_llm_agreement_section(st: Any, output_dir: Path, human_records: Sequence[Dict[str, Any]]) -> None:
    llm_records = _llm_rating_records(output_dir)
    st.markdown("**Human vs LLM agreement**")
    if not llm_records:
        st.info("No `llm_ratings*.jsonl` found yet. Run `scripts/llm_validation_judge.py` to populate LLM ratings.")
        return
    if not human_records:
        st.info("No human ratings yet, so there is nothing to compare the LLM judge against.")
        return
    models = _llm_model_ids(llm_records)
    model = st.selectbox("Compare human ratings against LLM model", options=models, index=0, key="agreement_model")
    model_records = [record for record in llm_records if str(record.get("annotator_id", "") or "") == model]
    st.dataframe(_agreement_rows(human_records, model_records), width="stretch", hide_index=True)
    st.caption(
        "Over items rated by both humans and the selected LLM (scores averaged per item across annotators). "
        "mean_diff = LLM - human (+ = LLM more lenient); within1 = share within 1 point; "
        "accept = score >= 4. accept_kappa (Cohen) is deflated when almost everything is accepted; "
        "accept_ac1 (Gwet) is prevalence-robust; quadratic_kappa weights near-misses on the 1-5 scale; "
        "pearson_r / spearman_r over shared items."
    )


def _render_llm_ratings_tab(
    st: Any,
    dataset: DemoDataset,
    samples: Sequence[Dict[str, Any]],
    admin_password: str,
) -> None:
    entered = st.text_input(
        "Admin password", type="password", key="llm_admin_password", help="Required to view LLM judge summaries."
    )
    if entered != admin_password:
        st.info("Enter the admin password to view LLM judge summaries.")
        return

    output_dir = _validation_output_dir(dataset)
    files = _discover_llm_rating_files(output_dir)
    st.subheader("LLM Judge Ratings")
    if not files:
        st.info(f"No `llm_ratings*.jsonl` found in `{output_dir}`. Run `scripts/llm_validation_judge.py` first.")
        return
    st.caption("Reading " + ", ".join(f"`{path.name}`" for path in files))

    all_records = _llm_rating_records(output_dir)
    models = _llm_model_ids(all_records)
    selected = st.multiselect("Judge model(s)", options=models, default=models, key="llm_models")
    active = set(selected) if selected else set(models)
    records = [record for record in all_records if str(record.get("annotator_id", "") or "") in active]

    unique_items = {_rating_item_key(record) for record in records}
    parse_failures = sum(1 for record in records if not bool(record.get("parse_ok", True)))
    unparsed_rows = sum(1 for record in records if int(record.get("unparsed_questions", 0) or 0) > 0)

    metric_cols = st.columns(5)
    metric_cols[0].metric("LLM ratings", f"{len(records):,}", f"{len(unique_items):,} unique")
    metric_cols[1].metric("Coverage", f"{len(unique_items) / max(1, len(samples)):.1%}")
    metric_cols[2].metric("Judge models", f"{len(active):,}")
    metric_cols[3].metric("Parse failures", f"{parse_failures:,}")
    metric_cols[4].metric("Rows w/ unparsed Q", f"{unparsed_rows:,}")

    st.markdown("**Rubric Scores (LLM judge)**")
    st.dataframe(_admin_question_rows(records), width="stretch", hide_index=True)

    if len(models) > 1:
        with st.expander("Per-model mean score by question", expanded=True):
            st.dataframe(_llm_per_model_question_means(all_records, models), width="stretch", hide_index=True)

    with st.expander("Diagnostic issue tags (LLM judge)", expanded=True):
        st.dataframe(_issue_tag_rows(records), width="stretch", hide_index=True)

    with st.expander("Sample judge notes", expanded=False):
        st.dataframe(_llm_notes_rows(records), width="stretch", hide_index=True)

    with st.expander("LLM rating export table", expanded=False):
        st.dataframe(_flatten_rating_records(records), width="stretch", hide_index=True)

    with st.expander("Artifact paths", expanded=False):
        st.code("\n".join(str(path) for path in files), language="text")


def _render_admin_tab(
    st: Any,
    dataset: DemoDataset,
    samples: Sequence[Dict[str, Any]],
    pairs: Sequence[Dict[str, Any]],
    admin_password: str,
) -> None:
    entered = st.text_input("Admin password", type="password", help="Required to view aggregate annotations.")
    if entered != admin_password:
        st.info("Enter the admin password to view annotation summaries.")
        return

    output_dir = _validation_output_dir(dataset)
    ratings_path = output_dir / "human_ratings.jsonl"
    pairwise_path = output_dir / "human_pairwise_preferences.jsonl"
    rating_records = _read_jsonl_records(ratings_path)
    pairwise_records = _read_jsonl_records(pairwise_path)

    st.subheader("Admin")
    st.caption(f"Reading `{ratings_path}` and `{pairwise_path}`.")
    st.code(
        "\n".join(
            [
                f"run_root: {dataset.paths.run_root}",
                f"instructions_jsonl: {dataset.paths.instructions_jsonl}",
                f"output_dir: {output_dir}",
                f"rating_items: {len(samples)}",
                f"pairwise_comparisons: {len(pairs)}",
            ]
        ),
        language="text",
    )

    unique_rating_items = {_rating_item_key(record) for record in rating_records}
    unique_pairwise_items = {_pairwise_item_key(record) for record in pairwise_records}
    annotators = {
        str(record.get("annotator_id", "") or "").strip()
        for record in [*rating_records, *pairwise_records]
        if str(record.get("annotator_id", "") or "").strip()
    }

    metric_cols = st.columns(5)
    metric_cols[0].metric("Ratings", f"{len(rating_records):,}", f"{len(unique_rating_items):,} unique")
    metric_cols[1].metric("Rating Coverage", f"{len(unique_rating_items) / max(1, len(samples)):.1%}")
    metric_cols[2].metric("Pairwise", f"{len(pairwise_records):,}", f"{len(unique_pairwise_items):,} unique")
    metric_cols[3].metric("Pairwise Coverage", f"{len(unique_pairwise_items) / max(1, len(pairs)):.1%}")
    metric_cols[4].metric("Annotators", f"{len(annotators):,}")

    st.markdown("**Rubric Scores**")
    question_rows = _admin_question_rows(rating_records)
    st.dataframe(question_rows, width="stretch", hide_index=True)

    with st.expander("Pairwise preference counts", expanded=True):
        st.dataframe(_preference_rows(pairwise_records), width="stretch", hide_index=True)

    with st.expander("Diagnostic issue tags", expanded=True):
        st.dataframe(_issue_tag_rows(rating_records), width="stretch", hide_index=True)

    _render_llm_agreement_section(st, output_dir, rating_records)
    _render_cross_llm_agreement_section(st, output_dir)

    with st.expander("Rating export table", expanded=False):
        st.dataframe(_flatten_rating_records(rating_records), width="stretch", hide_index=True)

    with st.expander("Recent annotations", expanded=False):
        st.dataframe(_recent_rows([*rating_records, *pairwise_records]), width="stretch", hide_index=True)

    with st.expander("Artifact paths", expanded=False):
        st.code(f"ratings: {ratings_path}\npairwise: {pairwise_path}", language="text")


def _render_rating_tab(st: Any, dataset: DemoDataset, samples: Sequence[Dict[str, Any]], *, cache_dir: Path, instruction_field: str) -> None:
    if not samples:
        st.warning("No instruction variants are available for the selected instruction field.")
        return

    annotator_id = _annotator_id(st)
    order_key = f"rating_{instruction_field}_{annotator_id}"
    if f"{order_key}_session_goal" not in st.session_state:
        st.session_state[f"{order_key}_session_goal"] = min(SESSION_BATCH_SIZE, len(samples))
    session_goal = min(int(st.session_state.get(f"{order_key}_session_goal", SESSION_BATCH_SIZE) or SESSION_BATCH_SIZE), len(samples))
    current_pos = _current_ordered_position(order_key, samples)
    if current_pos >= session_goal:
        st.success(f"Session complete: {session_goal} instruction(s) reviewed.")
        if session_goal < len(samples):
            if st.button("Do 10 more", type="primary", use_container_width=True):
                st.session_state[f"{order_key}_session_goal"] = min(session_goal + SESSION_BATCH_SIZE, len(samples))
                st.rerun()
        else:
            st.info("All loaded validation items have been shown in this session.")
        return

    _idx, sample = _current_ordered_item(order_key, samples)
    chain: ChainView = sample["chain"]
    step: StepView = sample["step"]
    record: Dict[str, Any] = sample["record"]
    output_path = _validation_output_dir(dataset) / "human_ratings.jsonl"
    source_row = dataset.manifest_by_clip.get(step.source_clip_id)
    target_row = dataset.manifest_by_clip.get(step.target_clip_id)
    existing_records = _read_jsonl_records(output_path)
    annotator_done = _answered_rating_keys(existing_records, annotator_id)
    all_done = _answered_rating_keys(existing_records)
    item_key = _rating_item_key(
        {
            "chain_id": chain.chain_id,
            "turn_index": step.turn_index,
            "variant_index": _record_variant_index(record),
            "instruction_field": instruction_field,
        }
    )

    _render_sticky_instruction_header(
        st,
        instruction=sample["instruction"],
        source_label=_clip_label(source_row, step.source_clip_id),
        target_label=_clip_label(target_row, step.target_clip_id),
        chain_id=chain.chain_id,
        turn_index=step.turn_index,
        variant_index=_record_variant_index(record),
        session_done=current_pos,
        session_goal=session_goal,
    )
    if item_key in all_done:
        st.warning("This item already has at least one saved rating.")

    _render_rater_instruction_block(st)
    _render_audio_pair(st, dataset, step, cache_dir=cache_dir)
    _render_evidence_details(st, dataset, step, cache_dir=cache_dir)

    form_key = f"rating_form_{chain.chain_id}_{step.turn_index}_{_record_variant_index(record)}_{instruction_field}"
    st.caption(f"Annotator: `{annotator_id}`")
    answers: Dict[str, str | None] = {}
    for question in RATING_QUESTIONS:
        qid = str(question["id"])
        options = [label for label, _score in _question_scale(question)] + ["Cannot judge"]
        if bool(question.get("allow_na", False)):
            options.append("Not applicable")
        st.markdown(f"**{question['statement']}**")
        st.caption(_rating_help(qid))
        answers[qid] = st.radio(
            str(question["statement"]),
            options=options,
            index=None,
            horizontal=True,
            label_visibility="collapsed",
            key=f"{form_key}_{qid}",
        )
    issue_tags = st.multiselect(
        "Diagnostic issue tags",
        options=list(ISSUE_TAGS),
        help="Optional lightweight labels for later analysis. Choose any that apply.",
    )
    missing = [qid for qid, label in answers.items() if not label]
    disabled = bool(missing)
    next_left, next_right = st.columns([0.72, 0.28], gap="large")
    with next_left:
        skipped_seen = st.button(
            "I've already seen this track",
            help="Skip this item and record that it was not rated because the track was already familiar.",
            use_container_width=True,
        )
    with next_right:
        st.markdown('<div class="ji-next-button">', unsafe_allow_html=True)
        submitted = st.button("Next item ->", type="primary", use_container_width=True, disabled=disabled)
        st.markdown("</div>", unsafe_allow_html=True)

    if skipped_seen:
        _append_validation_record(
            output_path,
            {
                "annotation_type": "single_variant_skip",
                "annotated_at_utc": datetime.now(timezone.utc).isoformat(),
                "annotator_id": annotator_id,
                "instruction_field": instruction_field,
                **_sample_identity(chain, step, record),
                "source_label": _clip_label(source_row, step.source_clip_id),
                "target_label": _clip_label(target_row, step.target_clip_id),
                "source_metadata": _metadata_view(source_row),
                "target_metadata": _metadata_view(target_row),
                "assignment": sample.get("assignment"),
                "instruction": sample["instruction"],
                "skip_reason": "already_seen_track",
                "answers": {},
                "issue_tags": [],
                "notes": "Rater skipped because they had already seen this track.",
            },
        )
        _advance(order_key, len(samples))
        st.rerun()

    if submitted:
        scored_answers: Dict[str, Any] = {}
        for question in RATING_QUESTIONS:
            qid = str(question["id"])
            label = answers[qid]
            score, cannot_judge, not_applicable = _rating_value(label, _question_scale(question))
            scored_answers[qid] = {
                "label": label,
                "score": score,
                "cannot_judge": cannot_judge,
                "not_applicable": not_applicable,
                "polarity": question["polarity"],
            }
        _append_validation_record(
            output_path,
            {
                "annotation_type": "single_variant_rating",
                "annotated_at_utc": datetime.now(timezone.utc).isoformat(),
                "annotator_id": annotator_id,
                "instruction_field": instruction_field,
                **_sample_identity(chain, step, record),
                "source_label": _clip_label(source_row, step.source_clip_id),
                "target_label": _clip_label(target_row, step.target_clip_id),
                "source_metadata": _metadata_view(source_row),
                "target_metadata": _metadata_view(target_row),
                "assignment": sample.get("assignment"),
                "instruction": sample["instruction"],
                "answers": scored_answers,
                "issue_tags": list(issue_tags),
                "notes": "",
            },
        )
        _advance(order_key, len(samples))
        st.rerun()


def _render_pairwise_tab(st: Any, dataset: DemoDataset, pairs: Sequence[Dict[str, Any]], *, cache_dir: Path, instruction_field: str) -> None:
    if not pairs:
        st.info("No steps with at least two instruction variants were found in the loaded slice.")
        return

    _idx, pair = _current_ordered_item(f"pairwise_{instruction_field}", pairs)
    chain: ChainView = pair["chain"]
    step: StepView = pair["step"]
    record_a: Dict[str, Any] = pair["a"]
    record_b: Dict[str, Any] = pair["b"]
    output_path = _validation_output_dir(dataset) / "human_pairwise_preferences.jsonl"
    source_row = dataset.manifest_by_clip.get(step.source_clip_id)
    target_row = dataset.manifest_by_clip.get(step.target_clip_id)

    left, right = st.columns(2, gap="large")
    with left:
        _render_instruction_card(
            st,
            "Instruction A",
            _instruction_text(record_a, instruction_field),
            variant_index=_record_variant_index(record_a),
        )
    with right:
        _render_instruction_card(
            st,
            "Instruction B",
            _instruction_text(record_b, instruction_field),
            variant_index=_record_variant_index(record_b),
        )

    _render_audio_pair(st, dataset, step, cache_dir=cache_dir)
    _render_evidence_details(st, dataset, step, cache_dir=cache_dir)
    st.info(
        "Choose which instruction is the better source-to-target instruction, considering audio support, "
        "caption/tag/metadata support, meaningfulness of the requested change, edit specificity, clarity, "
        "and absence of contradictions."
    )
    st.caption(
        f"Source: {_clip_label(source_row, step.source_clip_id)} | "
        f"Target: {_clip_label(target_row, step.target_clip_id)} | "
        f"Chain {chain.chain_id} turn {step.turn_index}"
    )

    form_key = (
        f"pairwise_form_{chain.chain_id}_{step.turn_index}_"
        f"{_record_variant_index(record_a)}_{_record_variant_index(record_b)}_{instruction_field}"
    )
    with st.form(form_key):
        annotator_id = _annotator_id(st)
        st.caption(f"Annotator: `{annotator_id}`")
        preference = st.radio(
            "Which instruction is the better validation candidate for this source-target pair?",
            options=PAIRWISE_OPTIONS,
            index=None,
        )
        submitted = st.form_submit_button("Save Preference", use_container_width=True)

    if submitted:
        if not preference:
            st.error("Choose a preference before saving.")
            return
        _append_validation_record(
            output_path,
            {
                "annotation_type": "pairwise_variant_preference",
                "annotated_at_utc": datetime.now(timezone.utc).isoformat(),
                "annotator_id": annotator_id,
                "instruction_field": instruction_field,
                "chain_id": chain.chain_id,
                "turn_index": step.turn_index,
                "split": step.split,
                "source_clip_id": step.source_clip_id,
                "target_clip_id": step.target_clip_id,
                "variant_a_index": _record_variant_index(record_a),
                "variant_b_index": _record_variant_index(record_b),
                "instruction_a": _instruction_text(record_a, instruction_field),
                "instruction_b": _instruction_text(record_b, instruction_field),
                "assignment": pair.get("assignment"),
                "preference": preference,
                "notes": "",
            },
        )
        st.success(f"Saved to {output_path}")
        _advance(f"pairwise_{instruction_field}", len(pairs))
        st.rerun()


def _render_streamlit_app(args: argparse.Namespace) -> None:
    import streamlit as st

    globals()["st"] = st
    st.set_page_config(page_title="Composed Music Retrieval validation", page_icon=str(FAVICON_PATH), layout="wide")
    st.markdown(_streamlit_css(), unsafe_allow_html=True)
    st.markdown(_validation_css(), unsafe_allow_html=True)

    max_chains = None if args.max_chains is not None and args.max_chains <= 0 else args.max_chains
    default_run_root = str(Path(args.run_root).expanduser()) if args.run_root else ""

    @st.cache_data(show_spinner="Loading validation slice...")
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

    instruction_field = "history_unaware_instruction"
    active_run_root = default_run_root or None
    active_manifest_csv = None if active_run_root else args.manifest_csv
    active_chains_jsonl = None if active_run_root else args.chains_jsonl
    if args.instructions_jsonl:
        active_instructions_jsonl = str(Path(args.instructions_jsonl).expanduser())
    elif active_run_root:
        active_instructions_jsonl = str(Path(active_run_root) / DEFAULT_INSTRUCTION_FOLDER / "chain_step_instructions.jsonl")
    else:
        active_instructions_jsonl = args.instructions_jsonl

    cache_dir = Path(tempfile.gettempdir()) / "jamendo_instruct_human_validation"
    frozen_sidecar_path = Path(args.frozen_sidecar_json).expanduser().resolve() if args.frozen_sidecar_json else None
    if frozen_sidecar_path is not None:
        try:
            dataset, assignments = _dataset_from_frozen_sidecar(frozen_sidecar_path)
        except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
            st.error(f"Could not load frozen validation sidecar: {exc}")
            st.stop()
    else:
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
            st.error(str(exc))
            st.stop()
        assignment_path = Path(args.assignment_jsonl).expanduser().resolve() if args.assignment_jsonl else None
        try:
            assignments = _read_assignments(assignment_path)
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.stop()
    samples, pairs = _filter_by_assignments(
        _available_samples(dataset, instruction_field),
        _available_pairs(dataset, instruction_field),
        assignments,
    )
    output_dir = _validation_output_dir(dataset)

    st.markdown(
        (
            '<section class="ji-validation-title">'
            "<h1>Composed Music Retrieval validation</h1>"
            "</section>"
        ),
        unsafe_allow_html=True,
    )

    intro_tab, rating_tab, pairwise_tab, admin_tab, llm_tab = st.tabs(
        ["Intro", "Rate Variant", "Compare Variants", "Admin", "LLM Ratings"]
    )
    with intro_tab:
        _render_intro_tab(st)
    with rating_tab:
        _render_rating_tab(st, dataset, samples, cache_dir=cache_dir, instruction_field=instruction_field)
    with pairwise_tab:
        _render_pairwise_tab(st, dataset, pairs, cache_dir=cache_dir, instruction_field=instruction_field)
    with admin_tab:
        _render_admin_tab(st, dataset, samples, pairs, args.admin_password)
    with llm_tab:
        _render_llm_ratings_tab(st, dataset, samples, args.admin_password)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch a Streamlit app for human validation of Jamendo-Instruct variants.")
    parser.add_argument("--run-root", help="Run artifact root, e.g. /path/to/<run_name>.")
    parser.add_argument("--manifest-csv", help="Explicit path to structured_clip_manifest.csv.")
    parser.add_argument("--chains-jsonl", help="Explicit path to sampled_chains.jsonl.")
    parser.add_argument("--instructions-jsonl", help="Optional path to chain_step_instructions.jsonl.")
    parser.add_argument("--assignment-jsonl", help="Optional frozen human-validation assignment JSONL.")
    parser.add_argument("--frozen-sidecar-json", help="Optional compact validation sidecar generated with the assignment.")
    parser.add_argument(
        "--admin-password",
        default=os.environ.get("VALIDATION_ADMIN_PASSWORD", "jamendo-admin"),
        help="Password required to view the Admin tab.",
    )
    parser.add_argument("--chain-offset", type=int, default=0, help="How many instructed chains to skip before loading.")
    parser.add_argument("--max-chains", type=int, default=0, help="Maximum chains to load; 0 loads all.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host for the Streamlit app.")
    parser.add_argument("--port", type=int, default=7861, help="Server port for the Streamlit app.")
    return parser


def _streamlit_forwarded_args(args: argparse.Namespace) -> List[str]:
    forwarded: List[str] = []
    for attr, flag in (
        ("run_root", "--run-root"),
        ("manifest_csv", "--manifest-csv"),
        ("chains_jsonl", "--chains-jsonl"),
        ("instructions_jsonl", "--instructions-jsonl"),
        ("assignment_jsonl", "--assignment-jsonl"),
        ("frozen_sidecar_json", "--frozen-sidecar-json"),
        ("admin_password", "--admin-password"),
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
            "Streamlit is required for the validation app. Install it with `pip install -e .[demo]` "
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
