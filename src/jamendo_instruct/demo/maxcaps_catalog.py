from __future__ import annotations

import argparse
import csv
import html
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


DEFAULT_MANIFEST = Path("/gpfs/scratch/acw749/datasets/maxcaps_instruct/v1/ingest/normalized_track_manifest.csv")
DEFAULT_AUDIO_ROOT = Path("/data/EECS-Pauwels-C4DM/JamendoMaxCaps")
PAGE_SIZE = 10
TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class TrackRecord:
    track_id: str
    title: str
    artist_name: str
    duration: str
    split: str
    audio_path: str
    captions: List[str]
    tags: List[str]
    search_text: str
    search_words: List[str]


@dataclass
class SearchIndex:
    all_indices: List[int]
    token_to_indices: Dict[str, List[int]]
    vocabulary: List[str]


def _parse_json_list(raw: str) -> List[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _dedupe(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        clean = str(item).strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
    return out


def _audio_path(track_id: str, audio_root: Path) -> str:
    shard = track_id[-2:].zfill(2)
    return str(audio_root / shard / f"{track_id}.mp3")


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(str(text or "").lower())


def _load_tracks(manifest_csv: Path, audio_root: Path, max_tracks: int | None = None) -> List[TrackRecord]:
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")
    if not audio_root.exists():
        raise FileNotFoundError(f"Audio root not found: {audio_root}")

    by_track: Dict[str, Dict[str, Any]] = {}
    with manifest_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            track_id = str(row.get("track_id", "") or "").strip()
            if not track_id:
                continue

            record = by_track.get(track_id)
            if record is None:
                record = {
                    "track_id": track_id,
                    "title": str(row.get("title", "") or "").strip(),
                    "artist_name": str(row.get("artist_name", "") or "").strip(),
                    "duration": str(row.get("duration", "") or "").strip(),
                    "split": str(row.get("split", "") or "").strip(),
                    "captions": [],
                    "tags": [],
                }
                by_track[track_id] = record
                if max_tracks is not None and len(by_track) >= max_tracks:
                    pass

            captions = _parse_json_list(str(row.get("track_captions_json", "") or ""))
            if not captions:
                captions = [str(row.get("caption", "") or "").strip()]
            record["captions"].extend(captions)
            record["tags"].extend(_parse_json_list(str(row.get("normalized_tags_json", "") or "")))

            if max_tracks is not None and len(by_track) >= max_tracks:
                # Finish the current row and stop early for quick smoke-test launches.
                break

    tracks: List[TrackRecord] = []
    for item in by_track.values():
        captions = _dedupe(item["captions"])
        tags = _dedupe(item["tags"])
        search_text = " ".join(captions).lower()
        search_words = _dedupe(_tokenize(search_text))
        tracks.append(
            TrackRecord(
                track_id=item["track_id"],
                title=item["title"],
                artist_name=item["artist_name"],
                duration=item["duration"],
                split=item["split"],
                audio_path=_audio_path(item["track_id"], audio_root),
                captions=captions,
                tags=tags,
                search_text=search_text,
                search_words=search_words,
            )
        )
    return tracks


def _build_search_index(tracks: Sequence[TrackRecord]) -> SearchIndex:
    token_to_indices: Dict[str, List[int]] = {}
    for idx, track in enumerate(tracks):
        for token in track.search_words:
            token_to_indices.setdefault(token, []).append(idx)
    return SearchIndex(
        all_indices=list(range(len(tracks))),
        token_to_indices=token_to_indices,
        vocabulary=sorted(token_to_indices.keys()),
    )


def _fuzzy_tokens(query_tokens: Sequence[str], vocabulary: Sequence[str], limit: int = 12) -> List[str]:
    matches: List[str] = []
    for query_token in query_tokens:
        if len(query_token) < 3:
            continue
        local_matches: List[tuple[float, str]] = []
        for token in vocabulary:
            if token.startswith(query_token[:2]) or query_token.startswith(token[:2]):
                ratio = SequenceMatcher(None, query_token, token).ratio()
                if ratio >= 0.82:
                    local_matches.append((ratio, token))
        local_matches.sort(reverse=True)
        matches.extend(token for _, token in local_matches[:3])
        if len(matches) >= limit:
            break
    return _dedupe(matches[:limit])


def _candidate_indices(tracks: Sequence[TrackRecord], index: SearchIndex, query: str) -> List[int]:
    query = query.strip().lower()
    if not query:
        return index.all_indices

    query_tokens = _dedupe(_tokenize(query))
    if not query_tokens:
        return index.all_indices

    candidates: set[int] = set()
    for token in query_tokens:
        candidates.update(index.token_to_indices.get(token, []))

    if not candidates:
        for fuzzy_token in _fuzzy_tokens(query_tokens, index.vocabulary):
            candidates.update(index.token_to_indices.get(fuzzy_token, []))

    if not candidates:
        for idx, track in enumerate(tracks):
            if query in track.search_text[:1200]:
                candidates.add(idx)

    if not candidates:
        return []
    return sorted(candidates)


def _score_query(query: str, track: TrackRecord) -> float:
    query = query.strip().lower()
    if not query:
        return 1.0

    haystack = track.search_text
    if query in haystack:
        return 4.0 + min(len(query) / 120.0, 1.0)

    query_tokens = _dedupe(_tokenize(query))
    if not query_tokens:
        return 0.0

    exact_hits = sum(1 for token in query_tokens if token in track.search_words)
    if exact_hits:
        return 3.0 + (exact_hits / len(query_tokens))

    partial_hits = sum(1 for token in query_tokens if token in haystack)
    if partial_hits:
        return 2.0 + (partial_hits / len(query_tokens))

    fuzzy_hits = 0
    for token in query_tokens:
        if any(SequenceMatcher(None, token, word).ratio() >= 0.84 for word in track.search_words[:120]):
            fuzzy_hits += 1
    if fuzzy_hits:
        return 1.0 + (fuzzy_hits / len(query_tokens))

    return 0.0


def _filter_tracks(tracks: Sequence[TrackRecord], index: SearchIndex, query: str) -> List[int]:
    candidates = _candidate_indices(tracks, index, query)
    if not query.strip():
        return candidates

    scored = [(idx, _score_query(query, tracks[idx])) for idx in candidates]
    scored = [(idx, score) for idx, score in scored if score > 0.0]
    scored.sort(key=lambda item: item[1], reverse=True)
    return [idx for idx, _ in scored]


def _caption_markdown(track: TrackRecord) -> str:
    title = track.title or f"Track {track.track_id}"
    artist = f" by {track.artist_name}" if track.artist_name else ""
    meta = " · ".join(part for part in [f"`{track.track_id}`", track.duration and f"{track.duration}s", track.split] if part)
    tags = ", ".join(f"`{html.escape(tag)}`" for tag in track.tags) or "_No tags_"
    captions = "\n\n".join(f"{i + 1}. {html.escape(caption)}" for i, caption in enumerate(track.captions))
    return f"### {html.escape(title)}{html.escape(artist)}\n{meta}\n\n**Tags:** {tags}\n\n**Captions**\n\n{captions}"


def _page_outputs(tracks: Sequence[TrackRecord], indices: Sequence[int], page: int) -> tuple[Any, ...]:
    total_pages = max(1, (len(indices) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = max(0, min(page, total_pages - 1))
    start = page * PAGE_SIZE
    page_indices = list(indices[start : start + PAGE_SIZE])

    outputs: List[Any] = [page, f"Page {page + 1} / {total_pages} · {len(indices):,} songs"]
    for idx in page_indices:
        track = tracks[idx]
        audio_value = track.audio_path if Path(track.audio_path).exists() else None
        outputs.extend([audio_value, _caption_markdown(track)])

    for _ in range(PAGE_SIZE - len(page_indices)):
        outputs.extend([None, ""])
    return tuple(outputs)


def _build_app(tracks: Sequence[TrackRecord], source_label: str, audio_root: Path):
    try:
        import gradio as gr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Gradio is not installed. Install the demo extra with: "
            "python -m pip install -e '.[demo]'"
        ) from exc

    search_index = _build_search_index(tracks)

    with gr.Blocks(title="JamendoMaxCaps Catalog") as demo:
        gr.Markdown(f"# JamendoMaxCaps Catalog\nLoaded {len(tracks):,} songs from `{source_label}`.")
        filtered_state = gr.State(list(range(len(tracks))))
        page_state = gr.State(0)

        with gr.Row():
            query_box = gr.Textbox(label="Fuzzy caption search", placeholder="Search captions, e.g. calm piano ocean")
            search_button = gr.Button("Search", variant="primary")
            clear_button = gr.Button("Clear")

        with gr.Row():
            prev_button = gr.Button("Previous")
            page_label = gr.Markdown()
            next_button = gr.Button("Next")

        row_components: List[Any] = []
        for i in range(PAGE_SIZE):
            with gr.Accordion(f"Song {i + 1}", open=i == 0):
                audio = gr.Audio(label="Audio", type="filepath")
                details = gr.Markdown()
                row_components.extend([audio, details])

        outputs = [page_state, page_label, *row_components]

        def search(query: str):
            indices = _filter_tracks(tracks, search_index, query)
            return (*_page_outputs(tracks, indices, 0), indices)

        def clear():
            indices = list(range(len(tracks)))
            return (*_page_outputs(tracks, indices, 0), indices, "")

        def turn_page(indices: Sequence[int], page: int, delta: int):
            return _page_outputs(tracks, indices, page + delta)

        demo.load(lambda: _page_outputs(tracks, list(range(len(tracks))), 0), outputs=outputs)
        query_box.submit(search, inputs=[query_box], outputs=[*outputs, filtered_state])
        search_button.click(search, inputs=[query_box], outputs=[*outputs, filtered_state])
        clear_button.click(clear, outputs=[*outputs, filtered_state, query_box])
        prev_button.click(lambda indices, page: turn_page(indices, page, -1), inputs=[filtered_state, page_state], outputs=outputs)
        next_button.click(lambda indices, page: turn_page(indices, page, 1), inputs=[filtered_state, page_state], outputs=outputs)

    return demo


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browse JamendoMaxCaps audio, captions, and tags with Gradio.")
    parser.add_argument("--manifest-csv", default=str(DEFAULT_MANIFEST), help="Ingest normalized_track_manifest.csv path.")
    parser.add_argument("--audio-root", default=str(DEFAULT_AUDIO_ROOT), help="Root containing two-digit sharded MP3 folders.")
    parser.add_argument("--max-tracks", type=int, default=None, help="Optional early-load limit for quick testing.")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest_csv = Path(args.manifest_csv).expanduser().resolve()
    audio_root = Path(args.audio_root).expanduser().resolve()
    tracks = _load_tracks(manifest_csv, audio_root, max_tracks=args.max_tracks)
    if not tracks:
        raise RuntimeError(f"No tracks loaded from {manifest_csv}")
    app = _build_app(tracks, str(manifest_csv), audio_root)
    app.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        allowed_paths=[str(audio_root)],
    )


if __name__ == "__main__":
    main()
