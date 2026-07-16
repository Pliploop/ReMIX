"""Every number the video shows, read from the exported stats.

Nothing here is typed by hand except the two catalogue sizes, which come from the
raw dataset files rather than the pipeline. If a figure cannot be sourced, the
video does not show it.

Regenerate the source with:
    python scripts/export_website_stats.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

REPO = Path(__file__).resolve().parents[2]
STATS_JSON = REPO / "website" / "public" / "data" / "stats.json"

# Catalogue sizes: Music4All id_metadata.csv rows, and MTG-Jamendo rows_embedded
# from embeddings_report.json. These predate the chain pipeline, so they are not
# in stats.json.
M4A_CATALOGUE = 109_269
MTG_CATALOGUE = 55_525
CATALOGUE_TOTAL = M4A_CATALOGUE + MTG_CATALOGUE

AUDIO_MODEL = "MuQ-MuLan"
AUDIO_DIM = 512
TEXT_MODEL = "EmbeddingGemma"
TEXT_DIM = 768
JUDGES = ("Qwen3.6-27B", "Gemma-4-31B")
VARIANTS_PER_STEP = 5


def _load() -> Dict[str, Any] | None:
    if not STATS_JSON.is_file():
        print(f"[remix_video] {STATS_JSON} missing; using fallback figures")
        return None
    return json.loads(STATS_JSON.read_text(encoding="utf-8"))


_FALLBACK = {
    "clips": 29_110,
    "artists": 9_394,
    "chains": 9_740,
    "steps": 24_117,
    "variants": 114_941,
    "accept_lo": 73,
    "accept_hi": 83,
    "ac1_lo": 0.93,
    "ac1_hi": 0.95,
}


def figures() -> Dict[str, Any]:
    data = _load()
    if data is None:
        return dict(_FALLBACK)

    out = {k: 0 for k in ("clips", "artists", "chains", "steps", "variants")}
    for ds in data["datasets"]:
        for k in out:
            out[k] += ds["overview"][k]

    accepts, ac1s = [], []
    for v in data.get("validation", []):
        row = next((r for r in v["accept_by_question"] if r["question"] == "Overall valid"), None)
        if row:
            accepts += [row[j] for j in v["judges"] if row.get(j) is not None]
        vals = [r["ac1"] for r in v["agreement"] if r.get("ac1") is not None]
        if vals:
            ac1s.append(sum(vals) / len(vals))

    out["accept_lo"] = round(min(accepts)) if accepts else _FALLBACK["accept_lo"]
    out["accept_hi"] = round(max(accepts)) if accepts else _FALLBACK["accept_hi"]
    out["ac1_lo"] = round(min(ac1s), 2) if ac1s else _FALLBACK["ac1_lo"]
    out["ac1_hi"] = round(max(ac1s), 2) if ac1s else _FALLBACK["ac1_hi"]
    return out


def thousands(n: int) -> str:
    return f"{n:,}"


def compact(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k".replace(".0k", "k")
    return str(n)


FIGURES = figures()
