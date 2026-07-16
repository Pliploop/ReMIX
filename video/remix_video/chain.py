"""The one real chain the whole video follows.

Pulled from the exported showcase data rather than invented, so every track,
instruction and delta on screen is genuine ReMIX output. chain_00000496 is
MTG-Jamendo, scored 5.0 by both judges, and its second turn ("Keep vocals, make
them robotic and metal.") states the thesis by itself: keep one thing, change
another.

MTG-Jamendo is Creative Commons, so if audio is ever added to the video the
tracks are already licence-clean.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[2]
CHAINS_JSON = REPO / "website" / "public" / "data" / "chains.json"

CHAIN_ID = "chain_00000496"
DATASET_KEY = "mtg_jamendo"

# Used only if the export is missing, so a render never dies on data plumbing.
_FALLBACK: Dict[str, Any] = {
    "chain_id": CHAIN_ID,
    "steps": [
        {
            "instruction": "swap guitars for industrial percussion and spoken word",
            "source": {"title": "Bone Dry", "artist": "Conway Hambone", "tags": ["rock", "guitar"]},
            "target": {"title": "Wired", "artist": "The Hate Eighties", "tags": ["industrial"]},
            "change_axes": ["instrumentation"],
        },
        {
            "instruction": "Keep vocals, make them robotic and metal.",
            "source": {"title": "Wired", "artist": "The Hate Eighties", "tags": ["industrial"]},
            "target": {"title": "Iron Lung", "artist": "After Many Days", "tags": ["metal"]},
            "change_axes": ["genre_style"],
        },
        {
            "instruction": "Shout vocals, fast punk, heavy aggressive",
            "source": {"title": "Iron Lung", "artist": "After Many Days", "tags": ["metal"]},
            "target": {"title": "Ignition", "artist": "Countdown", "tags": ["punk"]},
            "change_axes": ["vocal_style_or_gender"],
        },
        {
            "instruction": "Ditch the heavy metal for energetic pop-punk hooks.",
            "source": {"title": "Ignition", "artist": "Countdown", "tags": ["punk"]},
            "target": {"title": "Sugar Burn", "artist": "Crazed Outlook", "tags": ["pop-punk"]},
            "change_axes": ["genre_style"],
        },
    ],
}


def load_chain() -> Dict[str, Any]:
    if not CHAINS_JSON.is_file():
        print(f"[remix_video] {CHAINS_JSON} missing; using fallback chain")
        return _FALLBACK
    data = json.loads(CHAINS_JSON.read_text(encoding="utf-8"))
    for ds in data.get("datasets", []):
        if ds.get("key") != DATASET_KEY:
            continue
        for c in ds.get("chains", []):
            if c.get("chain_id") == CHAIN_ID:
                return c
    print(f"[remix_video] {CHAIN_ID} not in export; using fallback chain")
    return _FALLBACK


def steps() -> List[Dict[str, Any]]:
    return load_chain()["steps"]


def tracks() -> List[Dict[str, Any]]:
    """Every track along the chain: each source, plus the final target."""
    s = steps()
    return [s[0]["source"]] + [x["target"] for x in s]
