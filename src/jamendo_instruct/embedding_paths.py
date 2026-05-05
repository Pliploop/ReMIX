from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict


def clip_file_stem(clip_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", clip_id).strip("._")
    digest = hashlib.sha1(clip_id.encode("utf-8")).hexdigest()[:12]
    if not safe:
        safe = "clip"
    if len(safe) > 80:
        safe = safe[:80]
    return f"{safe}__{digest}"


def embedding_paths_for_clip(clip_id: str, *, audio_dir: Path, text_dir: Path) -> Dict[str, str]:
    stem = clip_file_stem(clip_id)
    return {
        "audio_embedding_path": str(audio_dir / f"{stem}.npy"),
        "text_embedding_path": str(text_dir / f"{stem}.npy"),
    }
