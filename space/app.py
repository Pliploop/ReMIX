"""FastAPI backend for the ReMIX rating app, built to run as a Hugging Face Space.

Why this shape:

A Space's filesystem is ephemeral -- it is wiped on restart, rebuild, and when a
free Space sleeps. Writing ratings to a local file would silently lose them.
``CommitScheduler`` is the primitive Hugging Face provides for exactly this: it
appends to a local JSONL and pushes it to a Dataset repo on an interval, so the
ratings end up versioned, durable, and next to the dataset itself.

The wire format is deliberately the same JSONL schema the Streamlit app writes,
so scripts/paper_validation_stats.py can read human ratings from here with no
changes.

Env:
  DATASET_REPO  target HF dataset repo, e.g. "Pliploop/remix-human-ratings"
  HF_TOKEN      write token (set as a Space secret)
  EVERY_MINUTES commit interval (default 5)
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

DATASET_REPO = os.environ.get("DATASET_REPO", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
EVERY_MINUTES = float(os.environ.get("EVERY_MINUTES", "5"))

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data/ratings"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
RATINGS_PATH = DATA_DIR / "human_ratings.jsonl"

STATIC_DIR = Path(os.environ.get("STATIC_DIR", "static"))

scheduler = None
if DATASET_REPO and HF_TOKEN:
    from huggingface_hub import CommitScheduler

    scheduler = CommitScheduler(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        folder_path=DATA_DIR,
        path_in_repo="data",
        every=EVERY_MINUTES,
        token=HF_TOKEN,
        private=True,
    )
    print(f"[remix] CommitScheduler -> {DATASET_REPO} every {EVERY_MINUTES} min")
else:
    # Local dev, or a misconfigured Space. Say so loudly rather than pretend.
    print("[remix] WARNING: DATASET_REPO/HF_TOKEN unset — ratings stay on the ephemeral local disk only")

app = FastAPI(title="ReMIX rating API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # the app is also served from GitHub Pages
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class Answer(BaseModel):
    label: Optional[str] = None
    score: Optional[int] = None
    cannot_judge: bool = False
    not_applicable: bool = False


class Rating(BaseModel):
    annotation_type: str = "human_single_variant_rating"
    annotator_id: str
    annotator_name: Optional[str] = None
    annotated_at_utc: Optional[str] = None
    dataset: str
    assignment_id: str
    chain_id: str
    turn_index: int
    variant_index: int
    instruction_field: str
    instruction: str
    bucket: Optional[str] = None
    is_sentinel: bool = False
    split: Optional[str] = None
    stable_hash: Optional[str] = None
    source_clip_id: str
    target_clip_id: str
    modality: str = "audio_and_text"
    audio_available: bool = True
    answers: Dict[str, Answer] = Field(default_factory=dict)
    issue_tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


def _append(record: Dict[str, Any]) -> None:
    line = json.dumps(record, ensure_ascii=False)
    # Take the scheduler's lock so we never append mid-upload.
    if scheduler is not None:
        with scheduler.lock:
            with RATINGS_PATH.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    else:
        with RATINGS_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "persisting_to": DATASET_REPO or None,
        "ratings_on_disk": sum(1 for _ in RATINGS_PATH.open()) if RATINGS_PATH.exists() else 0,
    }


@app.post("/api/ratings")
def post_rating(rating: Rating) -> Dict[str, Any]:
    record = rating.model_dump()
    record["received_at_utc"] = datetime.now(timezone.utc).isoformat()
    # Not setdefault: the key is always present (pydantic default), just None. The
    # sync script dedupes and orders on this, so it must never be null.
    if not record.get("annotated_at_utc"):
        record["annotated_at_utc"] = record["received_at_utc"]
    record["record_id"] = str(uuid.uuid4())
    try:
        _append(record)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"could not persist rating: {e}") from e
    return {"ok": True, "record_id": record["record_id"]}


@app.get("/api/progress/{annotator_id}")
def progress(annotator_id: str) -> Dict[str, Any]:
    """Lets a rater resume on another browser; localStorage alone cannot."""
    if not RATINGS_PATH.exists():
        return {"done": []}
    done = []
    with RATINGS_PATH.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("annotator_id") == annotator_id:
                done.append(rec.get("assignment_id"))
    return {"done": done}


# The built SPA. Mounted last so /api/* wins.
if STATIC_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        candidate = STATIC_DIR / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(STATIC_DIR / "index.html")
