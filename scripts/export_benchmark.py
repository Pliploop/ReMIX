#!/usr/bin/env python
"""Export the ReMIX-B retrieval artifacts (docs/evaluation.md) for one dataset:

  corpus_ids.json   test-split clip ids (row order for the matrices)
  corpus_audio.npy  (N, Da) MuQ-MuLan audio embeddings, L2-normalised
  corpus_text.npy   (N, Dt) EmbeddingGemma text embeddings, L2-normalised
  corpus.jsonl      per-clip metadata (caption, tags, vocals, speed, title, artist)
  queries.jsonl     one per pooled test step (seed + instruction + target)

qrels are read straight from the relevance-pool shards by the harness, so they
are not re-exported here.

  PYTHONPATH=src python scripts/export_benchmark.py --dataset music4all
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

csv.field_size_limit(10 ** 7)

DATASETS = {
    "music4all": "/gpfs/scratch/acw749/datasets/music4all_instruct/music4all_v1",
    "mtg_jamendo": "/gpfs/scratch/acw749/datasets/mtg_jamendo_instruct/v1",
}
FOLDER = "instructions_axis_focused_5"


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


def export(dataset: str, split: str = "test") -> None:
    root = Path(DATASETS[dataset])
    out = root / FOLDER / "benchmark"
    out.mkdir(parents=True, exist_ok=True)

    # per-clip metadata (test split)
    meta = {}
    with open(root / "structured_view" / "structured_clip_manifest.csv") as f:
        for r in csv.DictReader(f):
            if r.get("split") != split:
                continue
            meta[r["clip_id"]] = {
                "caption": r.get("primary_caption") or r.get("caption") or "",
                "tags": [t for t in (r.get("tags") or "").split(", ") if t],
                "vocals": r.get("vocals") or "", "speed": r.get("speed") or "",
                "title": r.get("title") or "", "artist": r.get("artist_name") or "",
            }

    # embedding paths (test split, both towers ok)
    ids, audio_paths, text_paths = [], [], []
    with open(root / "embeddings_reused_audio_text" / "embedding_lookup_manifest.csv") as f:
        for r in csv.DictReader(f):
            if r.get("split") != split or r.get("audio_embedding_status") != "ok" or r.get("text_embedding_status") != "ok":
                continue
            cid = r["clip_id"]
            if cid not in meta:
                continue
            ids.append(cid); audio_paths.append(r["audio_embedding_path"]); text_paths.append(r["text_embedding_path"])

    audio = _l2(np.stack([np.load(p).astype(np.float32) for p in audio_paths]))
    text = _l2(np.stack([np.load(p).astype(np.float32) for p in text_paths]))
    np.save(out / "corpus_audio.npy", audio)
    np.save(out / "corpus_text.npy", text)
    (out / "corpus_ids.json").write_text(json.dumps(ids))
    with (out / "corpus.jsonl").open("w") as fh:
        for cid in ids:
            fh.write(json.dumps({"clip_id": cid, **meta[cid]}) + "\n")
    print(f"corpus: {len(ids):,} clips  audio{audio.shape} text{text.shape} -> {out}")

    # instruction gate: (chain, turn) -> instructions
    gate = {}
    gpath = root / FOLDER / "validation" / "validated_instructions.jsonl"
    for line in open(gpath):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        gate[(r.get("chain_id"), r.get("turn_index"))] = (
            r.get("history_unaware_instruction") or "", r.get("history_aware_instruction") or "")

    # queries: one per pooled test step
    pool_dir = root / FOLDER / "relevance_pool"
    seen = set()
    n = 0
    with (out / "queries.jsonl").open("w") as qf:
        for shard in sorted(pool_dir.glob("chain_step_relevance_pools.shard*.jsonl")):
            for line in open(shard):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                key = (r.get("chain_id"), r.get("turn_index"))
                if key in seen:
                    continue
                seen.add(key)
                hu, ha = gate.get(key, ("", ""))
                qf.write(json.dumps({
                    "query_id": f"{key[0]}#{key[1]}",
                    "seed_clip_id": r.get("source_clip_id"),
                    "target_clip_id": r.get("target_clip_id"),
                    "instruction": hu, "instruction_history_aware": ha,
                }) + "\n")
                n += 1
    print(f"queries: {n:,} pooled test steps -> {out/'queries.jsonl'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(DATASETS), default="music4all")
    ap.add_argument("--split", default="test")
    export(ap.parse_args().dataset, ap.parse_args().split)


if __name__ == "__main__":
    main()
