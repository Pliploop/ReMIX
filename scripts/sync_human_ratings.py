#!/usr/bin/env python
"""Pull human ratings from the HF dataset repo back into the run roots.

Closes the loop: the Space collects ratings into a Dataset repo, and this drops
them where the existing analysis already looks --
``<run_root>/<instructions_folder>/validation/human_ratings.jsonl`` -- in the same
schema the Streamlit app writes. scripts/paper_validation_stats.py then needs no
changes.

Additive: writes only human_ratings.jsonl (and only with --write), never touches
instructions, chains or LLM ratings.

Usage:
    python scripts/sync_human_ratings.py --repo Pliploop/remix-human-ratings
    python scripts/sync_human_ratings.py --repo ... --write
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO = Path(__file__).resolve().parents[1]

RUN_ROOTS = {
    "mtg_jamendo": Path("/gpfs/scratch/acw749/datasets/mtg_jamendo_instruct/v1"),
    "music4all": Path("/gpfs/scratch/acw749/datasets/music4all_instruct/music4all_v1"),
}
INSTR_FOLDER = "instructions_axis_focused_5"


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def fetch(repo: str, token: str | None, revision: str) -> List[Dict[str, Any]]:
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        token=token,
        allow_patterns=["**/*.jsonl"],
    )
    records: List[Dict[str, Any]] = []
    for p in sorted(Path(local).rglob("*.jsonl")):
        got = list(_iter_jsonl(p))
        print(f"  {p.name}: {len(got):,} records")
        records += got
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="HF dataset repo holding the ratings.")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--token", default=None, help="HF token (or set HF_TOKEN).")
    ap.add_argument("--folder", default=INSTR_FOLDER)
    ap.add_argument("--write", action="store_true", help="Actually write; default is a dry run.")
    args = ap.parse_args()

    print(f"Fetching {args.repo} ...")
    records = fetch(args.repo, args.token, args.revision)
    if not records:
        sys.exit("No rating records found.")

    # A rater may re-rate an item; keep the newest per (rater, assignment, variant).
    latest: Dict[tuple, Dict[str, Any]] = {}
    for r in records:
        key = (r.get("annotator_id"), r.get("assignment_id"), r.get("variant_index"))
        prev = latest.get(key)
        if prev is None or str(r.get("annotated_at_utc") or "") >= str(prev.get("annotated_at_utc") or ""):
            latest[key] = r
    deduped = list(latest.values())
    print(f"\n{len(records):,} records -> {len(deduped):,} after keeping the latest per (rater, item)")

    by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for r in deduped:
        by_dataset.setdefault(r.get("dataset", "?"), []).append(r)

    for ds, recs in sorted(by_dataset.items()):
        root = RUN_ROOTS.get(ds)
        print(f"\n{ds}: {len(recs):,} ratings")
        print(f"  raters: {len({r.get('annotator_id') for r in recs})}")
        print(f"  buckets: {dict(Counter(r.get('bucket') for r in recs))}")

        # Sentinels are attention checks; a rater failing them is the signal to look at.
        sent = [r for r in recs if r.get("is_sentinel")]
        if sent:
            print(f"  sentinel ratings: {len(sent)} (check these before trusting a rater)")

        if root is None:
            print(f"  ! unknown dataset key, skipping", file=sys.stderr)
            continue
        out = root / args.folder / "validation" / "human_ratings.jsonl"
        if not args.write:
            print(f"  [dry run] would write {len(recs):,} -> {out}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for r in sorted(recs, key=lambda x: str(x.get("annotated_at_utc") or "")):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  wrote {out}")

    if not args.write:
        print("\nDry run. Pass --write to apply.")


if __name__ == "__main__":
    main()
