#!/usr/bin/env python
"""Export every chain plus a spherical embedding projection for the /explore page.

Additive and read-only.

Why a sphere is not just decoration: MuQ-MuLan audio embeddings are L2-normalised,
so the corpus already lives on a 512-d hypersphere where cosine similarity is
angular distance -- the same similarity the pipeline used to build the graph.
Projecting onto a 3-sphere keeps that geometry, so "close on screen" means
"close in the space the chains were sampled from".

  --method umap : UMAP with output_metric='haversine' embeds directly onto a
                  sphere (angles out, no post-hoc squashing). Slower; submit it
                  rather than running it on a login node.
  --method pca  : PCA to 3D then renormalise. Instant, cruder, fine for a smoke test.

Outputs, per dataset, under website/public/data/explorer/<key>/:
  tracks.json  columnar track table (ids, titles, artists, genres, audio refs)
  pos.bin      int16 xyz triples, unit sphere scaled by 32767
  chains.json  every chain, referencing tracks by index

By default only chains the judges validated are kept, and the cloud is subset to
the tracks those chains touch, so the explorer ships a vetted subset rather than
the raw corpus. Each rated chain carries `js` (per-judge mean step score); the
rating threshold is a slider on the page, not baked in here. Pass
--no-filter-chains for the whole corpus. As validation runs at larger scale, the
same command re-run simply keeps more.

Usage:
  python scripts/export_explorer_data.py --method umap        # submit via SLURM

  # Cheap refresh: keep the existing projection, re-apply the current filter, and
  # subset the cloud to what survives. No embeddings, no UMAP.
  python scripts/export_explorer_data.py --reuse-positions

  python scripts/export_explorer_data.py --no-filter-chains   # whole corpus
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_website_data import (  # noqa: E402  (reuse: single source of truth for audio refs)
    INSTR_FOLDER,
    JUDGE_FILES,
    M4A_RAW,
    MTG_RAW,
    _compact,
    _split_tags,
    build_track,
    load_jamendo_licenses,
    load_m4a_metadata,
    load_ratings,
)

csv.field_size_limit(10 ** 9)
REPO = Path(__file__).resolve().parents[1]

DATASETS = [
    ("MTG-Jamendo", "mtg_jamendo", Path("/gpfs/scratch/acw749/datasets/mtg_jamendo_instruct/v1")),
    ("Music4All", "music4all", Path("/gpfs/scratch/acw749/datasets/music4all_instruct/music4all_v1")),
]


def load_chain_steps(instr_path: Path, max_variants: int) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
    """chain -> turn -> variants, ordered by variant_index (best first).

    Every surviving variant is kept, not just the best one: the explorer's dropdown
    exists to show that a step has several valid phrasings. `max_variants` caps how
    many reach the browser, since each one is text we ship to every visitor.
    """
    out: Dict[str, Dict[int, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    with instr_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") not in (None, "ok"):
                continue
            if not (rec.get("history_unaware_instruction") or "").strip():
                continue
            out[rec["chain_id"]][int(rec.get("turn_index", 0))].append(_compact(rec))

    for turns in out.values():
        for turn, variants in turns.items():
            variants.sort(key=lambda r: int(r.get("variant_index") or 0))
            del variants[max_variants:]
    return out


def load_judge_scores(val_dir: Path, names: Tuple[str, ...]) -> List[Dict[Tuple[str, int, int], List[float]]]:
    """One (chain, turn, variant) -> [scores] map per judge file.

    load_ratings merges every path it is given into one map; calling it once per
    judge keeps the judges apart, which is what "pass for both models" needs.
    """
    return [load_ratings([val_dir / name]) for name in names]


def chain_model_means(
    chain_id: str,
    turns: Dict[int, List[Dict[str, Any]]],
    judge_scores: List[Dict[Tuple[str, int, int], List[float]]],
) -> Optional[List[float]]:
    """Per-model mean over the primary variant of each turn.

    Returns None if any turn is unrated by any judge -- which, with judges that
    have only scored the validation sidecar, is how an unvalidated chain is
    dropped rather than kept unseen. The primary variant is the one the explorer
    displays (lowest variant_index), so we gate exactly what a visitor reads.
    """
    means: List[float] = []
    for js in judge_scores:
        vals: List[float] = []
        for turn in sorted(turns):
            primary = turns[turn][0]
            key = (chain_id, turn, int(primary.get("variant_index") or 0))
            got = js.get(key)
            if not got:
                return None
            vals.append(sum(got) / len(got))
        means.append(sum(vals) / len(vals))
    return means


def load_embedding_paths(lookup_csv: Path, wanted: set[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with lookup_csv.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            cid = row.get("clip_id")
            if cid in wanted and row.get("audio_embedding_status") == "ok":
                p = row.get("audio_embedding_path")
                if p:
                    out[cid] = p
    return out


def load_manifest_rows(path: Path, wanted: set[str]) -> Dict[str, Dict[str, Any]]:
    keep = ("clip_id", "track_id", "tags", "split", "start_time", "end_time",
            "title", "artist_name", "vocals", "primary_caption", "caption")
    out: Dict[str, Dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            cid = row.get("clip_id")
            if cid in wanted:
                out[cid] = {k: row.get(k, "") for k in keep}
    return out


def project(vectors: np.ndarray, method: str, seed: int) -> np.ndarray:
    """-> (n, 3) unit-sphere coordinates."""
    if method == "umap":
        import umap  # imported lazily: heavy, and PCA path should not pay for it

        mapper = umap.UMAP(
            output_metric="haversine",
            n_components=2,
            metric="cosine",
            n_neighbors=25,
            min_dist=0.05,
            random_state=seed,
            verbose=True,
        ).fit(vectors)
        theta, phi = mapper.embedding_[:, 0], mapper.embedding_[:, 1]
        xyz = np.stack([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ], axis=1)
    else:
        from sklearn.decomposition import PCA

        xyz = PCA(n_components=3, random_state=seed).fit_transform(vectors)

    norms = np.linalg.norm(xyz, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return xyz / norms


def export(label: str, key: str, root: Path, args, jamendo, m4a) -> Optional[Dict[str, Any]]:
    instr_path = root / args.folder / "chain_step_instructions.jsonl"
    if not instr_path.is_file():
        print(f"  ! missing {instr_path}", file=sys.stderr)
        return None

    out_dir = (REPO / args.out / "explorer" / key).resolve()

    print("  reading chains ...")
    chains = load_chain_steps(instr_path, args.max_variants)

    # Per-chain judge means, attached to each chain as `js` so the explorer can
    # filter by rating in the browser. Filtering lives on the page, not here --
    # but a chain with no rating has nothing for that slider to act on, and
    # shipping the whole unrated corpus is ~30x the bytes, so by default we still
    # keep only validated chains (and subset the cloud to what they touch).
    means_by_chain: Dict[str, List[float]] = {}
    val_dir = root / args.folder / "validation"
    judge_scores = load_judge_scores(val_dir, JUDGE_FILES[key])
    if any(judge_scores):
        for cid, turns in chains.items():
            m = chain_model_means(cid, turns, judge_scores)
            if m is not None:
                means_by_chain[cid] = [round(x, 2) for x in m]
        if args.filter_chains:
            before = len(chains)
            chains = {c: t for c, t in chains.items() if c in means_by_chain}
            print(f"  validated-only: {before:,} -> {len(chains):,} chains "
                  f"(rated by all {len(JUDGE_FILES[key])} judges)")
        else:
            print(f"  scoring only, no filter: {len(means_by_chain):,}/{len(chains):,} chains are rated")
    else:
        print(f"  ! no judge ratings under {val_dir}; shipping chains unrated", file=sys.stderr)

    # Every variant of a turn shares the turn's endpoints, so variant 0 is enough.
    clip_ids = {
        c
        for turns in chains.values()
        for variants in turns.values()
        for c in (variants[0]["source_clip_id"], variants[0]["target_clip_id"])
    }
    print(f"  {len(chains):,} chains · {len(clip_ids):,} distinct clips")

    if args.reuse_positions:
        # Re-exporting text (e.g. to add variants) or narrowing the chain set must
        # not pay for embeddings and UMAP again: nothing here moves a point. We
        # take the previous run's coordinates as-is and, if the chain set shrank,
        # keep only the rows the surviving chains still touch -- in the previous
        # order, since that is what pos.bin was indexed by.
        prev_tracks = out_dir / "tracks.json"
        prev_pos = out_dir / "pos.bin"
        if not (prev_tracks.is_file() and prev_pos.is_file()):
            print(f"  ! --reuse-positions needs existing {prev_tracks} and {prev_pos}", file=sys.stderr)
            return None
        prev_col = json.loads(prev_tracks.read_text(encoding="utf-8"))
        prev_ids = prev_col["clip_id"]
        prev_xyz = (np.frombuffer(prev_pos.read_bytes(), dtype="<i2").astype(np.float32) / 32767.0).reshape(-1, 3)

        keep_rows = [i for i, cid in enumerate(prev_ids) if cid in clip_ids]
        usable = [prev_ids[i] for i in keep_rows]
        index = {cid: i for i, cid in enumerate(usable)}
        xyz = prev_xyz[keep_rows]
        # Per-track columns are subset row-wise; the deduped licence table (whose
        # length is not the track count) is carried through untouched.
        columnar = {
            k: ([v[i] for i in keep_rows] if isinstance(v, list) and len(v) == len(prev_ids) else v)
            for k, v in prev_col.items()
        }
        dropped = len(clip_ids - set(index))
        print(f"  reusing projection: {len(prev_ids):,} -> {len(usable):,} positions kept")
        if dropped:
            print(f"  ! {dropped:,} clips referenced by kept chains are absent from the "
                  f"previous export; re-run without --reuse-positions to include them", file=sys.stderr)
    else:
        manifest = load_manifest_rows(root / "ingest" / "normalized_track_manifest.csv", clip_ids)
        emb_paths = load_embedding_paths(root / "embeddings" / "embedding_lookup_manifest.csv", clip_ids)
        usable = sorted(clip_ids & set(manifest) & set(emb_paths))
        print(f"  usable (manifest + embedding): {len(usable):,}")
        if not usable:
            return None

        print("  loading embeddings ...")
        vecs = np.zeros((len(usable), 512), dtype=np.float32)
        missing = 0
        for i, cid in enumerate(usable):
            try:
                vecs[i] = np.load(emb_paths[cid])
            except OSError:
                missing += 1
            if i and i % 5000 == 0:
                print(f"    {i:,}/{len(usable):,}")
        if missing:
            print(f"  ! {missing} embeddings failed to load", file=sys.stderr)

        print(f"  projecting ({args.method}) ...")
        xyz = project(vecs, args.method, args.seed)

        index = {cid: i for i, cid in enumerate(usable)}

    if not args.reuse_positions:
        tracks = [build_track(cid, manifest, dataset=key, jamendo=jamendo, m4a=m4a) for cid in usable]

        # Columnar, and audio refs are reduced to their identifying id: the Jamendo
        # stream/page URLs are templated client-side and the licence strings are
        # deduped into a table. Inlining them per track tripled the file.
        licenses: List[Dict[str, str]] = []
        lic_index: Dict[Tuple[str, str], int] = {}
        kinds: List[int] = []       # 0 none, 1 jamendo, 2 spotify
        audio_ids: List[str] = []
        lic_refs: List[int] = []

        for t in tracks:
            a = t["audio"]
            if a["kind"] == "jamendo":
                kinds.append(1)
                # ".../?trackid=214&format=mp31" -> "214"
                audio_ids.append(a["url"].split("trackid=")[1].split("&")[0])
                lk = (a.get("license") or "", a.get("license_url") or "")
                if lk not in lic_index:
                    lic_index[lk] = len(licenses)
                    licenses.append({"name": lk[0], "url": lk[1]})
                lic_refs.append(lic_index[lk])
            elif a["kind"] == "spotify":
                kinds.append(2)
                audio_ids.append(a.get("id") or "")
                lic_refs.append(-1)
            else:
                kinds.append(0)
                audio_ids.append("")
                lic_refs.append(-1)

        columnar = {
            "clip_id": [t["clip_id"] for t in tracks],
            "title": [t["title"] for t in tracks],
            "artist": [t["artist"] for t in tracks],
            "tags": [t["tags"][:3] for t in tracks],
            "split": [manifest[cid].get("split", "") for cid in usable],
            "audio_kind": kinds,
            "audio_id": audio_ids,
            "license_ref": lic_refs,
            "licenses": licenses,
        }

    out_chains = []
    for chain_id, turns in chains.items():
        steps = []
        for turn in sorted(turns):
            variants = turns[turn]
            best = variants[0]
            si, ti = index.get(best["source_clip_id"]), index.get(best["target_clip_id"])
            if si is None or ti is None:
                steps = []
                break
            step = {
                "s": si,
                "t": ti,
                # i/c stay the best variant: the sphere labels and any older client
                # read them directly and must not have to understand `v`.
                "i": best.get("history_unaware_instruction") or "",
                "c": best.get("history_aware_instruction") or "",
                "e": best.get("_primary_edit") or "",
                "ax": best.get("selected_change_axes") or [],
                "sc": round(float(best.get("transition_score") or 0), 3),
            }
            # The variant dropdown ships only for rated chains. On the ~30k
            # unrated chains the extra four phrasings are unvetted text nobody
            # asked to compare, and at 5x per step they are what turns the export
            # from ~30MB into ~100MB. Rated chains keep them, since that is where
            # comparing phrasings is worth the bytes.
            if len(variants) > 1 and chain_id in means_by_chain:
                step["v"] = [
                    {
                        "i": v.get("history_unaware_instruction") or "",
                        "c": v.get("history_aware_instruction") or "",
                        "vb": v.get("verbosity") or "",
                    }
                    for v in variants
                ]
            steps.append(step)
        if steps:
            out_chain = {"id": chain_id, "sp": turns[sorted(turns)[0]][0].get("split", ""), "st": steps}
            if chain_id in means_by_chain:
                out_chain["js"] = means_by_chain[chain_id]  # per-judge mean step score
            out_chains.append(out_chain)
    # Best-rated first, so the default view opens on strong examples.
    out_chains.sort(key=lambda c: (-min(c.get("js") or [0]), c["id"]))
    print(f"  chains with all clips resolved: {len(out_chains):,}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Both paths now hold xyz + columnar (the reuse path subset them), and the
    # kept point set can differ from the previous export, so always rewrite all
    # three files together -- they must stay indexed by the same track order.
    # int16 keeps the cloud tiny; 1/32767 is far finer than a pixel on screen.
    (out_dir / "pos.bin").write_bytes(np.clip(xyz * 32767, -32767, 32767).astype("<i2").tobytes())
    (out_dir / "tracks.json").write_text(json.dumps(columnar, ensure_ascii=False), encoding="utf-8")
    (out_dir / "chains.json").write_text(json.dumps(out_chains, ensure_ascii=False), encoding="utf-8")

    sizes = {p.name: p.stat().st_size / 1024 for p in out_dir.iterdir()}
    print("  " + " · ".join(f"{k} {v:.0f}KB" for k, v in sorted(sizes.items())))

    return {
        "key": key,
        "label": label,
        "tracks": len(usable),
        "chains": len(out_chains),
        # Reusing positions leaves whatever projection produced them in place, so
        # claiming args.method here would misreport how the sphere was built.
        "method": "reused" if args.reuse_positions else args.method,
        "max_variants": args.max_variants,
        # Whether the corpus was pre-narrowed to validated chains. The rating
        # threshold itself now lives on the page, driven by each chain's `js`.
        "validated_only": bool(args.filter_chains),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=("pca", "umap"), default="pca")
    ap.add_argument("--folder", default=INSTR_FOLDER)
    ap.add_argument("--out", default="website/public/data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=None, help="Restrict to one dataset key.")
    ap.add_argument(
        "--max-variants",
        type=int,
        default=5,
        help="Instruction variants to ship per step (1 = the old best-only export).",
    )
    ap.add_argument(
        "--reuse-positions",
        action="store_true",
        help="Keep the previous run's projection instead of recomputing it: adopts "
             "its coordinates and subsets them to whatever chains survive. Skips "
             "embeddings and UMAP entirely, so a re-export takes minutes.",
    )
    ap.add_argument(
        "--filter-chains",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ship only validated chains (rated by every judge) and subset the "
             "cloud to the tracks they touch. Default on. --no-filter-chains ships "
             "the whole corpus, rated where possible. Either way each rated chain "
             "carries its per-judge mean score; the threshold slider is on the page.",
    )
    args = ap.parse_args()

    if args.max_variants < 1:
        sys.exit("--max-variants must be >= 1")

    # Neither licence table is read when positions are reused: they only feed
    # tracks.json, which that path does not rewrite.
    jamendo = {} if args.reuse_positions else load_jamendo_licenses(MTG_RAW / "audio_licenses.txt")
    m4a = {} if args.reuse_positions else load_m4a_metadata(M4A_RAW)

    manifest_out = []
    for label, key, root in DATASETS:
        if args.only and key != args.only:
            continue
        print(f"\n{label}:")
        got = export(label, key, root, args, jamendo, m4a)
        if got:
            manifest_out.append(got)

    if not manifest_out:
        sys.exit("Nothing exported.")

    index_path = (REPO / args.out / "explorer" / "index.json").resolve()
    index_path.write_text(json.dumps({"datasets": manifest_out}, indent=1), encoding="utf-8")
    print(f"\nWrote {index_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
