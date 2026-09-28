#!/usr/bin/env python
"""Build the two folders for the Streamlit rating Space (space_streamlit/README.md).

  <out>/space/    code only -> the public Space repo (Pliploop/Remix-Ratings)
  <out>/dataset/  clips + frozen sidecars -> the PRIVATE ratings dataset (Pliploop/Remix-Human-Ratings)

  sbatch -p compute -c 16 --mem=32G -t 1:00:00 --wrap \
      "PYTHONPATH=src python scripts/prepare_rating_space.py --out /gpfs/scratch/acw749/remix_rating_space"

Clips are cut from the source audio at each manifest row's start/end and saved as MP3, named by
chains_demo.clip_audio_name so the app finds them under REMIX_AUDIO_DIR.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from jamendo_instruct.demo.chains_demo import clip_audio_name

ROOT = Path(__file__).resolve().parents[1]
CATALOGUES = {
    "music4all": Path("/gpfs/scratch/acw749/datasets/music4all_instruct/music4all_v1/validation"),
    "mtg_jamendo": Path("/gpfs/scratch/acw749/datasets/mtg_jamendo_instruct/v1/validation"),
}


def _cut(job) -> str | None:
    src, start, end, dest = job
    if Path(dest).exists():
        return None
    import soundfile as sf
    try:
        with sf.SoundFile(src) as f:
            f.seek(int(start * f.samplerate))
            audio = f.read(int((end - start) * f.samplerate), dtype="float32", always_2d=True)
            sr = f.samplerate
        sf.write(dest, audio, sr, format="MP3")
        return None
    except Exception as e:
        return f"{src}: {e.__class__.__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--slice", default="assignment_axis_focused_5_v1",
                    help="sidecar stem: assignment_axis_focused_5_v1 (calibration, 100 items) or "
                         "assignment_full_validation_v1 (~1.1k items)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    a = ap.parse_args()
    out = Path(a.out)

    space = out / "space"
    if space.exists():
        shutil.rmtree(space)
    shutil.copytree(ROOT / "space_streamlit", space)
    shutil.copytree(ROOT / "src/jamendo_instruct/demo", space / "src/jamendo_instruct/demo",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(ROOT / "src/jamendo_instruct/__init__.py", space / "src/jamendo_instruct/__init__.py")
    (space / "assets").mkdir()
    shutil.copy(ROOT / "assets/favicon.png", space / "assets/favicon.png")                 # page icon
    (space / "website/public").mkdir(parents=True)
    shutil.copy(ROOT / "website/public/remix.mp4", space / "website/public/remix.mp4")    # intro video

    jobs = []
    for key, vdir in CATALOGUES.items():
        sidecar = vdir / f"{a.slice}.sidecar.json"
        (out / "dataset/sidecars").mkdir(parents=True, exist_ok=True)
        shutil.copy(sidecar, out / "dataset/sidecars" / f"{key}.sidecar.json")
        adir = out / "dataset/audio" / key
        adir.mkdir(parents=True, exist_ok=True)
        rows = json.loads(sidecar.read_text())["manifest_by_clip"]
        for cid, r in rows.items():
            jobs.append((r["file_path"], float(r.get("start_time") or 0), float(r.get("end_time") or 30),
                         str(adir / clip_audio_name(cid))))
        print(f"{key}: {len(rows)} clips from {sidecar.name}", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        errors = [e for e in ex.map(_cut, jobs, chunksize=8) if e]
    for e in errors:
        print("FAILED", e)
    size = sum(p.stat().st_size for p in (out / "dataset").rglob("*") if p.is_file())
    print(f"{len(jobs) - len(errors)}/{len(jobs)} clips, dataset folder {size / 1e6:.0f} MB -> {out}")


if __name__ == "__main__":
    main()
