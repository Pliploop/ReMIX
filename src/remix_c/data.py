"""ReMIX-C data: unfiltered variant index (CLI), clip loading, step dataset, datamodule.

Build the index once (CPU, a few minutes):
  PYTHONPATH=src python -m remix_c.data --run-root /gpfs/.../music4all_instruct/music4all_v1
Filtering by judge score happens at load time in `RemixData`, never in the index.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import lightning as L
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

SR = 24_000          # MuQ sample rate
WIN_SEC = 10         # MuQ-MuLan training clip length
FIELD = "history_unaware_instruction"
JUDGES = {"qwen": "llm_ratings_qwen_full.validated.jsonl", "gemma": "llm_ratings_gemma_full.validated.jsonl"}


# ---------------------------------------------------------------- index (one-off)
def _jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _key(r: dict):
    return r["chain_id"], int(r["turn_index"]), int(r.get("variant_index") or 0)


def build_index(run_root: str, folder: str = "instructions_axis_focused_5") -> Path:
    """Every instruction variant (all splits) + both judges' overall/mean scores. No filtering."""
    base = Path(run_root) / folder
    scores: Dict[tuple, dict] = {}
    for judge, fname in JUDGES.items():
        for r in _jsonl(base / "validation" / fname):
            if r.get("instruction_field") != FIELD or not r.get("parse_ok"):
                continue
            s = {q: a.get("score") for q, a in (r.get("answers") or {}).items()}
            vals = [v for v in s.values() if isinstance(v, (int, float))]
            scores.setdefault(_key(r), {}).update({
                f"{judge}_overall": s.get("overall_validity"),
                f"{judge}_mean": sum(vals) / len(vals) if vals else None})
    rows = {}
    for r in _jsonl(base / "chain_step_instructions.jsonl"):
        text = (r.get(FIELD) or "").strip()
        if r.get("status") == "ok" and text:
            rows[_key(r)] = dict(chain_id=r["chain_id"], turn_index=int(r["turn_index"]),
                                 variant_index=int(r.get("variant_index") or 0), split=r["split"],
                                 seed_clip_id=r["seed_clip_id"], target_clip_id=r["target_clip_id"],
                                 instruction=text, **scores.get(_key(r), {}))
    df = pd.DataFrame(rows.values())
    for c in [f"{j}_{s}" for j in JUDGES for s in ("overall", "mean")]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    used = set(df.seed_clip_id) | set(df.target_clip_id)
    clips = pd.read_csv(Path(run_root) / "ingest" / "normalized_track_manifest.csv",
                        usecols=["clip_id", "track_id", "file_path", "start_time", "end_time"])
    clips = clips[clips.clip_id.isin(used)].drop_duplicates("clip_id")
    out = base / "remix_c"
    out.mkdir(exist_ok=True)
    df.to_parquet(out / "index.parquet", index=False)
    clips.to_parquet(out / "clips.parquet", index=False)
    print(f"{len(df):,} variants ({df.split.value_counts().to_dict()}), {len(clips):,} clips -> {out}")
    return out


# ---------------------------------------------------------------- audio
def load_clip(path: str, start: float, end: float, n_windows: int, rand: bool = False) -> torch.Tensor:
    """(n_windows, WIN_SEC*SR) mono 24 kHz; contiguous span, random offset if rand, zero-padded."""
    need = n_windows * WIN_SEC
    off = start + (random.uniform(0, max(0.0, end - start - need)) if rand else 0.0)
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        f.seek(int(off * sr))
        x = f.read(int(need * sr), dtype="float32", always_2d=True)
    x = torchaudio.functional.resample(torch.from_numpy(x.mean(1)), sr, SR)
    x = torch.nn.functional.pad(x, (0, max(0, n_windows * WIN_SEC * SR - len(x))))
    return x[: n_windows * WIN_SEC * SR].view(n_windows, WIN_SEC * SR)


class Clips:
    """clip_id -> audio, plus a stable integer id per clip."""

    def __init__(self, clips: pd.DataFrame, n_windows: int):
        self.df = clips.set_index("clip_id")
        self.ids = {c: i for i, c in enumerate(self.df.index)}
        self.n_windows = n_windows

    def audio(self, clip_id: str, rand: bool = False) -> torch.Tensor:
        r = self.df.loc[clip_id]
        return load_clip(r.file_path, float(r.start_time), float(r.end_time), self.n_windows, rand)


class ClipSet(Dataset):
    """Plain clip loader (corpus embedding for the benchmark)."""

    def __init__(self, clip_ids: List[str], clips: Clips):
        self.clip_ids, self.clips = clip_ids, clips

    def __len__(self):
        return len(self.clip_ids)

    def __getitem__(self, i):
        return self.clips.audio(self.clip_ids[i])


# ---------------------------------------------------------------- dataset
def filter_variants(df: pd.DataFrame, criteria: str, threshold: Optional[float]) -> pd.DataFrame:
    """Keep variants both judges scored >= threshold on `criteria` (overall|mean). None = keep all."""
    if threshold is None:
        return df
    assert criteria in ("overall", "mean"), criteria
    return df[(df[f"qwen_{criteria}"] >= threshold) & (df[f"gemma_{criteria}"] >= threshold)]


class Steps(Dataset):
    """One item per (chain, turn); train samples one of the step's variants each draw."""

    def __init__(self, df: pd.DataFrame, clips: Clips, train: bool, max_steps: Optional[int] = None):
        g = (df.sort_values("variant_index").groupby(["chain_id", "turn_index"], sort=True)
               .agg(seed=("seed_clip_id", "first"), target=("target_clip_id", "first"),
                    instructions=("instruction", list)))
        if max_steps and len(g) > max_steps:
            g = g.sample(max_steps, random_state=0)
        self.items = list(g.itertuples(index=False))
        self.clips, self.train = clips, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        s = self.items[i]
        return dict(seed=self.clips.audio(s.seed, self.train), target=self.clips.audio(s.target, self.train),
                    instruction=random.choice(s.instructions) if self.train else s.instructions[0],
                    seed_id=self.clips.ids[s.seed], target_id=self.clips.ids[s.target])


class RemixData(L.LightningDataModule):
    def __init__(self, index_dir: str, filter_criteria: str = "overall", filter_threshold: Optional[float] = None,
                 val_filter_criteria: str = "overall", val_filter_threshold: Optional[float] = 4,
                 val_max_steps: Optional[int] = 2000, n_windows: int = 3, batch_size: int = 16,
                 num_workers: int = 8):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        h = self.hparams
        df = pd.read_parquet(Path(h.index_dir) / "index.parquet")
        self.clips = Clips(pd.read_parquet(Path(h.index_dir) / "clips.parquet"), h.n_windows)
        self.train_set = Steps(filter_variants(df[df.split == "train"], h.filter_criteria, h.filter_threshold),
                               self.clips, train=True)
        self.val_set = Steps(filter_variants(df[df.split == "val"], h.val_filter_criteria, h.val_filter_threshold),
                             self.clips, train=False, max_steps=h.val_max_steps)
        print(f"[remix_c] train {len(self.train_set):,} steps | val {len(self.val_set):,} steps")

    def _loader(self, ds, train):
        return DataLoader(ds, batch_size=self.hparams.batch_size, shuffle=train, drop_last=train,
                          num_workers=self.hparams.num_workers, pin_memory=True,
                          persistent_workers=self.hparams.num_workers > 0)

    def train_dataloader(self):
        return self._loader(self.train_set, True)

    def val_dataloader(self):
        return self._loader(self.val_set, False)


def _demo():
    df = pd.DataFrame(dict(qwen_overall=[5, 4, 3, np.nan], gemma_overall=[5, 3, 5, 5],
                           qwen_mean=[4.5, 4.2, 4.0, 4.1], gemma_mean=[4.1, 4.0, 3.9, np.nan]))
    assert len(filter_variants(df, "overall", None)) == 4
    assert filter_variants(df, "overall", 4).index.tolist() == [0]       # both >= 4, unrated dropped
    assert filter_variants(df, "mean", 4).index.tolist() == [0, 1]
    print("remix_c.data self-check OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", help="dataset run root (omit to run the self-check)")
    ap.add_argument("--folder", default="instructions_axis_focused_5")
    a = ap.parse_args()
    build_index(a.run_root, a.folder) if a.run_root else _demo()
