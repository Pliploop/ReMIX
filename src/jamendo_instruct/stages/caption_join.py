from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
from jamendo_instruct.progress import StageTracker, rich_tqdm
from omegaconf import DictConfig

CONF_DIR = str(Path(__file__).resolve().parents[3] / "conf")


def _log(cfg: DictConfig, message: str) -> None:
    if bool(cfg.stage.progress.enabled):
        print(f"[caption_join] {message}", flush=True)


def _split_captions(raw: str, cfg: DictConfig) -> List[str]:
    parts = raw.split(str(cfg.stage.rewrites.separator))
    out: List[str] = []
    for p in parts:
        x = p.strip() if bool(cfg.stage.rewrites.trim_each_part) else p
        if bool(cfg.stage.rewrites.drop_empty_parts) and not x:
            continue
        out.append(x)
    return out


def _pick_primary(captions: List[str], cfg: DictConfig) -> str:
    if not captions:
        return ""
    strategy = str(cfg.stage.rewrites.primary_caption_strategy)
    if strategy == "first":
        return captions[0]
    return max(captions, key=len)


def _load_rewrites(cfg: DictConfig) -> Tuple[Dict[str, Dict[str, str]], Dict[str, int]]:
    rewrites: Dict[str, Dict[str, str]] = {}
    stats = {"rows_read": 0, "rows_kept": 0}

    if not bool(cfg.stage.rewrites.enabled):
        return rewrites, stats

    csv_path = cfg.stage.rewrites.csv_path
    if not csv_path:
        return rewrites, stats

    path = Path(str(csv_path))
    if not path.exists():
        if bool(cfg.stage.behavior.require_rewrites_file):
            raise FileNotFoundError(f"Rewrites CSV not found: {path}")
        _log(cfg, f"Rewrites CSV not found, continuing without rewrites: {path}")
        return rewrites, stats

    _log(cfg, f"Loading rewrites from {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        id_col = str(cfg.stage.rewrites.id_column)
        cap_col = str(cfg.stage.rewrites.captions_column)
        if id_col not in (reader.fieldnames or []) or cap_col not in (reader.fieldnames or []):
            raise ValueError(
                f"Rewrites CSV missing required columns. Required: {id_col}, {cap_col}. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            stats["rows_read"] += 1
            track_id = str(row.get(id_col, "")).strip()
            caps_raw = str(row.get(cap_col, "")).strip()
            if not track_id:
                continue
            captions = _split_captions(caps_raw, cfg)
            if not captions:
                continue
            primary = _pick_primary(captions, cfg)
            rewrites[track_id] = {
                "rewritten_captions": str(cfg.stage.rewrites.separator).join(captions),
                "rewritten_primary_caption": primary,
                "rewritten_captions_json": json.dumps(captions, ensure_ascii=True),
            }
            stats["rows_kept"] += 1

    _log(cfg, f"Loaded {stats['rows_kept']:,} rewrite rows")
    return rewrites, stats


def run_caption_join(cfg: DictConfig) -> Dict[str, object]:
    in_csv = Path(str(cfg.stage.io.input_manifest_csv))
    out_dir = Path(str(cfg.stage.io.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / str(cfg.stage.io.output_manifest_csv)
    report_path = out_dir / str(cfg.stage.io.report_file)
    tracker = StageTracker(
        cfg,
        "caption_join",
        title="Attach Caption Rewrites",
        subtitle=f"input={in_csv}",
        total_steps=4,
    )

    if not in_csv.exists():
        raise FileNotFoundError(f"Input manifest CSV not found: {in_csv}")

    tracker.step("Load rewrite table", detail=str(cfg.stage.rewrites.csv_path) if cfg.stage.rewrites.csv_path else "rewrites disabled or unset")
    rewrites, rewrite_stats = _load_rewrites(cfg)

    tracker.step("Read base manifest", detail=str(in_csv))
    with in_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        base_fields = list(reader.fieldnames or [])

    every_n = int(cfg.stage.progress.every_n_rows)
    total = len(rows)

    extra_fields = [
        "has_rewrite",
        "rewritten_captions",
        "rewritten_primary_caption",
        "rewritten_captions_json",
    ]
    fieldnames = base_fields + [c for c in extra_fields if c not in base_fields]

    matched = 0
    tracker.step("Join rewrites into rows", detail=f"{total:,} manifest rows")
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with rich_tqdm(cfg, total=total, desc="Join rows", unit="row") as progress:
            for i, row in enumerate(rows, start=1):
                tid = str(row.get("track_id", "")).strip()
                rw = rewrites.get(tid)
                if rw:
                    matched += 1
                    row["has_rewrite"] = "1"
                    row["rewritten_captions"] = rw["rewritten_captions"]
                    row["rewritten_primary_caption"] = rw["rewritten_primary_caption"]
                    row["rewritten_captions_json"] = rw["rewritten_captions_json"]
                    if bool(cfg.stage.behavior.overwrite_caption_fields):
                        row["caption"] = rw["rewritten_captions"]
                        if "primary_caption" in row:
                            row["primary_caption"] = rw["rewritten_primary_caption"]
                        if "captions_json" in row:
                            row["captions_json"] = rw["rewritten_captions_json"]
                else:
                    row["has_rewrite"] = "0"
                    row["rewritten_captions"] = ""
                    row["rewritten_primary_caption"] = ""
                    row["rewritten_captions_json"] = "[]"
                    if not bool(cfg.stage.behavior.keep_unmatched_rows):
                        progress.update(1)
                        continue

                writer.writerow(row)

                if every_n > 0 and i % every_n == 0:
                    _log(cfg, f"Processed rows: {i:,}")
                progress.update(1)

    report = {
        "stage": "caption_join",
        "input": {
            "input_manifest_csv": str(in_csv),
            "rewrites_csv": str(cfg.stage.rewrites.csv_path) if cfg.stage.rewrites.csv_path else None,
        },
        "counts": {
            "input_rows": total,
            "rewrite_rows_read": rewrite_stats["rows_read"],
            "rewrite_rows_kept": rewrite_stats["rows_kept"],
            "matched_rows": matched,
            "unmatched_rows": max(0, total - matched),
        },
        "outputs": {
            "output_manifest_csv": str(out_csv),
            "report": str(report_path),
        },
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)

    tracker.step("Write outputs", detail=f"manifest={out_csv.name}, report={report_path.name}")
    tracker.finish(f"matched {matched:,}/{total:,} rows")
    _log(cfg, f"Caption join complete. Matched {matched:,}/{total:,} rows")
    return report


@hydra.main(version_base=None, config_path=CONF_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    report = run_caption_join(cfg)
    print(json.dumps({"status": "ok", "stage": "caption_join", "outputs": report["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
