#!/usr/bin/env python
"""Grade instruction variants and select the benchmark cut (variant fallback).

Two decoupled phases (see ``docs/validation_gate.md``):

  grade   instructions + llm_ratings  -> instruction_grades.jsonl   (all variants, nothing cut)
  select  instruction_grades.jsonl    -> validated_instructions.jsonl (one best-passing variant
                                         per step; contextual chains truncated at the first step
                                         with no passing variant). This is what relevance_pool reads.

`--phase both` (default) runs grade then select. `--phase select` re-cuts an
existing grades file with a different policy without re-judging. CPU-only.

  PYTHONPATH=src python scripts/build_validated_instructions.py \
    --run-root /gpfs/.../music4all_instruct/music4all_v1 \
    --instructions-folder instructions_axis_focused_5 \
    --threshold 4 --variant-select best --contextual-policy truncate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from jamendo_instruct.demo.human_validation_app import _llm_rating_records  # noqa: E402
from jamendo_instruct.validation_gate import GateConfig, grade_records, select_chain_variants  # noqa: E402


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-root")
    p.add_argument("--instructions-folder", default="instructions_axis_focused_5")
    p.add_argument("--instructions-jsonl")
    p.add_argument("--ratings-dir", help="Directory with llm_ratings*.jsonl (else <folder>/validation).")
    p.add_argument("--out-dir", help="Where to write grades/validated jsonl (else <ratings-dir>).")
    p.add_argument("--phase", default="both", choices=["grade", "select", "both"])
    p.add_argument("--instruction-field", default="history_unaware_instruction")
    p.add_argument("--threshold", type=float, default=4.0)
    p.add_argument("--chain-aggregate", default="min", choices=["min", "mean"])
    p.add_argument("--contextual-policy", default="truncate", choices=["truncate", "drop"])
    p.add_argument("--variant-select", default="best", choices=["best", "first"])
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.instructions_jsonl:
        instr_path = Path(args.instructions_jsonl).expanduser()
    elif args.run_root:
        instr_path = Path(args.run_root).expanduser() / args.instructions_folder / "chain_step_instructions.jsonl"
    else:
        raise SystemExit("Provide --instructions-jsonl or --run-root.")
    ratings_dir = Path(args.ratings_dir).expanduser() if args.ratings_dir else instr_path.parent / "validation"
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else ratings_dir
    grades_path = out_dir / "instruction_grades.jsonl"
    validated_path = out_dir / "validated_instructions.jsonl"

    config = GateConfig(
        accept_threshold=float(args.threshold),
        chain_aggregate=str(args.chain_aggregate),
        contextual_policy=str(args.contextual_policy),
        variant_select=str(args.variant_select),
        instruction_field=str(args.instruction_field),
    )
    summary: Dict[str, Any] = {"config": _cfg(config)}

    if args.phase in ("grade", "both"):
        if not instr_path.exists():
            raise SystemExit(f"Instructions not found: {instr_path}")
        ratings = _llm_rating_records(ratings_dir)
        if not ratings:
            raise SystemExit(f"No llm_ratings*.jsonl in {ratings_dir}. Run the graded judge first.")
        graded, grade_report = grade_records(list(_iter_jsonl(instr_path)), ratings, config)
        _write_jsonl(grades_path, graded)
        (out_dir / "instruction_grades_report.json").write_text(json.dumps(grade_report, indent=2), encoding="utf-8")
        summary["grades"] = {"path": str(grades_path), **grade_report["counts"]}

    if args.phase in ("select", "both"):
        if not grades_path.exists():
            raise SystemExit(f"Grades file not found: {grades_path}. Run --phase grade first.")
        graded = list(_iter_jsonl(grades_path))
        selected, select_report = select_chain_variants(graded, config)
        _write_jsonl(validated_path, selected)
        (out_dir / "validation_gate_report.json").write_text(json.dumps(select_report, indent=2), encoding="utf-8")
        c = select_report["counts"]
        summary["selected"] = {
            "path": str(validated_path),
            **c,
            "accept_rate": round(c["accepted"] / max(1, c["steps"]), 3),
        }

    print(json.dumps(summary, indent=2))


def _cfg(config: GateConfig) -> Dict[str, Any]:
    return {"accept_threshold": config.accept_threshold, "chain_aggregate": config.chain_aggregate,
            "contextual_policy": config.contextual_policy, "variant_select": config.variant_select,
            "instruction_field": config.instruction_field}


if __name__ == "__main__":
    main()
