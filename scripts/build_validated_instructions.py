#!/usr/bin/env python
"""Build ``validated_instructions.jsonl`` from graded rubric ratings.

This is the **instruction-validity** gate: it turns the graded LLM rubric
ratings (``llm_ratings*.jsonl``, produced by ``llm_validation_judge.py``) into a
per-step accept decision + chain-coherence rule, and writes the record schema
the ``relevance_pool`` stage already consumes (``validation.accepted``).

It is a distinct, upstream question from ``relevance_pool``'s candidate grading:
this decides *which (source -> target) instructions are good enough to be
benchmark queries*; ``relevance_pool`` then decides *which candidate tracks are
relevant ground truth* for each retained query. See ``docs/validation_gate.md``.

CPU-only; no GPU or model load. Example:

  PYTHONPATH=src python scripts/build_validated_instructions.py \
    --run-root /gpfs/.../music4all_instruct/music4all_v1 \
    --instructions-folder instructions_axis_focused_5 \
    --threshold 4 --contextual-policy truncate --chain-aggregate min
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
from jamendo_instruct.validation_gate import GateConfig, build_validated_records  # noqa: E402


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-root", help="Run artifact root (used to derive default paths).")
    p.add_argument("--instructions-folder", default="instructions_axis_focused_5")
    p.add_argument("--instructions-jsonl", help="Explicit chain_step_instructions.jsonl (else <run-root>/<folder>/...).")
    p.add_argument("--ratings-dir", help="Directory with llm_ratings*.jsonl (else <folder>/validation).")
    p.add_argument("--output", help="Output validated_instructions.jsonl (else <ratings-dir>/validated_instructions.jsonl).")
    p.add_argument("--instruction-field", default="history_unaware_instruction")
    p.add_argument("--threshold", type=float, default=4.0, help="overall_validity score >= threshold accepts.")
    p.add_argument("--chain-aggregate", default="min", choices=["min", "mean"])
    p.add_argument("--contextual-policy", default="truncate", choices=["truncate", "drop", "per_step"])
    p.add_argument("--unrated-policy", default="reject", choices=["reject", "pass"])
    p.add_argument("--primary-track", default="standalone", choices=["standalone", "contextual"])
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.instructions_jsonl:
        instr_path = Path(args.instructions_jsonl).expanduser()
    elif args.run_root:
        instr_path = Path(args.run_root).expanduser() / args.instructions_folder / "chain_step_instructions.jsonl"
    else:
        raise SystemExit("Provide --instructions-jsonl or --run-root.")
    if args.ratings_dir:
        ratings_dir = Path(args.ratings_dir).expanduser()
    else:
        ratings_dir = instr_path.parent / "validation"
    out_path = Path(args.output).expanduser() if args.output else ratings_dir / "validated_instructions.jsonl"

    if not instr_path.exists():
        raise SystemExit(f"Instructions not found: {instr_path}")
    ratings = _llm_rating_records(ratings_dir)
    if not ratings:
        raise SystemExit(f"No llm_ratings*.jsonl found in {ratings_dir}. Run the graded judge first.")

    instructions = list(_iter_jsonl(instr_path))
    config = GateConfig(
        accept_threshold=float(args.threshold),
        chain_aggregate=str(args.chain_aggregate),
        contextual_policy=str(args.contextual_policy),
        unrated_policy=str(args.unrated_policy),
        primary_track=str(args.primary_track),
        instruction_field=str(args.instruction_field),
    )
    validated, report = build_validated_records(instructions, ratings, config)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in validated:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
    (out_path.parent / "validation_gate_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    c = report["counts"]
    print(json.dumps({
        "output": str(out_path),
        "instructions": c["total"],
        "rated": c["rated"],
        "unrated": c["unrated"],
        "accepted": c["accepted"],
        "accept_rate_of_rated": round(c["accepted"] / max(1, c["rated"]), 3),
        "config": report["config"],
    }, indent=2))


if __name__ == "__main__":
    main()
