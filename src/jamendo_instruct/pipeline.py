from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import hydra
from jamendo_instruct.progress import StageTracker
from omegaconf import DictConfig, OmegaConf

from jamendo_instruct.run import STAGE_RUNNERS, run_stage

CONF_DIR = str(Path(__file__).resolve().parents[2] / "conf")
STAGE_ORDER = ["ingest", "caption_join", "lyrics", "structured_view", "embeddings", "neighborhood", "graph", "chains", "instructions", "validation", "relevance_pool"]


def _log(cfg: DictConfig, message: str) -> None:
    if bool(cfg.pipeline.log_stage_boundaries):
        print(f"[pipeline] {message}", flush=True)


def _stage_config_path(stage_name: str) -> Path:
    return Path(CONF_DIR) / "stage" / f"{stage_name}.yaml"


def _clone_cfg(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))


def _canonical_chain(cfg: DictConfig) -> List[str]:
    chain = ["ingest", "lyrics", "structured_view", "embeddings", "neighborhood", "graph", "chains", "instructions", "validation", "relevance_pool"]
    if bool(cfg.pipeline.include_caption_join):
        chain.insert(1, "caption_join")
    return chain


def _requested_chain(cfg: DictConfig) -> List[str]:
    chain = _canonical_chain(cfg)
    explicit = [str(x) for x in cfg.pipeline.run_stages if str(x)]
    if explicit:
        unknown = [name for name in explicit if name not in STAGE_ORDER]
        if unknown:
            raise ValueError(f"Unknown stage(s) in pipeline.run_stages: {unknown}")
        requested = [name for name in chain if name in explicit]
        if not requested:
            raise ValueError("pipeline.run_stages did not match any enabled stage in the current chain.")
        return requested

    start_stage = str(cfg.pipeline.start_stage)
    end_stage = str(cfg.pipeline.end_stage)
    start_idx = 0 if start_stage == "auto" else chain.index(start_stage)
    end_idx = chain.index(end_stage)
    if start_idx > end_idx:
        raise ValueError("pipeline.start_stage must not come after pipeline.end_stage")
    return chain[start_idx : end_idx + 1]


def _merge_stage_overrides(stage_name: str, base_stage_cfg: DictConfig, cfg: DictConfig) -> DictConfig:
    overrides = cfg.stage_overrides.get(stage_name)
    if overrides in (None, {}):
        return base_stage_cfg
    return OmegaConf.merge(base_stage_cfg, overrides)


def _build_stage_cfg(
    cfg: DictConfig,
    stage_name: str,
    known_outputs: Dict[str, str],
) -> DictConfig:
    stage_cfg = OmegaConf.load(_stage_config_path(stage_name))
    stage_cfg = _merge_stage_overrides(stage_name, stage_cfg, cfg)
    merged = _clone_cfg(cfg)
    merged.stage = stage_cfg

    if stage_name == "lyrics" and bool(cfg.pipeline.include_caption_join):
        caption_manifest = known_outputs.get("caption_join_manifest")
        if caption_manifest and Path(caption_manifest).exists():
            merged.stage.io.input_manifest_csv = caption_manifest

    if stage_name == "structured_view" and bool(cfg.pipeline.use_caption_join_output_for_structured_view):
        caption_manifest = known_outputs.get("caption_join_manifest")
        if caption_manifest and Path(caption_manifest).exists():
            merged.stage.io.input_manifest_csv = caption_manifest
    if stage_name == "structured_view":
        lyrics_manifest = known_outputs.get("lyrics_manifest")
        if lyrics_manifest and Path(lyrics_manifest).exists():
            merged.stage.io.input_manifest_csv = lyrics_manifest

    if str(cfg.pipeline.mode) == "from_scratch" and stage_name == "embeddings":
        if hasattr(merged.stage, "behavior"):
            merged.stage.behavior.overwrite_existing = True

    return merged


def _primary_output_path(stage_cfg: DictConfig) -> Path:
    stage_name = str(stage_cfg.stage.name)
    io = stage_cfg.stage.io
    if stage_name == "ingest":
        return Path(str(io.output_dir)) / str(io.normalized_manifest_file)
    if stage_name == "caption_join":
        return Path(str(io.output_dir)) / str(io.output_manifest_csv)
    if stage_name == "lyrics":
        return Path(str(io.output_dir)) / str(io.output_manifest_csv)
    if stage_name == "structured_view":
        return Path(str(io.output_dir)) / str(io.output_manifest_csv)
    if stage_name == "embeddings":
        return Path(str(io.output_dir)) / str(io.lookup_manifest_file)
    if stage_name == "neighborhood":
        return Path(str(io.output_dir)) / str(io.output_edges_csv)
    if stage_name == "graph":
        return Path(str(io.output_dir)) / str(io.output_adjacency_memmap_dir) / "metadata.json"
    if stage_name == "chains":
        return Path(str(io.output_dir)) / str(io.output_chains_jsonl)
    if stage_name == "instructions":
        return Path(str(io.output_dir)) / str(io.output_instructions_jsonl)
    if stage_name == "validation":
        return Path(str(io.output_dir)) / str(io.output_validated_jsonl)
    if stage_name == "relevance_pool":
        return Path(str(io.output_dir)) / str(io.output_pools_jsonl)
    raise ValueError(f"Unsupported stage for output path lookup: {stage_name}")


def _update_known_outputs(stage_name: str, stage_cfg: DictConfig, known_outputs: Dict[str, str], report: Dict[str, Any] | None) -> None:
    stage_output = _primary_output_path(stage_cfg)
    if stage_name == "caption_join":
        known_outputs["caption_join_manifest"] = str(stage_output)
    if stage_name == "lyrics":
        known_outputs["lyrics_manifest"] = str(stage_output)
    known_outputs[f"{stage_name}_primary_output"] = str(stage_output)
    if report:
        outputs = report.get("outputs", {})
        for key, value in outputs.items():
            if value is not None:
                known_outputs[f"{stage_name}:{key}"] = str(value)


def _resolve_resume_chain(cfg: DictConfig, chain: List[str], known_outputs: Dict[str, str]) -> List[str]:
    if str(cfg.pipeline.mode) != "resume":
        return chain
    if str(cfg.pipeline.start_stage) != "auto":
        return [name for name in chain]

    for offset, stage_name in enumerate(chain):
        stage_cfg = _build_stage_cfg(cfg, stage_name, known_outputs)
        output_path = _primary_output_path(stage_cfg)
        _update_known_outputs(stage_name, stage_cfg, known_outputs, report=None)
        if not output_path.exists():
            return chain[offset:]
    return []


@hydra.main(version_base=None, config_path=CONF_DIR, config_name="pipeline")
def main(cfg: DictConfig) -> None:
    requested_chain = _requested_chain(cfg)
    known_outputs: Dict[str, str] = {}
    chain = _resolve_resume_chain(cfg, requested_chain, known_outputs)
    tracker = StageTracker(
        cfg,
        "pipeline",
        title="Jamendo Instruct Pipeline",
        subtitle=f"run_name={cfg.runtime.run_name}",
        total_steps=len(chain),
    )

    if not chain:
        _log(cfg, f"All requested stage outputs already exist for run_name={cfg.runtime.run_name}. Nothing to do.")
        tracker.finish("all requested outputs already exist")
        print(json.dumps({"status": "ok", "run_name": str(cfg.runtime.run_name), "stages_run": []}, indent=2))
        return

    stage_reports: List[Dict[str, Any]] = []
    for stage_name in chain:
        stage_cfg = _build_stage_cfg(cfg, stage_name, known_outputs)
        output_path = _primary_output_path(stage_cfg)
        should_skip = str(cfg.pipeline.mode) == "resume" and output_path.exists()
        if should_skip:
            tracker.step(f"Skip {stage_name}", detail=f"output already exists at {output_path}")
            _log(cfg, f"Skipping stage '{stage_name}' because output already exists: {output_path}")
            _update_known_outputs(stage_name, stage_cfg, known_outputs, report=None)
            continue

        tracker.step(f"Run {stage_name}", detail=f"output -> {output_path}")
        _log(cfg, f"Starting stage '{stage_name}' for run_name={cfg.runtime.run_name}")
        report = run_stage(stage_cfg)
        stage_reports.append({"stage": stage_name, "outputs": report.get("outputs", {})})
        _update_known_outputs(stage_name, stage_cfg, known_outputs, report=report)
        _log(cfg, f"Completed stage '{stage_name}'")

    tracker.finish(f"completed {len(stage_reports):,} stage(s)")

    print(
        json.dumps(
            {
                "status": "ok",
                "run_name": str(cfg.runtime.run_name),
                "stages_run": stage_reports,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
