from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig

from jamendo_instruct.stages.caption_join import run_caption_join
from jamendo_instruct.stages.chains import run_chains
from jamendo_instruct.stages.embeddings import run_embeddings
from jamendo_instruct.stages.graph import run_graph
from jamendo_instruct.stages.ingest import run_ingest
from jamendo_instruct.stages.instructions import run_instructions
from jamendo_instruct.stages.lyrics import run_lyrics
from jamendo_instruct.stages.neighborhood import run_neighborhood
from jamendo_instruct.stages.relevance_pool import run_relevance_pool
from jamendo_instruct.stages.structured_view import run_structured_view
from jamendo_instruct.stages.validation import run_validation

CONF_DIR = str(Path(__file__).resolve().parents[2] / "conf")

STAGE_RUNNERS = {
    "ingest": run_ingest,
    "caption_join": run_caption_join,
    "lyrics": run_lyrics,
    "chains": run_chains,
    "structured_view": run_structured_view,
    "embeddings": run_embeddings,
    "neighborhood": run_neighborhood,
    "graph": run_graph,
    "instructions": run_instructions,
    "validation": run_validation,
    "relevance_pool": run_relevance_pool,
}


def run_stage(cfg: DictConfig) -> dict:
    stage_name = str(cfg.stage.name)
    try:
        runner = STAGE_RUNNERS[stage_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported stage: {stage_name}") from exc
    return runner(cfg)


@hydra.main(version_base=None, config_path=CONF_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    stage_name = str(cfg.stage.name)
    report = run_stage(cfg)
    print(json.dumps({"status": "ok", "stage": stage_name, "outputs": report.get("outputs", {})}, indent=2))


if __name__ == "__main__":
    main()
