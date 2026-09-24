"""Baseline registry (docs/evaluation.md).

`REGISTRY` maps name -> Baseline class. Embed/LLM/trained baselines are imported
lazily so a run of the trivial baselines needs no torch/transformers.
"""
import os

from .trivial import REGISTRY as _TRIVIAL

REGISTRY = dict(_TRIVIAL)


def _load_embed():
    from .embed_baselines import REGISTRY as _E
    REGISTRY.update(_E)


def _load_llm():
    from .llm import REGISTRY as _L
    REGISTRY.update(_L)


def _load_trained():
    from remix_c.model import RemixCBaseline  # checkpoint from $REMIX_C_CKPT (run_remix_b.py --ckpt)
    REGISTRY[RemixCBaseline.name] = RemixCBaseline


def get(name: str):
    for load in (_load_embed, _load_llm, _load_trained):
        if name in REGISTRY:
            break
        load()
    return REGISTRY[name]


def all_names(include_embed: bool = True, include_llm: bool = True):
    if include_embed:
        _load_embed()
    if include_llm:
        _load_llm()
    if os.environ.get("REMIX_C_CKPT"):  # trained baselines only when a checkpoint is given
        _load_trained()
    return list(REGISTRY)
