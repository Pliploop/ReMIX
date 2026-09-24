"""Baseline registry (docs/evaluation.md).

`REGISTRY` maps name -> Baseline class. Embed/LLM/trained baselines are imported
lazily so a run of the trivial baselines needs no torch/transformers.
"""
from .trivial import REGISTRY as _TRIVIAL

REGISTRY = dict(_TRIVIAL)


def _load_embed():
    from .embed_baselines import REGISTRY as _E
    REGISTRY.update(_E)


def _load_llm():
    from .llm import REGISTRY as _L
    REGISTRY.update(_L)


def get(name: str):
    if name not in REGISTRY:
        _load_embed()
    if name not in REGISTRY:
        _load_llm()
    return REGISTRY[name]


def all_names(include_embed: bool = True, include_llm: bool = True):
    if include_embed:
        _load_embed()
    if include_llm:
        _load_llm()
    return list(REGISTRY)
