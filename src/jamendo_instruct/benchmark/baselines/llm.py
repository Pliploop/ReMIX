"""LLM baselines (docs/evaluation.md).

All use the LLM to turn (seed caption + instruction) into the *target* description,
then retrieve in different spaces:
  - LLMCaptionRewrite   : rewritten caption -> EmbeddingGemma -> corpus captions.
  - MuLanRewrite        : rewritten caption -> MuQ-MuLan text tower -> corpus audio.
  - AnalogySteering     : seed_audio + w*(MuLan(target) - MuLan(seed_caption)) -> corpus audio.
  - HybridScoreFusion   : a*audio_sim(seed) + (1-a)*text_sim(rewritten) -> corpus.
  - LLMPointwiseReranker: cheap first stage top-N, LLM grades each 0-6, re-rank.

Model via env REMIX_LLM (default Qwen/Qwen3.6-27B-FP8), tensor-parallel via
REMIX_LLM_TP. Needs a GPU + local HF cache (HF_HOME, HF_HUB_OFFLINE=1).
Each baseline sets `self.flops` (approx: 2*params*tokens for models, 2*Nq*N*D for search).
"""
from __future__ import annotations

import os
import re
from typing import Dict, List

import numpy as np

from .base import Baseline, Corpus, Query, batched_topk, matmul_flops, topk_scores
from .embedders import GemmaText, MuLanText, _l2

MODEL_ID = os.environ.get("REMIX_LLM", "Qwen/Qwen3.6-27B-FP8")
TP = int(os.environ.get("REMIX_LLM_TP", "1"))
# rough parameter counts for the FLOPs estimate (name -> params)
_PARAMS = {"Qwen/Qwen3.6-27B-FP8": 27e9}

REWRITE_SYS = ("You rewrite a music caption so it describes the track a listener wants after "
               "applying an edit instruction to a seed track. Output only the rewritten caption: "
               "one paragraph, concrete about genre, instrumentation, vocals, tempo, and mood. No preamble.")


_ENGINE = None


def _get_llm() -> "_LLM":
    """One vLLM engine per process: a second LLM() in-process would OOM the GPUs.
    FLOPs counter reset so each baseline reports only its own tokens."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = _LLM()
    _ENGINE.flops = 0
    return _ENGINE


def _seed_caption(c: Corpus, q: Query) -> str:
    return (c.meta.get(q.seed_clip_id, {}) or {}).get("caption", "")


def _exclude(c: Corpus, queries: List[Query]) -> List[int]:
    return [c.id2row.get(q.seed_clip_id, -1) for q in queries]


class _LLM:
    def __init__(self):
        from jamendo_instruct.llm_backends import build_vllm_offline_chat_model, decode_vllm_chat_completions
        self.ctx = build_vllm_offline_chat_model(
            model_id=MODEL_ID, tensor_parallel_size=TP, kv_cache_dtype="fp8",
            max_model_len=8192, gpu_memory_utilization=0.75)  # leave room for MuLan/Gemma encoders on the same GPU
        self._decode = decode_vllm_chat_completions
        self.n_params = _PARAMS.get(MODEL_ID, 27e9)
        self.flops = 0

    def chat(self, messages_batch, max_tokens: int, temperature: float = 0.0) -> List[str]:
        if not messages_batch:
            return []
        outs = self._decode(self.ctx, messages_batch=messages_batch,
                            max_tokens=max_tokens, temperature=temperature, top_p=1.0)
        tok = self.ctx.tokenizer
        n_tok = 0
        for msgs, out in zip(messages_batch, outs):
            n_tok += sum(len(tok(m["content"]).input_ids) for m in msgs) + len(tok(out or "").input_ids)
        self.flops += 2 * self.n_params * n_tok
        return outs

    def rewrite(self, c: Corpus, queries: List[Query]) -> List[str]:
        msgs = [[{"role": "system", "content": REWRITE_SYS},
                 {"role": "user", "content": f"Seed caption:\n{_seed_caption(c, q)}\n\nEdit instruction:\n{q.instruction}\n\nRewritten caption:"}]
                for q in queries]
        caps = self.chat(msgs, max_tokens=200)
        return [cap or q.instruction for cap, q in zip(caps, queries)]


class LLMCaptionRewrite(Baseline):
    name = "llm_caption_rewrite"

    def prepare(self, corpus: Corpus) -> None:
        self.c = corpus; self.emb = GemmaText(); self.llm = _get_llm()

    def rank_all(self, queries: List[Query], k: int) -> Dict[str, List[str]]:
        Q = self.emb.encode(self.llm.rewrite(self.c, queries))
        lists = batched_topk(Q, self.c.text, self.c.ids, k, _exclude(self.c, queries))
        self.flops = self.llm.flops + self.emb.flops + matmul_flops(len(queries), *self.c.text.shape)
        return {q.query_id: lst for q, lst in zip(queries, lists)}


class MuLanRewrite(Baseline):
    name = "mulan_rewrite"

    def prepare(self, corpus: Corpus) -> None:
        self.c = corpus; self.emb = MuLanText(); self.llm = _get_llm()

    def rank_all(self, queries: List[Query], k: int) -> Dict[str, List[str]]:
        Q = self.emb.encode(self.llm.rewrite(self.c, queries))          # (Nq,512) joint space
        lists = batched_topk(Q, self.c.audio, self.c.ids, k, _exclude(self.c, queries))
        self.flops = self.llm.flops + self.emb.flops + matmul_flops(len(queries), *self.c.audio.shape)
        return {q.query_id: lst for q, lst in zip(queries, lists)}


class AnalogySteering(Baseline):
    """seed_audio + w*(MuLan(target_desc) - MuLan(seed_caption)) -> corpus audio."""
    name = "analogy_steering"

    def __init__(self, w: float = 1.0):
        self.w = w

    def prepare(self, corpus: Corpus) -> None:
        self.c = corpus; self.emb = MuLanText(); self.llm = _get_llm()

    def rank_all(self, queries: List[Query], k: int) -> Dict[str, List[str]]:
        tgt = self.emb.encode(self.llm.rewrite(self.c, queries))
        src = self.emb.encode([_seed_caption(self.c, q) for q in queries])
        dim = self.c.audio.shape[1]
        A = np.stack([self.c.audio[self.c.id2row[q.seed_clip_id]] if q.seed_clip_id in self.c.id2row
                      else np.zeros(dim, np.float32) for q in queries])
        Q = _l2(A + self.w * (tgt - src))
        lists = batched_topk(Q, self.c.audio, self.c.ids, k, _exclude(self.c, queries))
        self.flops = self.llm.flops + self.emb.flops + matmul_flops(len(queries), *self.c.audio.shape)
        return {q.query_id: lst for q, lst in zip(queries, lists)}


class HybridScoreFusion(Baseline):
    """alpha * audio_sim(seed) + (1-alpha) * text_sim(rewritten caption)."""
    name = "hybrid_score_fusion"

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def prepare(self, corpus: Corpus) -> None:
        self.c = corpus; self.emb = GemmaText(); self.llm = _get_llm()

    def rank_all(self, queries: List[Query], k: int) -> Dict[str, List[str]]:
        Tq = self.emb.encode(self.llm.rewrite(self.c, queries))          # (Nq,768)
        dim = self.c.audio.shape[1]
        A = np.stack([self.c.audio[self.c.id2row[q.seed_clip_id]] if q.seed_clip_id in self.c.id2row
                      else np.zeros(dim, np.float32) for q in queries])
        S = self.alpha * (A @ self.c.audio.T) + (1 - self.alpha) * (Tq @ self.c.text.T)
        lists = topk_scores(S, self.c.ids, k, _exclude(self.c, queries))
        self.flops = (self.llm.flops + self.emb.flops
                      + matmul_flops(len(queries), *self.c.audio.shape)
                      + matmul_flops(len(queries), *self.c.text.shape))
        return {q.query_id: lst for q, lst in zip(queries, lists)}


class LLMPointwiseReranker(Baseline):
    name = "llm_pointwise_rerank"
    SYS = ("You grade how well a candidate track answers a composed music-retrieval query "
           "(a seed track plus an edit instruction). Reply with a single integer 0-6: "
           "6 exact, 5 strong, 4 good, 3 partial, 2 weak, 1 off-target, 0 unrelated. Integer only.")

    def __init__(self, first_stage: str = "seed_audio_nn", n: int = 50):
        self.first_stage = first_stage; self.n = n

    def prepare(self, corpus: Corpus) -> None:
        from . import get as get_baseline
        self.c = corpus
        self.first = get_baseline(self.first_stage)(); self.first.prepare(corpus)
        self.llm = _get_llm()

    @staticmethod
    def _score(text: str) -> float:
        m = re.search(r"-?\d+", text or "")
        return float(m.group()) if m else 0.0

    def rank_all(self, queries: List[Query], k: int) -> Dict[str, List[str]]:
        base = (self.first.rank_all(queries, max(k, self.n)) if hasattr(self.first, "rank_all")
                else {q.query_id: self.first.rank(q, max(k, self.n)) for q in queries})
        msgs, index = [], []
        for q in queries:
            for cid in base[q.query_id][:self.n]:
                cap = (self.c.meta.get(cid, {}) or {}).get("caption", "")
                msgs.append([{"role": "system", "content": self.SYS},
                             {"role": "user", "content": f"Seed caption:\n{_seed_caption(self.c, q)}\n\nEdit instruction:\n{q.instruction}\n\nCandidate caption:\n{cap}\n\nScore (0-6):"}])
                index.append((q.query_id, cid))
        scores = self.llm.chat(msgs, max_tokens=4)
        by_q: Dict[str, Dict[str, float]] = {}
        for (qid, cid), s in zip(index, scores):
            by_q.setdefault(qid, {})[cid] = self._score(s)
        out = {}
        for q in queries:
            head = base[q.query_id][:self.n]
            scored = sorted(head, key=lambda cid: (-by_q.get(q.query_id, {}).get(cid, 0.0), head.index(cid)))
            out[q.query_id] = (scored + base[q.query_id][self.n:])[:k]
        self.flops = self.llm.flops + getattr(self.first, "flops", 0)
        return out


REGISTRY = {b.name: b for b in [LLMCaptionRewrite, MuLanRewrite, AnalogySteering,
                                HybridScoreFusion, LLMPointwiseReranker]}
