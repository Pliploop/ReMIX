"""LLM baselines (docs/evaluation.md).

- LLMCaptionRewrite : caption + instruction -> the LLM writes the *target* caption
                      -> EmbeddingGemma -> retrieve corpus captions.
- LLMPointwiseReranker : a cheap first-stage retrieves top-N, then the LLM grades
                      each candidate against (caption + instruction) and re-ranks.

Uses the pipeline's offline vLLM helpers. Model via env REMIX_LLM
(default Qwen/Qwen3.6-27B-FP8), tensor-parallel via REMIX_LLM_TP. Needs a GPU and
the local HF cache (HF_HOME, HF_HUB_OFFLINE=1).
"""
from __future__ import annotations

import os
import re
from typing import Dict, List

from .base import Baseline, Corpus, Query, batched_topk
from .embedders import GemmaText

MODEL_ID = os.environ.get("REMIX_LLM", "Qwen/Qwen3.6-27B-FP8")
TP = int(os.environ.get("REMIX_LLM_TP", "1"))


class _LLM:
    def __init__(self):
        from jamendo_instruct.llm_backends import build_vllm_offline_chat_model, decode_vllm_chat_completions
        self.ctx = build_vllm_offline_chat_model(
            MODEL_ID, tensor_parallel_size=TP, kv_cache_dtype="fp8",
            max_model_len=8192, gpu_memory_utilization=0.85)
        self._decode = decode_vllm_chat_completions

    def chat(self, messages_batch, max_tokens: int, temperature: float = 0.0) -> List[str]:
        if not messages_batch:
            return []
        return self._decode(self.ctx, messages_batch=messages_batch,
                            max_tokens=max_tokens, temperature=temperature, top_p=1.0)


def _seed_caption(c: Corpus, q: Query) -> str:
    return (c.meta.get(q.seed_clip_id, {}) or {}).get("caption", "")


def _exclude(c: Corpus, queries: List[Query]) -> List[int]:
    return [c.id2row.get(q.seed_clip_id, -1) for q in queries]


class LLMCaptionRewrite(Baseline):
    name = "llm_caption_rewrite"
    SYS = ("You rewrite a music caption so it describes the track a listener wants "
           "after applying an edit instruction to a seed track. Output only the rewritten "
           "caption: one paragraph, concrete about genre, instrumentation, vocals, tempo, "
           "and mood. No preamble.")

    def prepare(self, corpus: Corpus) -> None:
        self.c = corpus
        self.emb = GemmaText()
        self.llm = _LLM()

    def rank_all(self, queries: List[Query], k: int) -> Dict[str, List[str]]:
        msgs = [[{"role": "system", "content": self.SYS},
                 {"role": "user", "content": f"Seed caption:\n{_seed_caption(self.c, q)}\n\nEdit instruction:\n{q.instruction}\n\nRewritten caption:"}]
                for q in queries]
        caps = self.llm.chat(msgs, max_tokens=200)
        caps = [c or q.instruction for c, q in zip(caps, queries)]   # fall back to instruction if empty
        Q = self.emb.encode(caps)
        lists = batched_topk(Q, self.c.text, self.c.ids, k, _exclude(self.c, queries))
        return {q.query_id: lst for q, lst in zip(queries, lists)}


class LLMPointwiseReranker(Baseline):
    name = "llm_pointwise_rerank"
    SYS = ("You grade how well a candidate track answers a composed music-retrieval query "
           "(a seed track plus an edit instruction). Reply with a single integer 0-6: "
           "6 exact, 5 strong, 4 good, 3 partial, 2 weak, 1 off-target, 0 unrelated. Integer only.")

    def __init__(self, first_stage: str = "seed_audio_nn", n: int = 50):
        self.first_stage = first_stage
        self.n = n

    def prepare(self, corpus: Corpus) -> None:
        from . import get as get_baseline
        self.c = corpus
        self.first = get_baseline(self.first_stage)()
        self.first.prepare(corpus)
        self.llm = _LLM()

    @staticmethod
    def _score(text: str) -> float:
        m = re.search(r"-?\d+", text or "")
        return float(m.group()) if m else 0.0

    def rank_all(self, queries: List[Query], k: int) -> Dict[str, List[str]]:
        base = (self.first.rank_all(queries, max(k, self.n))
                if hasattr(self.first, "rank_all")
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
        return out


REGISTRY = {b.name: b for b in [LLMCaptionRewrite, LLMPointwiseReranker]}
