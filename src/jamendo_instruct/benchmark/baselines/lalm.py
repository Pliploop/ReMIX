"""Large audio-language model (LALM) baselines: the model listens to the clips.

  LALMDescribe (lalm_describe): the LALM hears the seed audio, reads the instruction, and
      writes a description of the wanted track -> EmbeddingGemma -> corpus captions.
      Counterpart of llm_caption_rewrite that listens instead of reading the seed caption.
  LALMHybrid   (lalm_hybrid)  : alpha * audio_sim(seed) + (1 - alpha) * text_sim(description).
  LALMRerank   (lalm_rerank)  : first stage lalm_hybrid top-N; the LALM hears the seed and the
      candidate, reads the instruction, grades 0-6; re-rank.

Model via env REMIX_LALM (default Qwen/Qwen3-Omni-30B-A3B-Instruct; also nvidia/music-flamingo-hf and
moonshotai/Kimi-Audio-7B-Instruct), tensor parallel via REMIX_LALM_TP. Run each model with
`run_remix_b.py --suffix` so its rows get their own names. Audio is decoded from the catalogue manifest under $REMIX_RUN_ROOT (set by
run_remix_b.py) at 16 kHz mono, 30 s per clip. Descriptions are cached to
<run>/instructions_axis_focused_5/benchmark/lalm_descriptions_<model>.json so the three
baselines, and re-runs, share one generation pass. FLOPs: 2 * active params * tokens, where
tokens include the audio tokens the model reads (the audio encoder is not counted separately).
"""
from __future__ import annotations

import json
import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List

import numpy as np

from .base import Baseline, Corpus, Query, matmul_flops, topk_scores
from .embedders import GemmaText

MODEL_ID = os.environ.get("REMIX_LALM", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
TP = int(os.environ.get("REMIX_LALM_TP", "2"))
_ACTIVE_PARAMS = {"Qwen/Qwen3-Omni-30B-A3B-Instruct": 3.3e9, "nvidia/music-flamingo-hf": 8.3e9,
                  "moonshotai/Kimi-Audio-7B-Instruct": 7.6e9}
KIMI = "kimi-audio" in MODEL_ID.lower()
# Audio Flamingo and Kimi-Audio accept one clip per prompt: several clips are joined with 1 s of silence
# and the prompt says so (a documented deviation for these two models)
ONE_CLIP = KIMI or "flamingo" in MODEL_ID.lower()
JOINED_NOTE = "The audio contains the seed track, then one second of silence, then the candidate track.\n"
SR = 16_000
CLIP_SEC = 30
CHUNK = 256          # prompts per vLLM call (bounds host memory for decoded audio)

DESCRIBE_SYS = ("You listen to a music track and read an edit instruction. Describe the track the listener "
                "wants after applying the instruction to what you hear. Output only the description: one "
                "paragraph, concrete about genre, instrumentation, vocals, tempo, and mood. No preamble.")
RERANK_SYS = ("You grade how well a candidate track answers a composed music-retrieval query: a seed track "
              "plus an edit instruction. Listen to both tracks. Reply with a single integer 0-6: 6 exact, "
              "5 strong, 4 good, 3 partial, 2 weak, 1 off-target, 0 unrelated. Integer only.")


# ---------------------------------------------------------------- audio
def _manifest() -> Dict[str, tuple]:
    import pandas as pd
    root = Path(os.environ["REMIX_RUN_ROOT"])
    df = pd.read_csv(root / "ingest" / "normalized_track_manifest.csv",
                     usecols=["clip_id", "file_path", "start_time", "end_time"]).drop_duplicates("clip_id")
    return {r.clip_id: (r.file_path, float(r.start_time)) for r in df.itertuples()}


def _decode(args) -> np.ndarray:
    path, start = args
    import soundfile as sf
    import torch
    import torchaudio
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        f.seek(int(start * sr))
        x = f.read(int(CLIP_SEC * sr), dtype="float32", always_2d=True).mean(1)
    return torchaudio.functional.resample(torch.from_numpy(x), sr, SR).numpy().astype(np.float16)


class _Audio:
    """clip_id -> 16 kHz mono float16 waveform, decoded in parallel and memoised."""

    def __init__(self):
        self.where, self.cache = _manifest(), {}

    def load(self, clip_ids: List[str]) -> None:
        todo = [c for c in dict.fromkeys(clip_ids) if c not in self.cache]
        # spawn, not fork: the parent may already hold a vLLM/CUDA engine
        with ProcessPoolExecutor(max_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")),
                                 mp_context=multiprocessing.get_context("spawn")) as ex:
            for cid, wav in zip(todo, ex.map(_decode, [self.where[c] for c in todo], chunksize=16)):
                self.cache[cid] = wav

    def __getitem__(self, cid: str) -> np.ndarray:
        return self.cache[cid].astype(np.float32)


# ---------------------------------------------------------------- engine
class _LALM:
    def __init__(self):
        from vllm import LLM
        # ~61 GB of bf16 weights: on 2x A100-40 only a few GB remain for KV cache and audio encoding
        self.llm = LLM(model=MODEL_ID, tensor_parallel_size=TP, max_model_len=8192, gpu_memory_utilization=float(os.environ.get("REMIX_LALM_MEM", "0.85")),
                       limit_mm_per_prompt={"audio": 1 if ONE_CLIP else 2}, max_num_seqs=16, trust_remote_code=KIMI)
        if KIMI:   # Kimi-Audio has no HF chat processor; vLLM ships its tokenizer
            from vllm.tokenizers import cached_get_tokenizer
            from vllm.tokenizers.kimi_audio import KimiAudioTokenizer
            self.tokenizer = cached_get_tokenizer(MODEL_ID, tokenizer_cls=KimiAudioTokenizer, trust_remote_code=True)
        else:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.n_active = _ACTIVE_PARAMS.get(MODEL_ID, 3.3e9)
        self.flops = 0

    def _prompt(self, system: str, auds: List[np.ndarray], text: str) -> dict:
        """Audio clips first, then the text, in each model's own chat format."""
        if ONE_CLIP and len(auds) > 1:
            gap = np.zeros(SR, dtype=np.float32)
            joined = auds[0]
            for a in auds[1:]:
                joined = np.concatenate([joined, gap, a])
            auds, text = [joined], JOINED_NOTE + text
        mm = {"audio": [(a, SR) for a in auds]}
        if KIMI:
            from vllm.inputs import TokensPrompt
            ph = "<|im_media_begin|><|im_kimia_text_blank|><|im_media_end|>"   # KimiAudio.AUDIO_PLACEHOLDER
            prompt = (f"<|im_kimia_user_msg_start|>{system}\n\n{ph}\n{text}"
                      f"<|im_msg_end|><|im_kimia_assistant_msg_start|>")
            return TokensPrompt(prompt_token_ids=self.tokenizer.encode(prompt), multi_modal_data=mm)
        if ONE_CLIP:   # Audio Flamingo: its chat template drops audio items, so the token goes in the text
            user = [{"type": "text", "text": self.processor.audio_token + "\n" + text}]
        else:
            user = [{"type": "audio", "audio": "x"} for _ in auds] + [{"type": "text", "text": text}]
        msgs = [{"role": "system", "content": [{"type": "text", "text": system}]}, {"role": "user", "content": user}]
        return {"prompt": self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False),
                "multi_modal_data": mm}

    def generate(self, system: str, audios: List[List[np.ndarray]], texts: List[str], max_tokens: int) -> List[str]:
        """One prompt per (audio list, text); audio precedes the text in the user turn."""
        from vllm import SamplingParams
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        outs = []
        for i in range(0, len(texts), CHUNK):
            prompts = [self._prompt(system, auds, text) for auds, text in zip(audios[i:i + CHUNK], texts[i:i + CHUNK])]
            for o in self.llm.generate(prompts, params, use_tqdm=False):
                outs.append(o.outputs[0].text.strip())
                self.flops += 2 * self.n_active * (len(o.prompt_token_ids) + len(o.outputs[0].token_ids))
        return outs


_ENGINE = None
_AUDIO = None


def _engine() -> _LALM:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = _LALM()
    _ENGINE.flops = 0
    return _ENGINE


def _audio() -> _Audio:
    global _AUDIO
    if _AUDIO is None:
        _AUDIO = _Audio()
    return _AUDIO


def _cache_path() -> Path:
    slug = MODEL_ID.replace("/", "__")
    return Path(os.environ["REMIX_RUN_ROOT"]) / "instructions_axis_focused_5" / "benchmark" / f"lalm_descriptions_{slug}.json"


def describe(queries: List[Query]) -> tuple[List[str], float]:
    """Descriptions of the wanted track for each query (cached on disk) and their FLOPs."""
    path = _cache_path()
    cache = json.loads(path.read_text()) if path.exists() else {"flops_per_query": {}, "text": {}}
    todo = [q for q in queries if q.query_id not in cache["text"]]
    if todo:
        aud = _audio()
        aud.load([q.seed_clip_id for q in todo])        # decode before the engine starts
        eng = _engine()
        texts = eng.generate(DESCRIBE_SYS, [[aud[q.seed_clip_id]] for q in todo],
                             [f"Edit instruction: {q.instruction}\n\nDescription of the wanted track:" for q in todo],
                             max_tokens=200)
        per_q = eng.flops / max(1, len(todo))
        for q, t in zip(todo, texts):
            cache["text"][q.query_id] = t or q.instruction
            cache["flops_per_query"][q.query_id] = per_q
        path.write_text(json.dumps(cache))
    return ([cache["text"][q.query_id] for q in queries],
            sum(cache["flops_per_query"].get(q.query_id, 0.0) for q in queries))


def _exclude(c: Corpus, queries: List[Query]) -> List[int]:
    return [c.id2row.get(q.seed_clip_id, -1) for q in queries]


# ---------------------------------------------------------------- baselines
class LALMDescribe(Baseline):
    name = "lalm_describe"

    def prepare(self, corpus: Corpus) -> None:
        self.c, self.emb = corpus, GemmaText()

    def rank_all(self, queries: List[Query], k: int) -> Dict[str, List[str]]:
        texts, gen_flops = describe(queries)
        Tq = self.emb.encode(texts)
        lists = topk_scores(Tq @ self.c.text.T, self.c.ids, k, _exclude(self.c, queries))
        self.flops = gen_flops + self.emb.flops + matmul_flops(len(queries), *self.c.text.shape)
        return {q.query_id: lst for q, lst in zip(queries, lists)}


class LALMHybrid(Baseline):
    """alpha * audio_sim(seed) + (1 - alpha) * text_sim(LALM description)."""
    name = "lalm_hybrid"

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def prepare(self, corpus: Corpus) -> None:
        self.c, self.emb = corpus, GemmaText()

    def scores(self, queries: List[Query]) -> np.ndarray:
        texts, gen_flops = describe(queries)
        Tq = self.emb.encode(texts)
        dim = self.c.audio.shape[1]
        A = np.stack([self.c.audio[self.c.id2row[q.seed_clip_id]] if q.seed_clip_id in self.c.id2row
                      else np.zeros(dim, np.float32) for q in queries])
        self.flops = (gen_flops + self.emb.flops + matmul_flops(len(queries), *self.c.audio.shape)
                      + matmul_flops(len(queries), *self.c.text.shape))
        return self.alpha * (A @ self.c.audio.T) + (1 - self.alpha) * (Tq @ self.c.text.T)

    def rank_all(self, queries: List[Query], k: int) -> Dict[str, List[str]]:
        lists = topk_scores(self.scores(queries), self.c.ids, k, _exclude(self.c, queries))
        return {q.query_id: lst for q, lst in zip(queries, lists)}


class LALMRerank(Baseline):
    """First stage lalm_hybrid top-N; the LALM grades each (seed, candidate) pair 0-6 by listening."""
    name = "lalm_rerank"

    def __init__(self, n: int = 20):
        self.n = n

    def prepare(self, corpus: Corpus) -> None:
        self.c = corpus
        self.first = LALMHybrid()
        self.first.prepare(corpus)

    @staticmethod
    def _grade(text: str) -> float:
        m = re.search(r"\d", text or "")
        return float(m.group()) if m else 0.0

    def rank_all(self, queries: List[Query], k: int) -> Dict[str, List[str]]:
        base = self.first.rank_all(queries, max(k, self.n))
        first_flops = self.first.flops
        eng, aud = _engine(), _audio()
        aud.load([q.seed_clip_id for q in queries] + [c for q in queries for c in base[q.query_id][:self.n]])
        pairs = [(q, cid) for q in queries for cid in base[q.query_id][:self.n]]
        grades = eng.generate(RERANK_SYS, [[aud[q.seed_clip_id], aud[cid]] for q, cid in pairs],
                              [f"The first audio is the seed track, the second is the candidate.\n"
                               f"Edit instruction: {q.instruction}\n\nScore (0-6):" for q, _ in pairs],
                              max_tokens=4)
        by_q: Dict[str, Dict[str, float]] = {}
        for (q, cid), g in zip(pairs, grades):
            by_q.setdefault(q.query_id, {})[cid] = self._grade(g)
        out = {}
        for q in queries:
            head = base[q.query_id][:self.n]
            s = by_q.get(q.query_id, {})
            out[q.query_id] = (sorted(head, key=lambda cid: (-s.get(cid, 0.0), head.index(cid)))
                               + base[q.query_id][self.n:])[:k]
        self.flops = first_flops + eng.flops
        return out


REGISTRY = {b.name: b for b in [LALMDescribe, LALMHybrid, LALMRerank]}
