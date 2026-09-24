"""Text embedders for the composed/LLM baselines, matching the pipeline's stages.

- GemmaText  : google/embeddinggemma-300m, mean-pooled + L2 (same recipe as the
               corpus text embeddings; input is lower-cased like `normalized_caption`).
- MuLanText  : OpenMuQ/MuQ-MuLan-large text tower -> the joint audio-text space,
               so instruction text can be searched against corpus *audio* vectors.

Models are read from the local HF cache (set HF_HUB_OFFLINE=1, HF_HOME=<cache>).
"""
from __future__ import annotations

import os
from typing import List

import numpy as np


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


class GemmaText:
    dim = 768

    def __init__(self, model_id: str = "google/embeddinggemma-300m", device: str = "cuda",
                 max_length: int = 512):
        import torch
        from transformers import AutoModel, AutoTokenizer
        tok = os.environ.get("HF_TOKEN") or None
        self.torch = torch
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=tok)
        self.model = AutoModel.from_pretrained(model_id, token=tok).to(device).eval()

    def encode(self, texts: List[str], batch: int = 256) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                chunk = [t.lower() for t in texts[i:i + batch]]
                enc = self.tokenizer(chunk, padding=True, truncation=True,
                                     max_length=self.max_length, return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                hs = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1)
                emb = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)
                emb = torch.nn.functional.normalize(emb, dim=-1)
                out.append(emb.cpu().numpy().astype(np.float32))
        return _l2(np.concatenate(out))


class MuLanText:
    dim = 512

    def __init__(self, model_id: str = "OpenMuQ/MuQ-MuLan-large", device: str = "cuda"):
        import torch
        from muq import MuQMuLan
        self.torch = torch
        self.device = device
        self.model = MuQMuLan.from_pretrained(model_id).to(device).eval()

    def encode(self, texts: List[str], batch: int = 64) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                emb = self.model(texts=list(texts[i:i + batch]))
                out.append(emb.detach().cpu().numpy().astype(np.float32))
        return _l2(np.concatenate(out))
