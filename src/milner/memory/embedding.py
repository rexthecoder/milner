"""Text embedders for the retrieval backend."""

from __future__ import annotations

import math
import zlib
from abc import ABC, abstractmethod
from typing import Dict, List

from ..text import content_words


class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return one L2-normalised vector per text."""


class HashingEmbedder(Embedder):
    """Dependency-free bag-of-words embedding via feature hashing.

    Uses CRC32 rather than ``hash()`` so vectors are identical across processes and
    machines, which keeps runs reproducible. Good enough to exercise the retrieval pipeline
    offline; use :class:`SentenceTransformerEmbedder` for real experiments.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        counts: Dict[str, int] = {}
        for w in content_words(text):
            counts[w] = counts.get(w, 0) + 1
        for w, c in counts.items():
            vec[zlib.crc32(w.encode()) % self.dim] += float(c)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError("install the extra: pip install 'milner[embeddings]'") from exc
        self._model = SentenceTransformer(model)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()


def build_embedder(spec: dict | None) -> Embedder:
    spec = dict(spec or {"type": "hashing"})
    kind = spec.pop("type")
    if kind == "hashing":
        return HashingEmbedder(**spec)
    if kind in ("sentence-transformers", "sentence_transformers"):
        return SentenceTransformerEmbedder(**spec)
    raise ValueError(f"unknown embedder type {kind!r}")
