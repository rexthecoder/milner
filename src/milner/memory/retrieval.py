"""Vector-store retrieval: embed every turn, return the top-k most similar to the question."""

from __future__ import annotations

from typing import List, Optional

from ..schema import Turn
from .base import MemoryBackend, MemoryContext
from .embedding import Embedder, build_embedder


class Retrieval(MemoryBackend):
    """Top-k retrieval over an in-memory vector store, one vector per turn.

    Retrieved turns are shown in their original order, not by score. Ties break on turn
    position so results are deterministic.
    """

    name = "retrieval"

    def __init__(self, top_k: int = 3, embedder: dict | Embedder | None = None, **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.embedder = embedder if isinstance(embedder, Embedder) else build_embedder(embedder)
        self._turns: List[Turn] = []
        self._vectors: List[List[float]] = []

    def reset(self) -> None:
        self._turns, self._vectors = [], []

    def add_turn(self, turn: Turn) -> None:
        self._turns.append(turn)
        self._vectors.append(self.embedder.embed([turn.content])[0])

    def retrieve(self, question: str) -> MemoryContext:
        if not self._turns:
            return MemoryContext("", [])
        q = self.embedder.embed([question])[0]
        scores = [sum(a * b for a, b in zip(q, v)) for v in self._vectors]
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        chosen: Optional[List[int]] = sorted(order[: self.top_k])
        turns = [self._turns[i] for i in chosen]
        return MemoryContext(self.render(turns), [t.turn_id for t in turns])
