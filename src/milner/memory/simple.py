"""The two baselines that bracket every other strategy."""

from __future__ import annotations

from typing import List, Optional

from ..schema import Turn
from .base import MemoryBackend, MemoryContext


class NoMemory(MemoryBackend):
    """Lower bound: the model sees only the question."""

    name = "none"

    def reset(self) -> None:
        pass

    def add_turn(self, turn: Turn) -> None:
        pass

    def retrieve(self, question: str) -> MemoryContext:
        return MemoryContext(text="", retrieved_turn_ids=[])


class FullContext(MemoryBackend):
    """Upper-bound-on-recall baseline: put the whole conversation in the prompt.

    ``max_turns`` keeps only the most recent turns, which models a finite context window.
    """

    name = "full"

    def __init__(self, max_turns: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.max_turns = max_turns
        self._turns: List[Turn] = []

    def reset(self) -> None:
        self._turns = []

    def add_turn(self, turn: Turn) -> None:
        self._turns.append(turn)

    def retrieve(self, question: str) -> MemoryContext:
        turns = self._turns[-self.max_turns :] if self.max_turns else self._turns
        return MemoryContext(self.render(turns), [t.turn_id for t in turns])
