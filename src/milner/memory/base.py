"""The interface every memory strategy implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from ..schema import Turn
from ..usage import Usage


@dataclass
class MemoryContext:
    """What a strategy hands to the model for one question.

    ``retrieved_turn_ids`` is the set of source turns the context draws on verbatim, used
    for the retrieval hit rate. Use ``None`` for strategies that do not keep turn identity
    (for example summaries), which excludes them from that metric.
    """

    text: str
    retrieved_turn_ids: Optional[List[str]] = None


class MemoryBackend(ABC):
    """Ingest a conversation turn by turn, then produce context for a question.

    A backend that calls a model while ingesting (summarisation, say) must charge that to
    ``self.usage`` so the cost shows up in the results.
    """

    name = "memory"

    def __init__(self, **_):
        self.usage = Usage()

    @abstractmethod
    def reset(self) -> None:
        """Forget everything. Called before each example."""

    @abstractmethod
    def add_turn(self, turn: Turn) -> None:
        """Observe the next turn of the session."""

    @abstractmethod
    def retrieve(self, question: str) -> MemoryContext:
        """Return the context to show the model for ``question``."""

    @staticmethod
    def render(turns: List[Turn]) -> str:
        return "\n".join(t.render() for t in turns)
