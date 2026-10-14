"""Rolling summarisation: fold the conversation into a bounded summary as it arrives."""

from __future__ import annotations

from typing import List

from ..models.base import ModelAdapter
from ..schema import Turn
from .base import MemoryBackend, MemoryContext

SUMMARY_SYSTEM = (
    "You maintain a running summary of a conversation. Merge the new turns into the previous "
    "summary. Keep every concrete fact (names, numbers, places, preferences) and drop small talk."
)


class Summary(MemoryBackend):
    """Every ``chunk_size`` turns, ask the model to merge them into a running summary.

    The summarisation calls are real model calls; their tokens and latency are charged to
    ``self.usage`` so this strategy is not credited with free compression.
    """

    name = "summary"

    def __init__(
        self,
        model: ModelAdapter,
        chunk_size: int = 10,
        max_summary_words: int = 120,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if model is None:
            raise ValueError("the summary memory needs a model to summarise with")
        self.model = model
        self.chunk_size = chunk_size
        self.max_summary_words = max_summary_words
        self._summary = ""
        self._pending: List[Turn] = []

    def reset(self) -> None:
        self._summary, self._pending = "", []

    def add_turn(self, turn: Turn) -> None:
        self._pending.append(turn)
        if len(self._pending) >= self.chunk_size:
            self._fold()

    def _fold(self) -> None:
        if not self._pending:
            return
        user = (
            f"Previous summary:\n{self._summary or '(empty)'}\n\n"
            f"New turns:\n{self.render(self._pending)}\n\n"
            f"Write the updated summary in at most {self.max_summary_words} words.\n"
            "Updated summary:"
        )
        out = self.model.generate(
            [{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": user}],
            max_tokens=self.max_summary_words * 3,
        )
        self.usage.input_tokens += out.input_tokens
        self.usage.output_tokens += out.output_tokens
        self.usage.latency_s += out.latency_s
        self.usage.calls += 1
        self._summary, self._pending = out.text, []

    def retrieve(self, question: str) -> MemoryContext:
        self._fold()  # flush turns that did not fill a whole chunk
        return MemoryContext(self._summary, retrieved_turn_ids=None)
