"""The single interface every model backend implements."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List

Message = Dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float


class ModelAdapter(ABC):
    """Turn a chat transcript into a completion, reporting token usage and latency.

    Subclasses implement :meth:`_complete`; :meth:`generate` adds timing.
    """

    name = "model"

    def __init__(self, temperature: float = 0.0, max_tokens: int = 128, seed: int = 0, **_):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed

    @abstractmethod
    def _complete(self, messages: List[Message], max_tokens: int) -> tuple[str, int, int]:
        """Return ``(text, input_tokens, output_tokens)``."""

    def generate(self, messages: List[Message], max_tokens: int | None = None) -> Completion:
        start = time.perf_counter()
        text, n_in, n_out = self._complete(messages, max_tokens or self.max_tokens)
        return Completion(text.strip(), n_in, n_out, time.perf_counter() - start)
