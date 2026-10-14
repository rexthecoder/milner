"""Cost accounting shared by models and memory backends."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Usage:
    """Tokens, wall-clock time and number of model calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    calls: int = 0

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.latency_s += other.latency_s
        self.calls += other.calls

    def reset(self) -> None:
        self.input_tokens = self.output_tokens = self.calls = 0
        self.latency_s = 0.0
