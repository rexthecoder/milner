"""A deterministic, offline stand-in for an LLM.

It exists so the harness, the tests and the quickstart run without a GPU or an API key. It is
*not* a language model. It does two things:

* answering: picks the memory line that shares the most content words with the question and
  returns that line's final word;
* summarising: keeps only the most recent ``summary_words`` words of the text it is given, a
  lossy compression that makes the rolling-summary strategy forget early facts, as a small
  real model would.

Results from this model demonstrate the harness mechanics, not model quality.
"""

from __future__ import annotations

from typing import List

from ..text import content_words, count_tokens
from .base import Message, ModelAdapter

# The summary backend's system prompt contains this phrase; the mock keys on it.
SUMMARY_REQUEST_PHRASE = "running summary"


class MockModel(ModelAdapter):
    name = "mock"

    def __init__(self, summary_words: int = 40, **kwargs):
        super().__init__(**kwargs)
        self.summary_words = summary_words

    def _complete(self, messages: List[Message], max_tokens: int):
        prompt = "\n".join(m["content"] for m in messages)
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        if SUMMARY_REQUEST_PHRASE in system:
            text = self._summarize(messages[-1]["content"])
        else:
            text = self._answer(messages[-1]["content"])
        return text, count_tokens(prompt), count_tokens(text)

    def _summarize(self, user_message: str) -> str:
        body = user_message.split("Previous summary:", 1)[-1].split("Updated summary:", 1)[0]
        lines = [ln for ln in body.splitlines() if ln.strip() and not ln.startswith("New turns:")]
        kept, budget = [], self.summary_words
        for line in reversed(lines):  # newest lines first
            n = len(line.split())
            if n > budget:
                break
            kept.append(line)
            budget -= n
        return "\n".join(reversed(kept))

    @staticmethod
    def _answer(user_message: str) -> str:
        memory, _, question = user_message.rpartition("Question:")
        q_words = set(content_words(question.replace("Answer:", "")))
        best, best_score = "", 0
        for line in memory.splitlines():
            score = len(q_words & set(content_words(line)))
            if score > best_score:
                best, best_score = line, score
        if best_score == 0:
            return "unknown"
        tail = best.split()[-1] if best.split() else "unknown"
        return tail.strip(".,!?;:\"'")
