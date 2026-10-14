"""Small text helpers shared by memory backends, models and metrics."""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9']+")
_TOKEN = re.compile(r"\w+|[^\w\s]")

STOPWORDS = frozenset(
    """a an the and or but of to in on at by for with from is are was were be been am do does did
    my your our his her its their me you we he she it they i what which who whom whose when where
    why how this that these those there here as if so than then too very can will just not no
    have has had""".split()
)


def words(text: str) -> list[str]:
    """Lower-cased word tokens (letters, digits, apostrophes)."""
    return _WORD.findall(text.lower())


def content_words(text: str) -> list[str]:
    """Word tokens with stopwords removed."""
    return [w for w in words(text) if w not in STOPWORDS]


def count_tokens(text: str) -> int:
    """Model-agnostic token estimate used when an adapter reports no usage.

    Counts word and punctuation pieces. It tracks real tokenizers loosely but is
    deterministic and dependency-free, which is what the offline mock needs.
    """
    return len(_TOKEN.findall(text))


def normalize_answer(text: str) -> str:
    """Normalise an answer for matching: lower-case, drop punctuation and articles."""
    return " ".join(w for w in words(text) if w not in {"a", "an", "the"})
