"""Per-example scoring and aggregation over raw run logs."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from .text import normalize_answer


def exact_match(prediction: str, gold: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(gold)


def contains_match(prediction: str, gold: str) -> bool:
    """Whole-word containment of the normalised gold answer in the prediction."""
    g, p = normalize_answer(gold), normalize_answer(prediction)
    return bool(g) and f" {g} " in f" {p} "


def retrieval_hit(retrieved: Optional[List[str]], evidence: List[str]) -> Optional[float]:
    """Fraction of evidence turns present in the retrieved set.

    ``None`` when the strategy keeps no turn identity or the example has no evidence label.
    """
    if retrieved is None or not evidence:
        return None
    got = set(retrieved)
    return sum(e in got for e in evidence) / len(evidence)


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def aggregate(records: List[dict]) -> Dict[str, Optional[float]]:
    """Summarise one strategy's per-example log records.

    ``input_tokens`` and ``latency_s`` are totals per example: the answering call plus any
    model calls the memory made while ingesting, so memory overhead is not hidden.
    """
    return {
        "n": len(records),
        "accuracy": _mean(float(r["correct"]) for r in records),
        "exact_match": _mean(float(r["exact_match"]) for r in records),
        "input_tokens": _mean(r["input_tokens"] + r["memory_input_tokens"] for r in records),
        "output_tokens": _mean(r["output_tokens"] + r["memory_output_tokens"] for r in records),
        "latency_s": _mean(r["latency_s"] + r["memory_latency_s"] for r in records),
        "retrieval_hit_rate": _mean(r["retrieval_hit"] for r in records),
    }
