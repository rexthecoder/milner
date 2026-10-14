"""Memory strategies and their registry."""

from __future__ import annotations

from typing import Callable, Dict, Optional

from ..models.base import ModelAdapter
from .base import MemoryBackend, MemoryContext
from .retrieval import Retrieval
from .simple import FullContext, NoMemory
from .summary import Summary

MEMORIES: Dict[str, Callable[..., MemoryBackend]] = {
    "none": NoMemory,
    "full": FullContext,
    "retrieval": Retrieval,
    "summary": Summary,
}


def register_memory(name: str, factory: Callable[..., MemoryBackend]) -> None:
    """Make a custom strategy available to YAML configs under ``type: <name>``.

    The factory is called with the spec's keys as keyword arguments, plus ``model`` (the
    experiment's model adapter) so strategies may call the LLM.
    """
    MEMORIES[name] = factory


def build_memory(spec: dict, model: Optional[ModelAdapter] = None) -> MemoryBackend:
    """Build a backend from a config entry, e.g. ``{"type": "retrieval", "top_k": 5}``."""
    spec = dict(spec)
    kind = spec.pop("type")
    spec.pop("label", None)
    if kind not in MEMORIES:
        raise ValueError(f"unknown memory type {kind!r}; available: {sorted(MEMORIES)}")
    return MEMORIES[kind](model=model, **spec)


def memory_label(spec: dict) -> str:
    """Name a configured strategy in results: its ``label`` or ``type`` plus parameters."""
    if "label" in spec:
        return str(spec["label"])
    params = [f"{k}={v}" for k, v in spec.items() if k != "type" and not isinstance(v, dict)]
    return spec["type"] + (f"({','.join(params)})" if params else "")


__all__ = [
    "MEMORIES",
    "FullContext",
    "MemoryBackend",
    "MemoryContext",
    "NoMemory",
    "Retrieval",
    "Summary",
    "build_memory",
    "memory_label",
    "register_memory",
]
