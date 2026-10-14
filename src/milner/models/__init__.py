"""Model adapters and their registry."""

from __future__ import annotations

from typing import Callable, Dict

from .base import Completion, Message, ModelAdapter
from .mock import MockModel


def _openai(**kw):
    from .openai_adapter import OpenAIModel

    return OpenAIModel(**kw)


def _anthropic(**kw):
    from .anthropic_adapter import AnthropicModel

    return AnthropicModel(**kw)


def _hf(**kw):
    from .hf_adapter import HFModel

    return HFModel(**kw)


# Factories import lazily so optional dependencies are only needed when used.
MODELS: Dict[str, Callable[..., ModelAdapter]] = {
    "mock": MockModel,
    "openai": _openai,
    "anthropic": _anthropic,
    "hf": _hf,
}


def register_model(name: str, factory: Callable[..., ModelAdapter]) -> None:
    """Make a custom adapter available to YAML configs under ``type: <name>``."""
    MODELS[name] = factory


def build_model(spec: dict, seed: int = 0) -> ModelAdapter:
    spec = dict(spec)
    kind = spec.pop("type")
    if kind not in MODELS:
        raise ValueError(f"unknown model type {kind!r}; available: {sorted(MODELS)}")
    spec.setdefault("seed", seed)
    return MODELS[kind](**spec)


__all__ = [
    "Completion",
    "Message",
    "ModelAdapter",
    "MockModel",
    "MODELS",
    "build_model",
    "register_model",
]
