"""Anthropic Messages API."""

from __future__ import annotations

from typing import List

from .base import Message, ModelAdapter


class AnthropicModel(ModelAdapter):
    name = "anthropic"

    def __init__(self, model: str, api_key: str | None = None, client=None, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - exercised only without the extra
                raise ImportError("install the extra: pip install 'milner[anthropic]'") from exc
            client = anthropic.Anthropic(api_key=api_key)
        self.client = client

    def _complete(self, messages: List[Message], max_tokens: int):
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        chat = [m for m in messages if m["role"] != "system"]
        kwargs = {"system": system} if system else {}
        resp = self.client.messages.create(
            model=self.model,
            messages=chat,
            temperature=self.temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return text, resp.usage.input_tokens, resp.usage.output_tokens
