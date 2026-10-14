"""OpenAI and OpenAI-compatible servers (vLLM, llama.cpp ``llama-server``, Ollama, ...)."""

from __future__ import annotations

from typing import List

from ..text import count_tokens
from .base import Message, ModelAdapter


class OpenAIModel(ModelAdapter):
    """Chat completions through the ``openai`` client.

    Set ``base_url`` to point at a local vLLM / llama.cpp server, which makes this the
    "local model" path as well as the hosted-API path.
    """

    name = "openai"

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        client=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - exercised only without the extra
                raise ImportError("install the extra: pip install 'milner[openai]'") from exc
            # Local servers ignore the key but the client insists on one.
            client = OpenAI(base_url=base_url, api_key=api_key or ("local" if base_url else None))
        self.client = client

    def _complete(self, messages: List[Message], max_tokens: int):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
            seed=self.seed,
        )
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        if usage is not None:
            return text, usage.prompt_tokens, usage.completion_tokens
        prompt = "\n".join(m["content"] for m in messages)
        return text, count_tokens(prompt), count_tokens(text)
