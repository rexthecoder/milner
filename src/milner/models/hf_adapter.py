"""Local inference with Hugging Face transformers."""

from __future__ import annotations

from typing import List

from .base import Message, ModelAdapter


class HFModel(ModelAdapter):
    """Run a chat model in-process with ``transformers``.

    Decoding is greedy when ``temperature`` is 0, and the global seed is fixed, so runs
    are repeatable on the same hardware and library versions.
    """

    name = "hf"

    def __init__(self, model: str, device: str | None = None, **kwargs):
        super().__init__(**kwargs)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError("install the extra: pip install 'milner[hf]'") from exc
        set_seed(self.seed)
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model)
        if device:
            self.model.to(device)
        self.device = self.model.device

    def _complete(self, messages: List[Message], max_tokens: int):
        ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.device)
        n_in = ids["input_ids"].shape[1]
        sample = self.temperature > 0
        with self._torch.no_grad():
            out = self.model.generate(
                **ids,
                max_new_tokens=max_tokens,
                do_sample=sample,
                temperature=self.temperature if sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new = out[0][n_in:]
        return self.tokenizer.decode(new, skip_special_tokens=True), n_in, len(new)
