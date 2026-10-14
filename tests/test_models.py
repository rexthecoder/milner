from types import SimpleNamespace

import pytest

from milner.models import MockModel, build_model, register_model
from milner.models.anthropic_adapter import AnthropicModel
from milner.models.openai_adapter import OpenAIModel


def test_mock_answers_from_memory():
    out = MockModel().generate(
        [
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": "Memory:\n[t1] user: my favorite color is teal.\n"
                "[t2] user: nice weather.\n\nQuestion: What is my favorite color?\nAnswer:",
            },
        ]
    )
    assert out.text == "teal"
    assert out.input_tokens > 0 and out.output_tokens == 1 and out.latency_s >= 0


def test_mock_says_unknown_without_evidence():
    out = MockModel().generate(
        [
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": "Memory:\n(empty)\n\nQuestion: What is my pet name?\nAnswer:",
            },
        ]
    )
    assert out.text == "unknown"


def test_build_model_passes_seed_and_rejects_unknown():
    assert build_model({"type": "mock"}, seed=7).seed == 7
    with pytest.raises(ValueError, match="unknown model"):
        build_model({"type": "nope"})


def test_register_custom_model():
    class Echo(MockModel):
        name = "echo"

    register_model("echo", Echo)
    assert isinstance(build_model({"type": "echo"}), Echo)


class FakeOpenAI:
    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=" Paris "))],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=2),
        )


def test_openai_adapter_reports_usage_and_passes_determinism_args():
    client = FakeOpenAI()
    model = OpenAIModel(model="m", client=client, seed=3, temperature=0.0)
    out = model.generate([{"role": "user", "content": "capital of France?"}])
    assert (out.text, out.input_tokens, out.output_tokens) == ("Paris", 12, 2)
    assert client.calls[0]["seed"] == 3 and client.calls[0]["temperature"] == 0.0


class FakeAnthropic:
    def __init__(self):
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Paris")],
            usage=SimpleNamespace(input_tokens=9, output_tokens=1),
        )


def test_anthropic_adapter_splits_system_prompt():
    client = FakeAnthropic()
    model = AnthropicModel(model="m", client=client)
    out = model.generate(
        [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
    )
    assert out.text == "Paris" and out.input_tokens == 9
    call = client.calls[0]
    assert call["system"] == "be brief"
    assert call["messages"] == [{"role": "user", "content": "hi"}]
