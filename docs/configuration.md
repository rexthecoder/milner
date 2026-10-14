# Configuration

One YAML file describes one experiment. Relative paths resolve against the config file, so
it works from any directory.

```yaml
name: my-experiment        # run directory name
seed: 0                    # passed to every model adapter
dataset: data.jsonl
output_dir: runs           # logs go to runs/<name>/
limit: null                # optionally only the first N examples
max_answer_tokens: 32

model:
  type: openai             # mock | openai | anthropic | hf
  model: gpt-4o-mini
  temperature: 0.0

memories:
  - type: none
  - type: full
    max_turns: null        # keep only the most recent N turns
  - type: retrieval
    top_k: 3
    embedder: {type: hashing, dim: 256}   # or {type: sentence-transformers, model: ...}
  - type: summary
    chunk_size: 10
    max_summary_words: 120
    label: rolling         # optional display name; labels must be unique
```

## Model adapters

| `type` | Notes |
|---|---|
| `mock` | Deterministic offline stand-in. For harness checks and tests, not for science. |
| `openai` | Hosted API, or any OpenAI-compatible server through `base_url` (vLLM, llama.cpp, Ollama). |
| `anthropic` | Anthropic Messages API. |
| `hf` | In-process `transformers`; greedy decoding at temperature 0. |

## Task schema

JSONL, one example per line.

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | unique example id |
| `turns` | yes | list of `{turn_id, role, content}`; `turn_id` defaults to `t<index>` |
| `question` | yes | asked after the whole session is ingested |
| `answer` | yes | gold answer |
| `evidence` | no | `turn_id`s that contain the answer; enables the retrieval hit rate |
| `session_id` | no | defaults to `id` |

## Reproducibility

What milner does: fixes seeds and passes them to adapters, defaults to temperature 0,
copies the resolved config beside the logs, rebuilds every table from the raw logs, and
uses a hash embedder that is identical across processes.

What it cannot promise: hosted APIs may change models behind a name or ignore `seed`;
`transformers` outputs can differ across hardware and library versions. Record the model
version and library versions with your results; the per-example logs make differences
inspectable.
