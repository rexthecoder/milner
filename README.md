# milner

**A benchmark harness for comparing memory strategies in LLM agents.**

Run the same task through different memory designs under one standard configuration, and
get accuracy, token cost, latency and retrieval hit rate side by side. Every prompt and
response is logged to disk, and the results table and plots are rebuilt from those raw
logs.

The name honours [Brenda Milner](https://en.wikipedia.org/wiki/Brenda_Milner), whose work
with patient H.M. showed that memory is not one system but several, the idea this tool
puts to the test for LLM agents.

## Why

Papers on agent memory (rolling summaries, vector retrieval, full-context prompting) each
use their own prompts, models, scoring and cost accounting, so their numbers cannot be
compared. milner fixes those variables: one interface for memory strategies, one for
models, one JSONL schema for tasks, and one metrics definition that charges a strategy
for the model calls it makes while remembering.

## Install

```bash
pip install milner              # core: runs offline, no API key needed
pip install "milner[plots]"     # + matplotlib figures
pip install "milner[openai]"    # OpenAI API, or local vLLM / llama.cpp / Ollama servers
pip install "milner[anthropic]" # Anthropic API
pip install "milner[hf]"        # local models via transformers
pip install "milner[embeddings]"  # sentence-transformers embeddings for retrieval
```

From a clone: `pip install -e ".[dev]"`. Python 3.9 or newer.

## Quickstart

No GPU and no API key are needed; the quickstart uses a deterministic mock model.

```bash
milner make-toy -o examples/toy.jsonl      # synthetic benchmark (already committed)
milner run examples/toy.yaml
```

```
| Memory                                       | N  | Accuracy | Exact match | Input tok | Output tok | Latency (s) | Retrieval hit |
|---|---|---|---|---|---|---|---|
| none                                         | 20 | 0.000    | 0.000       | 49.8      | 1.0        | 0.0000      | 0.000         |
| full                                         | 20 | 1.000    | 1.000       | 445.9     | 1.0        | 0.0001      | 1.000         |
| retrieval(top_k=3)                           | 20 | 1.000    | 1.000       | 88.2      | 1.0        | 0.0005      | 1.000         |
| summary(chunk_size=10,max_summary_words=60)  | 20 | 0.050    | 0.050       | 764.0     | 151.1      | 0.0001      | -             |
```

This writes to `runs/toy/`:

| File | Contents |
|---|---|
| `<memory>.jsonl` | one record per example: full prompt, response, tokens, latency, retrieved turns |
| `config.yaml` | the exact config that produced the run |
| `summary.json` | aggregate metrics |
| `results.md`, `results.csv`, `plot.png` | table and figure, rebuilt with `milner report runs/toy` |

The mock model is a stand-in that makes the harness runnable offline; the numbers above
show how the harness accounts for cost (note the summary strategy's extra tokens from its
summarisation calls), not how good any real model is. To use a real model, change the
`model:` block, for example `examples/openai_compatible.yaml`.

## Concepts

**Memory backends** (`milner.memory`): `none` (question only), `full` (whole conversation
in the prompt), `retrieval` (embed each turn, return top-k), `summary` (rolling
summarisation by the model). Add your own with `register_memory`.

**Model adapters** (`milner.models`): `mock`, `openai` (also any OpenAI-compatible local
server through `base_url`), `anthropic`, `hf`. Add your own with `register_model`.

**Tasks** are JSONL, one example per line:

```json
{"id": "ex-0", "turns": [{"turn_id": "t0", "role": "user", "content": "..."}],
 "question": "What is my favorite color?", "answer": "teal", "evidence": ["t7"]}
```

`evidence` lists the turns that contain the answer; the retrieval hit rate is the fraction
of them that made it into the context.

**Metrics**: accuracy (normalised whole-word containment), exact match, input and output
tokens per question (answering call *plus* any model calls the memory made), latency,
retrieval hit rate.

**Reproducibility**: one YAML per experiment, seeds passed to every adapter, temperature 0
by default, the resolved config copied next to the logs, and a deterministic CRC32-based
default embedder. See the [docs](https://rexthecoder.github.io/milner/) for what is and
is not guaranteed for hosted APIs.

## Extending

```python
from milner import MemoryBackend, MemoryContext, register_memory


class LastN(MemoryBackend):
    def __init__(self, n=5, **kw):
        super().__init__(**kw)
        self.n, self.turns = n, []

    def reset(self):
        self.turns = []

    def add_turn(self, turn):
        self.turns.append(turn)

    def retrieve(self, question):
        kept = self.turns[-self.n :]
        return MemoryContext(self.render(kept), [t.turn_id for t in kept])


register_memory("last_n", LastN)  # now usable as `type: last_n` in YAML
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## Citation

If you use milner, please cite the JOSS paper once published (see `paper.md`) and the
software via `CITATION.cff`.

## License

MIT. See [LICENSE](LICENSE).
