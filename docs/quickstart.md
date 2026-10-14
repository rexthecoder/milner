# Quickstart

```bash
pip install "milner[plots]"
git clone https://github.com/rexfordasamoah/milner && cd milner
milner run examples/toy.yaml
```

The run uses the offline mock model, so it needs no GPU or API key. It prints a table and
writes `runs/toy/`:

- `<memory>.jsonl`: one record per example with the full prompt, response, token counts,
  latency and retrieved turns.
- `config.yaml`: the exact config used.
- `summary.json`, `results.md`, `results.csv`, `plot.png`.

Rebuild the table and figure from the raw logs at any time:

```bash
milner report runs/toy
```

## Using a real model

Start any OpenAI-compatible server (vLLM, llama.cpp's `llama-server`, Ollama) and use
`examples/openai_compatible.yaml`:

```bash
pip install "milner[openai,embeddings,plots]"
vllm serve Qwen/Qwen2.5-1.5B-Instruct
milner run examples/openai_compatible.yaml
```

For the hosted OpenAI API drop `base_url` and set `OPENAI_API_KEY`. For Anthropic use
`type: anthropic` and `ANTHROPIC_API_KEY`. For in-process inference use `type: hf`.

## Your own benchmark

Convert it to JSONL (see [Configuration](configuration.md#task-schema)) and point
`dataset:` at the file. Validation errors name the file and line.
