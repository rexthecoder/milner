---
title: "milner: a benchmark harness for comparing memory strategies in LLM agents"
tags:
  - Python
  - large language models
  - agents
  - memory
  - benchmarking
  - reproducibility
authors:
  - name: Rexford Asamoah
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: "Independent researcher"
    index: 1
date: 14 October 2026
bibliography: paper.bib
---

# Summary

Large language model (LLM) agents that hold long conversations or run for long periods
need some form of memory, because the context window is finite and expensive. Common
designs include placing the entire history in the prompt, retrieving the most relevant
past turns from a vector store [@lewis2020rag], and keeping a rolling summary that is
updated as the conversation grows [@packer2023memgpt]. `milner` is a Python library that
runs the same task through any of these designs under one standard configuration and
reports answer accuracy, token cost, latency and retrieval hit rate side by side. Every
prompt and response is written to disk, and the results table and plots are rebuilt from
those raw logs.

The package is named after Brenda Milner, whose work with the patient H.M.
[@scoville1957loss] showed that memory is a set of distinct systems rather than one
faculty, the question `milner` asks of LLM agents.

# Statement of need

Benchmarks of long-term conversational memory exist [@maharana2024locomo; @wu2024longmemeval],
but they ship their own evaluation scripts. Memory strategies from different papers are
therefore compared under different prompts, models, answer-matching rules and cost
accounting, and a researcher who wants to know whether a new strategy beats retrieval on
their own data must reimplement the baselines. Cost is the most commonly missing axis:
a summarisation strategy that calls a model every few turns appears cheap when those calls
are left out of the token count.

`milner` targets researchers and engineers who need a controlled comparison of memory
designs, including on data of their own. It offers:

- **Pluggable memory backends** behind one three-method interface: no memory,
  full context, vector retrieval (top-$k$) and rolling summarisation, with a registry for
  custom strategies.
- **A standard task schema** (JSONL: session turns, question, gold answer, evidence
  location), so any dataset can be converted once and run against every strategy.
- **Model adapters** behind one interface: OpenAI-compatible servers (hosted OpenAI, or
  local vLLM [@kwon2023vllm], llama.cpp and Ollama through a base URL), the Anthropic
  API, and in-process Hugging Face `transformers`. A deterministic mock model makes the
  whole pipeline runnable offline for testing and review.
- **Cost-faithful metrics**: accuracy, exact match, input and output tokens, latency and
  retrieval hit rate. Model calls a memory makes while ingesting (for example
  summarisation) are charged to that strategy.
- **Reproducible, config-driven runs**: one YAML file per experiment, seeds passed to
  every adapter, temperature zero by default, the resolved configuration saved beside
  the logs, and tables and figures regenerated from the raw per-example logs.

`milner` does not propose a new memory algorithm or dataset; it provides the common
scaffolding for comparing them.

# Software design

A run takes a dataset and a list of memory specifications. For each strategy and each
example, the memory is reset, fed the session turn by turn, and asked for a context for the
question; the context and question are assembled into a fixed prompt and sent to the
model. The record written for each example contains the prompt, response, usage reported by
the model and by the memory, the retrieved turn identifiers and the score. Because the
retrieval hit rate is computed against labelled evidence turns, it separates failures of
memory (the information was not surfaced) from failures of reading (it was surfaced but
answered wrongly). Strategies that quote no source turns, such as summaries, report no hit
rate instead of a misleading zero.

The test suite (run in continuous integration on Linux and macOS across Python 3.9 to
3.12) covers the schema validators, every memory backend, metric definitions, the
adapters against fake clients, determinism of repeated runs, and the command-line
quickstart end to end.

# Acknowledgements

TODO: funding and contributors.

# References
