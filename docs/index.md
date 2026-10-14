# milner

A benchmark harness for comparing **memory strategies in LLM agents**.

Run the same task through different memory designs under one standard configuration and get
accuracy, token cost, latency and retrieval hit rate side by side, with every prompt and
response logged to disk.

## Statement of need

Agent-memory work (rolling summaries, retrieval over past turns, long-context prompting)
is usually evaluated with bespoke scripts: different prompts, models, scoring rules and
cost accounting. Results are therefore hard to compare or reproduce, and the cost side is
often missing: a summary strategy that makes a model call every few turns looks cheap if
those calls are not counted.

milner provides the shared scaffolding: a memory interface, a model interface, a JSONL task
schema, one metrics definition that charges memory overhead to the strategy, and
config-driven, seeded, fully logged runs.

## What it is not

It is not a new memory algorithm and not a new dataset. It is the harness that lets you
compare algorithms on datasets, including ones you bring.

## Naming

Named after [Brenda Milner](https://en.wikipedia.org/wiki/Brenda_Milner), whose studies of
patient H.M. established that memory comprises distinct systems.
