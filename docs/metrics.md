# Metrics

All metrics are computed from the raw per-example logs (`<memory>.jsonl`).

| Metric | Definition |
|---|---|
| Accuracy | Normalised gold answer appears as whole words in the prediction. Normalisation: lower-case, strip punctuation and the articles *a/an/the*. |
| Exact match | Normalised prediction equals normalised gold. |
| Input tokens | Mean per question: tokens in the answering prompt **plus** tokens of any model calls the memory made (for example summarisation). |
| Output tokens | Same, for generated tokens. |
| Latency (s) | Mean per question: the answering call plus wall-clock time spent ingesting turns and retrieving. |
| Retrieval hit rate | Mean fraction of an example's `evidence` turns present in the context. Strategies that keep no turn identity (summary) report `-` and are excluded rather than counted as zero. |

Token counts come from the model adapter's reported usage. The mock model uses a simple
word-and-punctuation estimate.

## Choosing a different scorer

`correct` is whole-word containment, which suits short factual answers. For free-form
answers, write the predictions out of the logs and score them with your own judge; the
`prediction` and `gold` fields are in every record.
