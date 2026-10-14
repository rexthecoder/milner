# Extending milner

## A new memory strategy

Subclass `MemoryBackend`, implement `reset`, `add_turn` and `retrieve`, and register it.

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


register_memory("last_n", LastN)
```

Then use it in YAML:

```yaml
memories:
  - {type: last_n, n: 8}
```

Rules that keep comparisons fair:

- The factory receives the experiment's `model`. If your strategy calls it, charge the
  tokens and latency to `self.usage` (see `milner/memory/summary.py`), or the strategy
  gets free compression.
- Return `retrieved_turn_ids` when the context quotes source turns verbatim, and `None`
  when it does not, so the retrieval hit rate stays meaningful.

## A new model

Subclass `ModelAdapter`, implement `_complete(messages, max_tokens)` returning
`(text, input_tokens, output_tokens)`, and `register_model("name", YourClass)`.

## A new dataset

Write a converter that yields `milner.Example` objects and `milner.schema.dump_jsonl` them.
See `milner/toy.py` for a complete example.
