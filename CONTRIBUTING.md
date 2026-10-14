# Contributing

Bug reports, new memory strategies, model adapters and dataset converters are welcome.

```bash
git clone https://github.com/rexthecoder/milner && cd milner
pip install -e ".[dev]"
pytest
ruff check .
```

- Open an issue before large changes.
- New backends and adapters need tests; adapters for hosted APIs should be tested against
  a fake client (see `tests/test_models.py`), never against the live service.
- Memory strategies that call a model must charge the usage to `self.usage`.
Report problems with the software through GitHub issues.
