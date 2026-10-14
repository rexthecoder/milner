import pytest

from milner.toy import make_toy


@pytest.fixture
def toy():
    return make_toy(n=6, turns=24, facts=3, seed=1)


@pytest.fixture
def toy_path(tmp_path, toy):
    from milner.schema import dump_jsonl

    path = tmp_path / "toy.jsonl"
    dump_jsonl(toy, path)
    return path
