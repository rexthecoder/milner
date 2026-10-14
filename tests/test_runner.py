import json

import pytest
import yaml

from milner.config import load_config
from milner.runner import read_logs, run_experiment


def write_config(tmp_path, toy_path, **overrides):
    cfg = {
        "name": "t",
        "seed": 0,
        "dataset": str(toy_path),
        "output_dir": str(tmp_path / "runs"),
        "model": {"type": "mock"},
        "memories": [
            {"type": "none"},
            {"type": "full"},
            {"type": "retrieval", "top_k": 3},
            {"type": "summary", "chunk_size": 6, "max_summary_words": 30},
        ],
        **overrides,
    }
    path = tmp_path / "exp.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_end_to_end_orders_strategies_as_expected(tmp_path, toy_path):
    cfg = load_config(write_config(tmp_path, toy_path))
    res = run_experiment(cfg)
    assert res["none"]["accuracy"] == 0.0
    assert res["full"]["accuracy"] == 1.0
    assert res["retrieval(top_k=3)"]["accuracy"] == 1.0
    assert res["retrieval(top_k=3)"]["input_tokens"] < res["full"]["input_tokens"]
    summ = res["summary(chunk_size=6,max_summary_words=30)"]
    assert summ["accuracy"] < 1.0
    assert summ["retrieval_hit_rate"] is None


def test_every_prompt_and_response_is_logged(tmp_path, toy_path, toy):
    cfg = load_config(write_config(tmp_path, toy_path))
    run_experiment(cfg)
    logs = read_logs(cfg.run_dir)
    assert list(logs) == [
        "none",
        "full",
        "retrieval(top_k=3)",
        "summary(chunk_size=6,max_summary_words=30)",
    ]
    for records in logs.values():
        assert len(records) == len(toy)
        for r in records:
            assert r["prompt"][0]["role"] == "system"
            assert r["question"] in r["prompt"][-1]["content"]
            assert "response" in r and "memory_calls" in r
    assert (cfg.run_dir / "config.yaml").exists()
    assert (cfg.run_dir / "summary.json").exists()


def test_runs_are_deterministic(tmp_path, toy_path):
    def run(sub):
        d = tmp_path / sub
        d.mkdir()
        cfg = load_config(write_config(d, toy_path))
        run_experiment(cfg)
        return {
            k: [(r["prediction"], r["prompt"], r["retrieved_turn_ids"]) for r in v]
            for k, v in read_logs(cfg.run_dir).items()
        }

    assert run("a") == run("b")


def test_limit_and_duplicate_labels(tmp_path, toy_path):
    cfg = load_config(write_config(tmp_path, toy_path, limit=2))
    assert all(v["n"] == 2 for v in run_experiment(cfg).values())

    dup = load_config(
        write_config(tmp_path, toy_path, memories=[{"type": "full"}, {"type": "full"}])
    )
    with pytest.raises(ValueError, match="unique"):
        run_experiment(dup)


def test_config_validation(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\n")
    with pytest.raises(ValueError, match="missing required"):
        load_config(bad)


def test_relative_paths_resolve_against_the_config(tmp_path, toy_path):
    path = write_config(tmp_path, toy_path, dataset=toy_path.name, output_dir="out")
    cfg = load_config(path)
    assert cfg.dataset == str(tmp_path / toy_path.name)
    assert cfg.run_dir == (tmp_path / "out" / "t").resolve()


def test_summary_json_is_valid(tmp_path, toy_path):
    cfg = load_config(write_config(tmp_path, toy_path))
    run_experiment(cfg)
    data = json.loads((cfg.run_dir / "summary.json").read_text())
    assert set(data["full"]) >= {"accuracy", "input_tokens", "latency_s", "retrieval_hit_rate"}
