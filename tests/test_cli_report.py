import pytest
import yaml

from milner.cli import main
from milner.report import build_report, to_markdown


def test_cli_quickstart_flow(tmp_path, capsys):
    data = tmp_path / "toy.jsonl"
    assert main(["make-toy", "-o", str(data), "-n", "5", "--turns", "20"]) == 0
    cfg = tmp_path / "exp.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "name": "q",
                "dataset": "toy.jsonl",
                "output_dir": "runs",
                "model": {"type": "mock"},
                "memories": [{"type": "none"}, {"type": "full"}],
            }
        )
    )
    assert main(["run", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "| Memory |" in out and "full" in out
    run_dir = tmp_path / "runs" / "q"
    assert (run_dir / "results.md").exists() and (run_dir / "results.csv").exists()
    assert main(["report", str(run_dir)]) == 0  # rebuilt from raw logs alone


def test_plot_is_written_when_matplotlib_available(tmp_path, toy_path):
    pytest.importorskip("matplotlib")
    from milner.config import load_config
    from milner.runner import run_experiment

    cfg_path = tmp_path / "e.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "name": "p",
                "dataset": str(toy_path),
                "output_dir": str(tmp_path / "r"),
                "model": {"type": "mock"},
                "memories": [{"type": "none"}, {"type": "full"}],
            }
        )
    )
    cfg = load_config(cfg_path)
    run_experiment(cfg)
    written = build_report(cfg.run_dir)
    assert written["plot"].stat().st_size > 0


def test_report_on_empty_dir_is_a_clean_error(tmp_path, capsys):
    assert main(["report", str(tmp_path)]) == 1
    assert "no run logs" in capsys.readouterr().err


def test_markdown_table_renders_missing_values_as_dash():
    row = {
        "memory": "m",
        "n": 1,
        "accuracy": 1.0,
        "exact_match": 1.0,
        "input_tokens": 10.0,
        "output_tokens": 1.0,
        "latency_s": 0.01,
        "retrieval_hit_rate": None,
    }
    assert to_markdown([row]).splitlines()[-1].endswith("| - |")


def test_list_command(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "retrieval" in out and "mock" in out
