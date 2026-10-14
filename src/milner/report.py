"""Build the results table and plots from raw logs (never from in-memory state)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

from .metrics import aggregate
from .runner import read_logs

COLUMNS = [
    ("memory", "Memory"),
    ("n", "N"),
    ("accuracy", "Accuracy"),
    ("exact_match", "Exact match"),
    ("input_tokens", "Input tok"),
    ("output_tokens", "Output tok"),
    ("latency_s", "Latency (s)"),
    ("retrieval_hit_rate", "Retrieval hit"),
]


def _fmt(key: str, value) -> str:
    if value is None:
        return "-"
    if key in ("memory", "n"):
        return str(value)
    if key in ("accuracy", "exact_match", "retrieval_hit_rate"):
        return f"{value:.3f}"
    if key == "latency_s":
        return f"{value:.4f}"
    return f"{value:.1f}"


def build_table(logs: Dict[str, List[dict]]) -> List[dict]:
    rows = []
    for label, records in logs.items():
        rows.append({"memory": label, **aggregate(records)})
    return rows


def to_markdown(rows: List[dict]) -> str:
    head = "| " + " | ".join(title for _, title in COLUMNS) + " |"
    rule = "|" + "|".join("---" for _ in COLUMNS) + "|"
    body = ["| " + " | ".join(_fmt(k, r[k]) for k, _ in COLUMNS) + " |" for r in rows]
    return "\n".join([head, rule, *body]) + "\n"


def to_csv(rows: List[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([k for k, _ in COLUMNS])
        for r in rows:
            writer.writerow(["" if r[k] is None else r[k] for k, _ in COLUMNS])


def plot(rows: List[dict], path: Path) -> bool:
    """Write a three-panel figure. Returns False if matplotlib is not installed."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    labels = [r["memory"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    panels = [
        ("accuracy", "Accuracy", axes[0]),
        ("input_tokens", "Input tokens / question", axes[1]),
        ("latency_s", "Latency (s) / question", axes[2]),
    ]
    for key, title, ax in panels:
        ax.bar(range(len(rows)), [r[key] or 0 for r in rows], color="#4c78a8")
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def build_report(run_dir: str | Path) -> Optional[Dict[str, Path]]:
    """Write ``results.md``, ``results.csv`` and ``plot.png`` next to the logs."""
    run_dir = Path(run_dir)
    logs = read_logs(run_dir)
    if not logs:
        raise FileNotFoundError(f"no run logs (*.jsonl) found in {run_dir}")
    rows = build_table(logs)
    md, csv_path, png = run_dir / "results.md", run_dir / "results.csv", run_dir / "plot.png"
    md.write_text(to_markdown(rows))
    to_csv(rows, csv_path)
    written = {"markdown": md, "csv": csv_path}
    if plot(rows, png):
        written["plot"] = png
    return written
