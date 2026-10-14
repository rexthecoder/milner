"""Experiment configuration: one YAML file per experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class ExperimentConfig:
    name: str
    dataset: str
    model: dict
    memories: List[dict]
    seed: int = 0
    output_dir: str = "runs"
    limit: Optional[int] = None
    max_answer_tokens: int = 32
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.name


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    for key in ("name", "dataset", "model", "memories"):
        if key not in raw:
            raise ValueError(f"{path}: missing required key {key!r}")
    if "type" not in raw["model"]:
        raise ValueError(f"{path}: model needs a 'type'")
    if not raw["memories"] or any("type" not in m for m in raw["memories"]):
        raise ValueError(f"{path}: every entry in 'memories' needs a 'type'")
    # Paths in a config are relative to the config file, so it works from any directory.
    base = path.resolve().parent
    dataset = Path(raw["dataset"])
    if not dataset.is_absolute():
        dataset = base / dataset
    out = Path(raw.get("output_dir", "runs"))
    if not out.is_absolute():
        out = base / out
    out = out.resolve()
    return ExperimentConfig(
        name=str(raw["name"]),
        dataset=str(dataset),
        model=raw["model"],
        memories=raw["memories"],
        seed=int(raw.get("seed", 0)),
        output_dir=str(out),
        limit=raw.get("limit"),
        max_answer_tokens=int(raw.get("max_answer_tokens", 32)),
        raw=raw,
    )
