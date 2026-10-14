"""Run an experiment: every memory strategy over every example, logging everything."""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path
from typing import Dict, List

import yaml

from .config import ExperimentConfig
from .memory import build_memory, memory_label
from .metrics import aggregate, contains_match, exact_match, retrieval_hit
from .models import build_model
from .schema import Example, load_jsonl

ANSWER_SYSTEM = (
    "Answer the question using only the conversation memory provided. "
    "Reply with just the answer, as briefly as possible. If the memory does not contain "
    "the answer, reply 'unknown'."
)


def build_prompt(context: str, question: str) -> List[dict]:
    memory = context if context else "(empty)"
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": f"Memory:\n{memory}\n\nQuestion: {question}\nAnswer:"},
    ]


def _slug(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_")


def run_example(memory, model, ex: Example, max_answer_tokens: int, label: str) -> dict:
    """Score one example and return the full log record, prompt and response included."""
    memory.reset()
    memory.usage.reset()
    t0 = time.perf_counter()
    for turn in ex.turns:
        memory.add_turn(turn)
    ctx = memory.retrieve(ex.question)
    memory_wall = time.perf_counter() - t0

    prompt = build_prompt(ctx.text, ex.question)
    out = model.generate(prompt, max_tokens=max_answer_tokens)
    return {
        "example_id": ex.id,
        "memory": label,
        "question": ex.question,
        "gold": ex.answer,
        "prediction": out.text,
        "correct": contains_match(out.text, ex.answer),
        "exact_match": exact_match(out.text, ex.answer),
        "prompt": prompt,
        "response": out.text,
        "input_tokens": out.input_tokens,
        "output_tokens": out.output_tokens,
        "latency_s": out.latency_s,
        "memory_input_tokens": memory.usage.input_tokens,
        "memory_output_tokens": memory.usage.output_tokens,
        "memory_latency_s": memory_wall,
        "memory_calls": memory.usage.calls,
        "retrieved_turn_ids": ctx.retrieved_turn_ids,
        "evidence": ex.evidence,
        "retrieval_hit": retrieval_hit(ctx.retrieved_turn_ids, ex.evidence),
    }


def run_experiment(cfg: ExperimentConfig, progress=None) -> Dict[str, dict]:
    """Run ``cfg`` and write raw logs plus a summary to ``cfg.run_dir``.

    Returns ``{memory_label: aggregate_metrics}``. Raw per-example logs are written as
    ``<label>.jsonl`` and are the source of truth for :mod:`milner.report`.
    """
    examples = load_jsonl(cfg.dataset)
    if cfg.limit:
        examples = examples[: cfg.limit]
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.raw, sort_keys=False))

    labels = [memory_label(m) for m in cfg.memories]
    if len(set(labels)) != len(labels):
        raise ValueError(f"memory labels must be unique, got {labels}; add a 'label' key")

    results: Dict[str, dict] = {}
    for spec, label in zip(cfg.memories, labels):
        # Reseed and rebuild per strategy so each one sees identical conditions.
        random.seed(cfg.seed)
        model = build_model(cfg.model, seed=cfg.seed)
        memory = build_memory(spec, model=model)
        records = []
        log_path = run_dir / f"{_slug(label)}.jsonl"
        with open(log_path, "w", encoding="utf-8") as fh:
            for i, ex in enumerate(examples):
                rec = run_example(memory, model, ex, cfg.max_answer_tokens, label)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                records.append(rec)
                if progress:
                    progress(label, i + 1, len(examples))
        results[label] = aggregate(records)

    (run_dir / "summary.json").write_text(json.dumps(results, indent=2))
    return results


def read_logs(run_dir: str | Path) -> Dict[str, List[dict]]:
    """Load every ``*.jsonl`` log in ``run_dir``, keyed by memory label."""
    from .schema import iter_jsonl

    logs: Dict[str, List[dict]] = {}
    for path in sorted(Path(run_dir).glob("*.jsonl")):
        records = list(iter_jsonl(path))
        if records:
            logs[records[0]["memory"]] = records
    summary = Path(run_dir) / "summary.json"
    if summary.exists():  # keep the order strategies were listed in the config
        order = list(json.loads(summary.read_text()))
        logs = {k: logs[k] for k in order if k in logs} | {
            k: v for k, v in logs.items() if k not in order
        }
    return logs
