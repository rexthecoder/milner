"""The standard task schema.

Every benchmark is converted to JSONL, one :class:`Example` per line::

    {"id": "ex-0",
     "session_id": "s-0",
     "turns": [{"turn_id": "t0", "role": "user", "content": "..."}],
     "question": "What is my favorite color?",
     "answer": "teal",
     "evidence": ["t7"]}

``evidence`` lists the ``turn_id`` values that contain the information needed to answer,
which is what the retrieval hit rate is computed against.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Union

PathLike = Union[str, Path]


@dataclass(frozen=True)
class Turn:
    turn_id: str
    role: str
    content: str

    def render(self) -> str:
        return f"[{self.turn_id}] {self.role}: {self.content}"


@dataclass(frozen=True)
class Example:
    id: str
    turns: list[Turn]
    question: str
    answer: str
    evidence: list[str] = field(default_factory=list)
    session_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Example:
        missing = {"id", "turns", "question", "answer"} - set(data)
        if missing:
            raise ValueError(f"example is missing required fields: {sorted(missing)}")
        turns = []
        for i, t in enumerate(data["turns"]):
            if "content" not in t:
                raise ValueError(f"example {data['id']!r}: turn {i} has no 'content'")
            turns.append(
                Turn(
                    turn_id=str(t.get("turn_id", f"t{i}")),
                    role=str(t.get("role", "user")),
                    content=str(t["content"]),
                )
            )
        ids = [t.turn_id for t in turns]
        if len(set(ids)) != len(ids):
            raise ValueError(f"example {data['id']!r}: duplicate turn_id values")
        evidence = [str(e) for e in data.get("evidence", [])]
        unknown = [e for e in evidence if e not in ids]
        if unknown:
            raise ValueError(f"example {data['id']!r}: evidence not in turns: {unknown}")
        return cls(
            id=str(data["id"]),
            turns=turns,
            question=str(data["question"]),
            answer=str(data["answer"]),
            evidence=evidence,
            session_id=str(data.get("session_id", data["id"])),
        )


def load_jsonl(path: PathLike) -> list[Example]:
    """Load and validate a JSONL benchmark file."""
    examples = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(Example.from_dict(json.loads(line)))
            except (ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return examples


def dump_jsonl(examples: Iterable[Example], path: PathLike) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")


def iter_jsonl(path: PathLike) -> Iterator[dict]:
    """Iterate raw dict records of any JSONL file (used for run logs)."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
