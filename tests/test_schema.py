import json

import pytest

from milner.schema import Example, dump_jsonl, load_jsonl


def test_roundtrip(tmp_path, toy):
    path = tmp_path / "x.jsonl"
    dump_jsonl(toy, path)
    assert load_jsonl(path) == toy


def test_defaults_fill_in_turn_ids_and_roles():
    ex = Example.from_dict(
        {"id": "a", "turns": [{"content": "hi"}, {"content": "yo"}], "question": "q", "answer": "a"}
    )
    assert [t.turn_id for t in ex.turns] == ["t0", "t1"]
    assert ex.turns[0].role == "user"


def test_missing_field_is_rejected():
    with pytest.raises(ValueError, match="missing required"):
        Example.from_dict({"id": "a", "turns": [], "question": "q"})


def test_evidence_must_point_at_real_turns():
    bad = {
        "id": "a",
        "turns": [{"turn_id": "t0", "content": "x"}],
        "question": "q",
        "answer": "a",
        "evidence": ["t9"],
    }
    with pytest.raises(ValueError, match="evidence"):
        Example.from_dict(bad)


def test_duplicate_turn_ids_rejected():
    bad = {
        "id": "a",
        "turns": [{"turn_id": "t0", "content": "x"}, {"turn_id": "t0", "content": "y"}],
        "question": "q",
        "answer": "a",
    }
    with pytest.raises(ValueError, match="duplicate"):
        Example.from_dict(bad)


def test_error_names_file_and_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"id": "only"}) + "\n")
    with pytest.raises(ValueError, match=r"bad\.jsonl:1"):
        load_jsonl(path)
