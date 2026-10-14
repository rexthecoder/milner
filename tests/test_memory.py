from milner.memory import FullContext, NoMemory, Retrieval, Summary, build_memory, memory_label
from milner.models import MockModel
from milner.schema import Turn


def turns(n):
    return [Turn(f"t{i}", "user", f"filler number {i} about nothing") for i in range(n)]


def feed(mem, ts):
    mem.reset()
    for t in ts:
        mem.add_turn(t)


def test_no_memory_is_empty():
    mem = NoMemory()
    feed(mem, turns(5))
    ctx = mem.retrieve("anything")
    assert ctx.text == "" and ctx.retrieved_turn_ids == []


def test_full_context_keeps_everything_in_order():
    mem = FullContext()
    feed(mem, turns(5))
    ctx = mem.retrieve("q")
    assert ctx.retrieved_turn_ids == [f"t{i}" for i in range(5)]
    assert ctx.text.splitlines()[0].startswith("[t0]")


def test_full_context_max_turns_keeps_the_most_recent():
    mem = FullContext(max_turns=2)
    feed(mem, turns(5))
    assert mem.retrieve("q").retrieved_turn_ids == ["t3", "t4"]


def test_retrieval_finds_the_relevant_turn():
    ts = turns(20)
    ts[7] = Turn("t7", "user", "By the way, my favorite color is teal.")
    mem = Retrieval(top_k=2)
    feed(mem, ts)
    ctx = mem.retrieve("What is my favorite color?")
    assert "t7" in ctx.retrieved_turn_ids
    assert len(ctx.retrieved_turn_ids) == 2
    # shown in conversation order, not score order
    assert ctx.retrieved_turn_ids == sorted(ctx.retrieved_turn_ids, key=lambda s: int(s[1:]))


def test_retrieval_is_deterministic():
    ts = turns(15)
    a, b = Retrieval(top_k=3), Retrieval(top_k=3)
    feed(a, ts)
    feed(b, ts)
    assert a.retrieve("filler").retrieved_turn_ids == b.retrieve("filler").retrieved_turn_ids


def test_retrieval_empty_store():
    mem = Retrieval()
    mem.reset()
    assert mem.retrieve("q").text == ""


def test_summary_charges_model_usage_and_flushes_remainder():
    mem = Summary(model=MockModel(), chunk_size=4)
    feed(mem, turns(10))  # two full chunks folded eagerly, 2 turns pending
    assert mem.usage.calls == 2
    ctx = mem.retrieve("q")
    assert mem.usage.calls == 3  # remainder folded at retrieval time
    assert mem.usage.input_tokens > 0 and mem.usage.output_tokens > 0
    assert ctx.retrieved_turn_ids is None


def test_summary_is_lossy_with_the_mock():
    mem = Summary(model=MockModel(summary_words=15), chunk_size=5)
    ts = turns(30)
    ts[0] = Turn("t0", "user", "By the way, my pet name is Comet.")
    feed(mem, ts)
    assert "Comet" not in mem.retrieve("q").text


def test_reset_clears_state():
    mem = Retrieval()
    feed(mem, turns(5))
    mem.reset()
    assert mem.retrieve("q").retrieved_turn_ids == []


def test_build_memory_and_labels():
    mem = build_memory({"type": "retrieval", "top_k": 5}, model=None)
    assert isinstance(mem, Retrieval) and mem.top_k == 5
    assert memory_label({"type": "retrieval", "top_k": 5}) == "retrieval(top_k=5)"
    assert memory_label({"type": "none"}) == "none"
    assert memory_label({"type": "full", "label": "everything"}) == "everything"


def test_unknown_memory_type():
    import pytest

    with pytest.raises(ValueError, match="unknown memory"):
        build_memory({"type": "nope"})
