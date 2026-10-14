from milner.metrics import aggregate, contains_match, exact_match, retrieval_hit


def test_exact_match_normalises_case_punctuation_articles():
    assert exact_match("The Teal.", "teal")
    assert not exact_match("teal green", "teal")


def test_contains_match_is_whole_word():
    assert contains_match("I think it is teal!", "teal")
    assert not contains_match("tealish", "teal")
    assert not contains_match("anything", "")


def test_retrieval_hit_fraction_and_none_cases():
    assert retrieval_hit(["t1", "t2"], ["t2"]) == 1.0
    assert retrieval_hit(["t1"], ["t2", "t3"]) == 0.0
    assert retrieval_hit(["t2"], ["t2", "t3"]) == 0.5
    assert retrieval_hit(None, ["t2"]) is None
    assert retrieval_hit(["t1"], []) is None


def _rec(correct, hit, memory_in=0):
    return {
        "correct": correct,
        "exact_match": correct,
        "input_tokens": 100,
        "memory_input_tokens": memory_in,
        "output_tokens": 2,
        "memory_output_tokens": 0,
        "latency_s": 0.5,
        "memory_latency_s": 0.1,
        "retrieval_hit": hit,
    }


def test_aggregate_charges_memory_overhead_and_skips_missing_hits():
    out = aggregate([_rec(True, 1.0, 50), _rec(False, None, 50)])
    assert out["n"] == 2 and out["accuracy"] == 0.5
    assert out["input_tokens"] == 150  # answer call + memory overhead
    assert abs(out["latency_s"] - 0.6) < 1e-9
    assert out["retrieval_hit_rate"] == 1.0  # None excluded, not counted as 0
    assert aggregate([_rec(True, None)])["retrieval_hit_rate"] is None
