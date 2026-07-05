from collections import Counter
from synth_utils import ClassBalancedSampler


def test_sampler_balances_across_classes():
    # 3 classes, plenty of items each; drawing 300 picks should be ~even
    pool = {0: ["a", "b"], 1: ["c", "d"], 2: ["e", "f"]}
    s = ClassBalancedSampler(pool, seed=1)
    picks = s.sample(300)
    counts = Counter(cid for cid, _ in picks)
    assert set(counts) == {0, 1, 2}
    assert max(counts.values()) - min(counts.values()) <= 1  # balanced to within 1


def test_sampler_returns_valid_items():
    pool = {5: ["only.png"]}
    s = ClassBalancedSampler(pool, seed=0)
    assert s.sample(3) == [(5, "only.png"), (5, "only.png"), (5, "only.png")]


def test_sampler_state_persists_across_calls():
    pool = {0: ["a"], 1: ["b"]}
    s = ClassBalancedSampler(pool, seed=0)
    first = s.sample(1)[0][0]
    second = s.sample(1)[0][0]
    assert first != second  # least-used class is chosen next
