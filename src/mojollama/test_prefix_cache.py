#!/usr/bin/env python3
"""Tests for PrefixCache.

Run:  OMP_NUM_THREADS=32 python3 -u test_prefix_cache.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mojollama.prefix_cache import PrefixCache


def test_basic_put_get():
    """Basic put followed by get with matching prefix."""
    cache = PrefixCache(max_entries=16, prefix_len=4)
    tokens_a = (1, 100, 200, 300, 400, 500)
    cache.put(tokens_a, "snap_a")
    # Full match
    assert cache.get((1, 100, 200, 300, 400, 500, 600)) == "snap_a"
    # Partial suffix different but prefix matches
    assert cache.get((1, 100, 200, 300, 999, 888)) == "snap_a"
    print("  [PASS] test_basic_put_get")


def test_cache_miss():
    """Cache miss when prefix doesn't match."""
    cache = PrefixCache(max_entries=16, prefix_len=4)
    cache.put((10, 20, 30, 40, 50), "snap_x")
    assert cache.get((99, 98, 97, 96)) is None
    assert cache.get((10, 20, 30, 41)) is None  # last token differs
    print("  [PASS] test_cache_miss")


def test_lru_eviction():
    """LRU eviction when cache is full."""
    cache = PrefixCache(max_entries=3, prefix_len=2)
    cache.put((1, 10, 100), "A")
    cache.put((2, 20, 200), "B")
    cache.put((3, 30, 300), "C")
    assert cache.size == 3

    # Access A to make it MRU
    cache.get((1, 10, 42))
    # Add D — should evict B (LRU)
    cache.put((4, 40, 400), "D")
    assert cache.size == 3
    assert cache.get((2, 20, 42)) is None  # B evicted
    assert cache.get((1, 10, 42)) == "A"    # A still there
    assert cache.get((4, 40, 42)) == "D"    # D added

    # Exhaustive: fill and evict one by one
    cache2 = PrefixCache(max_entries=2, prefix_len=2)
    cache2.put((5, 50, 500), "E")
    cache2.put((6, 60, 600), "F")
    cache2.put((7, 70, 700), "G")  # evicts E
    assert cache2.size == 2
    assert cache2.get((5, 50, 42)) is None  # E evicted
    assert cache2.get((7, 70, 42)) == "G"
    print("  [PASS] test_lru_eviction")


def test_partial_prefix_match():
    """Partial prefix matching — request shorter than prefix_len."""
    cache = PrefixCache(max_entries=16, prefix_len=8)
    # Store an entry with 12 tokens (prefix_len=8)
    long_tokens = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    cache.put(long_tokens, "snap_long")

    # A shorter sequence whose first N tokens match
    # (where N < prefix_len) should still match
    short_tokens = (1, 2, 3, 4)  # only 4 tokens
    # The cache should match as long as the available tokens are a prefix
    match = cache.get(short_tokens)
    assert match == "snap_long", f"Expected snap_long, got {match}"
    print("  [PASS] test_partial_prefix_match")


def test_multiple_prefix_lengths():
    """Multiple cache entries with different prefix lengths."""
    cache = PrefixCache(max_entries=16, prefix_len=8)
    # Entry A: first 8 tokens = 1-8
    cache.put((1, 2, 3, 4, 5, 6, 7, 8, 100, 200), "A")
    # Entry B: first 8 tokens = 10-18
    cache.put((10, 11, 12, 13, 14, 15, 16, 17, 300), "B")
    # Entry C: same first 8 tokens as A but different continuation
    cache.put((1, 2, 3, 4, 5, 6, 7, 8, 999, 888), "C")

    # A and C share same 8-token prefix → get should return one of them
    # (the most recently accessed)
    match = cache.get((1, 2, 3, 4, 5, 6, 7, 8, 42, 43))
    assert match is not None, "Should match either A or C"

    # B should match its own prefix
    match_b = cache.get((10, 11, 12, 13, 14, 15, 16, 17, 55))
    assert match_b == "B"

    # Non-matching prefix
    assert cache.get((99, 98, 97, 96, 95, 94, 93, 92)) is None
    print("  [PASS] test_multiple_prefix_lengths")


def test_invalidate():
    """invalidate removes specific entry."""
    cache = PrefixCache(max_entries=16, prefix_len=4)
    cache.put((1, 2, 3, 4, 77), "X")
    cache.put((5, 6, 7, 8, 88), "Y")
    assert cache.get((1, 2, 3, 4, 99)) == "X"

    # Invalidate X
    assert cache.invalidate((1, 2, 3, 4, 77)) is True
    assert cache.get((1, 2, 3, 4, 99)) is None  # X gone
    assert cache.get((5, 6, 7, 8, 55)) == "Y"   # Y still there
    print("  [PASS] test_invalidate")


def test_evict_method():
    """Evict explicitly removes the LRU entry."""
    cache = PrefixCache(max_entries=3, prefix_len=2)
    cache.put((1, 10), "A")
    cache.put((2, 20), "B")
    cache.put((3, 30), "C")
    # Access B to make it MRU, A is LRU
    cache.get((2, 20))
    assert cache.evict() == 1  # evicts A
    assert cache.get((1, 10, 42)) is None
    assert cache.size == 2
    print("  [PASS] test_evict_method")


def test_clear():
    """Clear removes everything."""
    cache = PrefixCache(max_entries=16, prefix_len=4)
    cache.put((1, 2, 3, 4), "A")
    cache.put((5, 6, 7, 8), "B")
    assert cache.size == 2
    cache.clear()
    assert cache.size == 0
    assert cache.get((1, 2, 3, 4, 99)) is None
    print("  [PASS] test_clear")


def test_empty_cache():
    """Get on empty cache returns None."""
    cache = PrefixCache()
    assert cache.get((1, 2, 3)) is None
    assert cache.get(()) is None
    assert cache.evict() == 0
    print("  [PASS] test_empty_cache")


def test_update_same_prefix():
    """Put with same prefix updates the stored snapshot."""
    cache = PrefixCache(max_entries=4, prefix_len=4)
    cache.put((1, 2, 3, 4, 100), "old")
    # Update with same prefix but new value
    cache.put((1, 2, 3, 4, 200), "new")
    assert cache.get((1, 2, 3, 4, 999)) == "new"
    assert cache.size == 1  # no duplicate
    print("  [PASS] test_update_same_prefix")


if __name__ == "__main__":
    print("PrefixCache tests:")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("\nAll PrefixCache tests PASSED")
