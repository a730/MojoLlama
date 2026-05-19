#!/usr/bin/env python3
"""Prefix caching for KV cache reuse across requests.

Stores KV cache snapshots indexed by a hash of the first N token prefix.
On match, the cached KV data can be copied into a new KVBlock, advancing
the sequence length to the cached length before processing remaining tokens.

Usage:
    from mojollama.prefix_cache import PrefixCache
    cache = PrefixCache(max_entries=64, prefix_len=8)

    # Store a KV snapshot for a given token prefix
    cache.put(tokens_tuple, kv_snapshot)

    # Retrieve — returns kv_snapshot or None
    match = cache.get(tokens_tuple)
"""

import hashlib
from collections import OrderedDict
from typing import Optional, Any, Tuple, List


class PrefixCache:
    """LRU prefix cache for KV block reuse.

    The cache stores snapshots of KV cache data keyed by the first N tokens
    of a prompt.  When a new request has a matching prefix, the cached state
    can be restored and only the remaining (non-matching) tokens need to be
    processed.

    Attributes:
        max_entries: Maximum number of prefixes to cache (LRU eviction).
        prefix_len:  Number of leading tokens used to compute the hash key.
    """

    def __init__(self, max_entries: int = 64, prefix_len: int = 8):
        self.max_entries = max_entries
        self.prefix_len = prefix_len
        # _cache: hash_key -> (full_tokens_tuple, kv_snapshot)
        # OrderedDict gives LRU ordering (front = LRU, back = MRU).
        self._cache: "OrderedDict[str, Tuple[Tuple[int, ...], Any]]" = OrderedDict()

    # ── Public API ──────────────────────────────────────────────────────

    def put(self, tokens_tuple: Tuple[int, ...], kv_snapshot: Any) -> None:
        """Store a KV cache snapshot keyed by the prefix of *tokens_tuple*.

        The first ``prefix_len`` tokens are hashed to create the cache key.
        If the cache is full, the least recently used entry is evicted first.
        If an entry with the same prefix already exists, it is updated and
        promoted to MRU.
        """
        if not tokens_tuple:
            return
        prefix = tokens_tuple[: self.prefix_len]
        key = self._hash(prefix)

        if key in self._cache:
            # Update existing entry — promote to MRU
            self._cache.move_to_end(key)
            self._cache[key] = (tokens_tuple, kv_snapshot)
            return

        # Evict LRU if needed, then insert
        self._evict_if_needed()
        self._cache[key] = (tokens_tuple, kv_snapshot)

    def get(self, tokens_tuple: Tuple[int, ...]) -> Optional[Any]:
        """Return the cached KV snapshot if the prefix matches.

        Returns ``None`` on cache miss.  On hit the entry is promoted to
        MRU (most recently used) position.

        Matching rules:
          - If the query has ``prefix_len`` or more tokens, the first
            ``prefix_len`` tokens must exactly match a cached entry (O(1)
            hash lookup).
          - If the query has fewer than ``prefix_len`` tokens (partial
            prefix match), all available tokens must match the start of a
            cached prefix (linear scan over all entries).
        """
        if not tokens_tuple:
            return None

        n_query = min(self.prefix_len, len(tokens_tuple))
        query_prefix = tokens_tuple[:n_query]

        if n_query == self.prefix_len:
            # Full-length prefix — O(1) hash lookup
            key = self._hash(query_prefix)
            if key in self._cache:
                stored_tokens = self._cache[key][0]
                if stored_tokens[: self.prefix_len] == query_prefix:
                    self._cache.move_to_end(key)
                    return self._cache[key][1]
                # Hash collision (astronomically unlikely with SHA-256) —
                # fall through to linear scan below
            return None

        # Partial prefix match (query shorter than prefix_len) — linear scan
        for key, (cached_tokens, snapshot) in list(self._cache.items()):
            if cached_tokens[:n_query] == query_prefix:
                self._cache.move_to_end(key)
                return snapshot

        return None

    def evict(self) -> int:
        """Evict the least recently used entry.

        Returns the number of entries evicted (0 or 1).
        """
        if not self._cache:
            return 0
        self._cache.popitem(last=False)  # LRU is at front
        return 1

    def invalidate(self, tokens_tuple: Tuple[int, ...]) -> bool:
        """Remove the cache entry for a specific prefix.

        Returns True if an entry was removed, False otherwise.
        """
        if not tokens_tuple:
            return False
        prefix = tokens_tuple[: self.prefix_len]
        key = self._hash(prefix)
        if key in self._cache:
            stored_tokens, _ = self._cache[key]
            if stored_tokens[: self.prefix_len] == prefix:
                del self._cache[key]
                return True
        return False

    def clear(self) -> None:
        """Remove all entries from the cache."""
        self._cache.clear()

    @property
    def size(self) -> int:
        """Number of entries currently cached."""
        return len(self._cache)

    def keys(self) -> List[Tuple[int, ...]]:
        """Return the stored token prefixes (for inspection/debug)."""
        return list(t for t, _ in self._cache.values())

    # ── Internal helpers ────────────────────────────────────────────────

    @staticmethod
    def _hash(prefix: Tuple[int, ...]) -> str:
        """Deterministic SHA-256 hash of a token prefix tuple."""
        raw = b",".join(t.to_bytes(4, "little", signed=True) for t in prefix)
        return hashlib.sha256(raw).hexdigest()

    def _evict_if_needed(self) -> None:
        """Evict one LRU entry if at capacity."""
        if len(self._cache) >= self.max_entries:
            self._cache.popitem(last=False)


# ── Engine-specific helpers ──────────────────────────────────────────
# These functions create / restore KV snapshots for the concrete KV
# cache layouts used by TurboEngineV77 (dense) and TurboEngineV7MoE.

def snapshot_dense(engine) -> dict:
    """Create a KV snapshot from a TurboEngineV77 (dense) instance.

    Returns a dict containing copies of the KV caches and position info.
    """
    max_pos = engine.kvk.shape[1]
    snapshot = {
        "kvk": engine.kvk[:, :max_pos, :].copy(),
        "kvv": engine.kvv[:, :max_pos, :].copy(),
        "kvl": engine.kvl.copy(),
        "pos": int(engine.pos),
    }
    return snapshot


def restore_dense(engine, snapshot: dict) -> None:
    """Restore KV cache state on a TurboEngineV77 from a snapshot."""
    sl = int(snapshot["kvl"].max()) if snapshot["kvl"].any() else 0
    engine.kvk[:, :, :] = 0
    engine.kvv[:, :, :] = 0
    engine.kvk[:, :sl, :] = snapshot["kvk"][:, :sl, :]
    engine.kvv[:, :sl, :] = snapshot["kvv"][:, :sl, :]
    engine.kvl[:] = snapshot["kvl"]
    engine.pos = snapshot["pos"]


def snapshot_moe(engine) -> dict:
    """Create a KV snapshot from a TurboEngineV7MoE instance."""
    max_pos = engine.kv_k.shape[1]
    return {
        "kv_k": engine.kv_k[:, :max_pos, :].copy(),
        "kv_v": engine.kv_v[:, :max_pos, :].copy(),
        "kv_len": engine.kv_len.copy(),
        "pos": int(engine.pos),
    }


def restore_moe(engine, snapshot: dict) -> None:
    """Restore KV cache state on a TurboEngineV7MoE from a snapshot."""
    sl = int(snapshot["kv_len"].max()) if snapshot["kv_len"].any() else 0
    engine.kv_k[:, :, :] = 0
    engine.kv_v[:, :, :] = 0
    engine.kv_k[:, :sl, :] = snapshot["kv_k"][:, :sl, :]
    engine.kv_v[:, :sl, :] = snapshot["kv_v"][:, :sl, :]
    engine.kv_len[:] = snapshot["kv_len"]
    engine.pos = snapshot["pos"]


if __name__ == "__main__":
    # Quick smoke test
    cache = PrefixCache(max_entries=4, prefix_len=4)
    cache.put((1, 2, 3, 4, 5, 6), "snapshot_A")
    cache.put((1, 2, 3, 7, 8, 9), "snapshot_B")
    assert cache.get((1, 2, 3, 4, 10, 11)) == "snapshot_A"
    assert cache.get((1, 2, 3, 7, 42, 99)) == "snapshot_B"
    assert cache.get((9, 9, 9, 9)) is None
    print("PrefixCache smoke test PASSED")
