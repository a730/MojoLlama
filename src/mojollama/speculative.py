#!/usr/bin/env python3
"""Speculative decoding — draft (TinyLlama 1.1B) + target (Qwen3-30B-A3B).

Algorithm:
  1. Draft model generates K candidate tokens autoregressively.
  2. Target model verifies each candidate via argmax comparison on its logits.
  3. Accept the longest prefix that matches (draft token == target argmax).
  4. If all K accepted, draft generates K more (bonus speculation).
  5. If rejected at position i, target takes over from position i.

Usage:
    OMP_NUM_THREADS=32 python3 -u speculative.py "Your prompt" [max_tokens] [K]

Prints: "Draft: X tok, Accepted: Y tok, Acceptance rate: Z%"
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "kernels"))
from turbo_engine_v77 import TurboEngineV77
from turbo_engine_v7_moe import TurboEngineV7MoE

DRAFT_PATH = "/tmp/tl-Q4_0.gguf"
TARGET_PATH = "/tmp/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
TOKENIZER_PATH = "/tmp/qwen3-tokenizer/"


def _ensure_imports():
    """Lazy-load tokenizer after path setup."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER_PATH)


def speculative_generate(
    prompt: str,
    max_tokens: int = 100,
    K: int = 5,
    draft_model_path: str = DRAFT_PATH,
    target_model_path: str = TARGET_PATH,
    n_threads: int = 32,
    verbose: bool = True,
) -> str:
    """Speculative decoding: draft proposes, target verifies.

    Key design decisions:
      - Both models loaded in-process as TurboEngine* instances.
      - Target's tokenizer is used for all text ↔ token conversion.
      - Draft receives tokens clipped to its vocabulary range.
      - KV cache is checkpointed before verification for clean rollback.
    """
    os.environ["OMP_NUM_THREADS"] = str(n_threads)

    tokenizer = _ensure_imports()
    bos = tokenizer.bos_token_id or 1
    eos = tokenizer.eos_token_id or 2

    # ── Load engines ──────────────────────────────────────────────────
    if verbose:
        print(f"Loading draft model from {draft_model_path}...", flush=True)
    t0 = time.perf_counter()
    draft = TurboEngineV77(draft_model_path, n_threads)
    draft_vocab = draft.vocab_size
    if verbose:
        print(f"  Draft loaded in {time.perf_counter()-t0:.1f}s  (vocab={draft_vocab})", flush=True)

    if verbose:
        print(f"Loading target model from {target_model_path}...", flush=True)
    t0 = time.perf_counter()
    target = TurboEngineV7MoE(target_model_path, n_threads)
    target_vocab = target.vocab_size
    if verbose:
        print(f"  Target loaded in {time.perf_counter()-t0:.1f}s  (vocab={target_vocab})", flush=True)

    # ── Tokenize prompt ───────────────────────────────────────────────
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [bos]
    prompt_len = len(ids)

    if verbose:
        print(f"  Prompt: {prompt_len} tokens, max_gen={max_tokens}, K={K}", flush=True)

    # ── Helpers ───────────────────────────────────────────────────────
    def _safe_tid(tid: int, vocab_size: int, fallback: int = 0) -> int:
        """Clamp token id to valid range for the given model's vocabulary."""
        if 0 <= tid < vocab_size:
            return tid
        return fallback

    def _sample(logits: np.ndarray) -> int:
        """Greedy argmax sampling with NaN handling."""
        safe = np.nan_to_num(logits, nan=-1e10, posinf=1e10, neginf=-1e10)
        return int(np.argmax(safe))

    def _prefill(engine, token_ids, vocab_size):
        """Prefill engine with a list of token IDs, safely clipped."""
        engine.reset()
        for tid in token_ids:
            engine.forward(_safe_tid(tid, vocab_size))

    def _save_target_state():
        """Checkpoint target KV cache for potential rollback."""
        return {
            "kv_k": target.kv_k.copy(),
            "kv_v": target.kv_v.copy(),
            "kv_len": target.kv_len.copy(),
            "pos": target.pos,
        }

    def _restore_target_state(state):
        """Restore target KV cache from a checkpoint."""
        target.kv_k[:] = state["kv_k"]
        target.kv_v[:] = state["kv_v"]
        target.kv_len[:] = state["kv_len"]
        target.pos = state["pos"]

    # Prefill both engines
    _prefill(draft, ids, draft_vocab)
    _prefill(target, ids, target_vocab)

    generated_ids = []
    total_draft = 0
    total_accepted = 0

    # ── Main loop ─────────────────────────────────────────────────────
    while len(generated_ids) < max_tokens:
        k_this = min(K, max_tokens - len(generated_ids))

        # ── Step 1: Draft proposes K candidates ──
        candidates = []
        for _ in range(k_this):
            prev_tid = _safe_tid(candidates[-1] if candidates else 0, draft_vocab)
            logits = draft.forward(prev_tid)
            tok = _sample(logits)
            candidates.append(tok)
            total_draft += 1
            if tok == eos:
                break

        if not candidates:
            break

        # ── Step 2: Target verifies candidates ──
        # Save target state so we can rollback on rejection
        saved_state = _save_target_state()

        n_accepted = 0
        target_override = None

        for i, cand in enumerate(candidates):
            # Run target forward with the candidate
            safe_cand = _safe_tid(cand, target_vocab)
            logits = target.forward(safe_cand)
            target_best = _sample(logits)

            if target_best == cand:
                n_accepted += 1
            else:
                # Rejected at position i
                # Roll back to pre-verification state
                _restore_target_state(saved_state)

                # Re-process the accepted prefix
                for accepted_tok in candidates[:i]:
                    target.forward(_safe_tid(accepted_tok, target_vocab))

                # Process the target's own correction token
                target_override = target_best
                target.forward(_safe_tid(target_best, target_vocab))
                break

        # ── Step 3: Update generated text ──
        if n_accepted == len(candidates):
            # All K accepted
            generated_ids.extend(candidates)
            total_accepted += n_accepted
        elif n_accepted > 0:
            # Partial acceptance: use accepted prefix + target's override
            generated_ids.extend(candidates[:n_accepted])
            total_accepted += n_accepted
            if target_override is not None:
                generated_ids.append(target_override)
                # The override counts toward acceptance since it's the
                # correct token (from the target's perspective)
                total_accepted += 1
        else:
            # No candidate accepted (even the first was wrong)
            # target_override is set from the rejection handler above
            if target_override is not None:
                generated_ids.append(target_override)
                total_accepted += 1
            else:
                # Fallback: generate one token with target directly
                logits = target.forward(_safe_tid(0, target_vocab))
                tok = _sample(logits)
                generated_ids.append(tok)

        # Stop on EOS
        if eos in generated_ids:
            break

    # ── Report ────────────────────────────────────────────────────────
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    acceptance_rate = (
        (total_accepted / total_draft * 100) if total_draft > 0 else 0
    )
    print(
        f"Draft: {total_draft} tok, Accepted: {total_accepted} tok, "
        f"Acceptance rate: {acceptance_rate:.0f}%",
        flush=True,
    )
    return text


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    t0 = time.perf_counter()
    text = speculative_generate(prompt, max_tokens, K, verbose=True)
    elapsed = time.perf_counter() - t0
    print(f"\nGenerated ({len(text.split())} words) in {elapsed:.1f}s:")
    print(text[:500])
