"""MojoLlama Sampling — top-k, top-p, min-p, temperature, repetition penalty.

llama.cpp's sampling pipeline (ordered as applied):
1. Repetition penalty: penalize tokens that appeared recently
2. Temperature: scale logits before sampling
3. Top-k: keep only top k candidates
4. Top-p (nucleus): keep smallest set with cumulative prob >= p
5. Min-p: filter tokens with prob < min_p * max_prob
6. Greedy/argmax: for temperature=0

All operations are SIMD-friendly where possible.
This module provides both F32 array operations (for Mojo kernels)
and Python-callable wrappers (for the server).
"""

from std.math import exp, log, sqrt
from std.memory.unsafe_pointer import alloc


# ─── Temperature Scaling ───────────────────────────────────────────────

fn apply_temperature(
    logits: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    temperature: Float32,
):
    """Scale logits by 1/temperature. temperature=0 → greedy (handled by caller).
    Temperature < 1.0 = sharper (more peaked), > 1.0 = flatter (more random).
    """
    if temperature < 1e-10:
        return  # Caller should use argmax for greedy
    
    var inv_temp = 1.0 / temperature
    var inv_v = SIMD[DType.float32, 1](inv_temp)
    var i = 0
    while i + 8 <= n:
        var v = logits.load[width=8](i)
        logits.store[width=8](i, v * inv_v)
        i += 8
    while i < n:
        logits.store(i, logits.load(i) * inv_temp)
        i += 1


# ─── Softmax ────────────────────────────────────────────────────────────

fn softmax_inplace(
    logits: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
):
    """In-place softmax: converts logits to probabilities.
    Uses online max for numerical stability.
    """
    # Find max
    var max_val: Float32 = -1e30
    var i = 0
    while i + 8 <= n:
        var v = logits.load[width=8](i)
        var m = v.reduce_max()
        if m > max_val: max_val = m
        i += 8
    while i < n:
        if logits.load(i) > max_val: max_val = logits.load(i)
        i += 1
    
    # Exp and sum
    var sum: Float32 = 0.0
    i = 0
    while i + 8 <= n:
        var v = logits.load[width=8](i)
        var result = SIMD[DType.float32, 8]()
        for j in range(8):
            result[j] = exp(v[j] - max_val)
        logits.store[width=8](i, result)
        sum += result.reduce_add()
        i += 8
    while i < n:
        var val = exp(logits.load(i) - max_val)
        logits.store(i, val)
        sum += val
        i += 1
    
    # Normalize
    var inv_sum = 1.0 / sum
    var inv_v = SIMD[DType.float32, 8](inv_sum)
    i = 0
    while i + 8 <= n:
        var v = logits.load[width=8](i)
        logits.store[width=8](i, v * inv_v)
        i += 8
    while i < n:
        logits.store(i, logits.load(i) * inv_sum)
        i += 1


# ─── Repetition Penalty ────────────────────────────────────────────────

fn apply_repetition_penalty(
    logits: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    recent_tokens: UnsafePointer[Int32, MutAnyOrigin],  # recently generated token IDs
    n_recent: Int,           # number of recent tokens
    penalty: Float32,        # >1.0 reduces repetition, <1.0 increases
):
    """Apply repetition penalty to logits of recently seen tokens.
    penalty > 1.0: makes recent tokens less likely (default: 1.0 = no effect)
    penalty < 1.0: makes recent tokens more likely (dangerous!)
    
    llama.cpp-style: divide positive logits by penalty, multiply negative by penalty.
    This makes recently seen tokens with positive logits much less likely.
    """
    if penalty < 1e-10 or penalty == 1.0:
        return
    
    for i in range(n_recent):
        var token_id = recent_tokens.load(i)
        if token_id >= 0 and token_id < n:
            var logit = logits.load(token_id)
            if logit > 0.0:
                logits.store(token_id, logit / penalty)
            else:
                logits.store(token_id, logit * penalty)


# ─── Top-K Filtering ────────────────────────────────────────────────────

struct TokenScore:
    var id: Int
    var score: Float32

fn top_k_filter(
    logits: UnsafePointer[Float32, MutAnyOrigin],
    candidates: UnsafePointer[TokenScore, MutAnyOrigin],
    n: Int,
    k: Int,
) -> Int:
    """Filter to top-k candidates. Sets logits below top-k to -inf.
    Returns number of valid candidates (min(k, n)).
    
    Uses partial sort for O(n) average case instead of O(n log n) full sort.
    """
    if k <= 0 or k >= n:
        # No filtering needed
        for i in range(n):
            candidates.store(i, TokenScore(i, logits.load(i)))
        return n
    
    # Find the k-th largest score using selection
    # Simple approach: collect all scores, then partial sort
    var all_scores = alloc[TokenScore](n)
    for i in range(n):
        all_scores.store(i, TokenScore(i, logits.load(i)))
    
    # Partial selection sort: find top k
    for i in range(k):
        var best_idx = i
        var best_score = all_scores.load(i).score
        for j in range(i + 1, n):
            var s = all_scores.load(j).score
            if s > best_score:
                best_score = s
                best_idx = j
        # Swap
        if best_idx != i:
            var tmp = all_scores.load(i)
            all_scores.store(i, all_scores.load(best_idx))
            all_scores.store(best_idx, tmp)
    
    # Set logits below top-k to -inf
    var kth_score = all_scores.load(k - 1).score
    for i in range(n):
        if logits.load(i) < kth_score:
            logits.store(i, -1e30)
    
    # Copy top-k candidates
    for i in range(k):
        candidates.store(i, all_scores.load(i))
    
    return k


# ─── Top-P (Nucleus) Filtering ──────────────────────────────────────────

fn top_p_filter(
    logits: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    p: Float32,
) -> Int:
    """Filter to smallest set of tokens with cumulative probability >= p.
    Applies AFTER softmax. Sets remaining tokens to -inf, requires re-softmax.
    
    Returns number of tokens in the nucleus.
    """
    if p >= 1.0 or p < 0.0:
        return n
    
    # First apply softmax to get probabilities
    softmax_inplace(logits, n)
    
    # Simple approach: find tokens sorted by probability
    # Then find cutoff where cumulative probability >= p
    # For efficiency, we sort token IDs by probability
    
    var indices = alloc[Int](n)
    var probs = alloc[Float32](n)
    for i in range(n):
        indices.store(i, i)
        probs.store(i, logits.load(i))
    
    # Sort by probability descending (selection sort for small vocab)
    for i in range(n):
        var best = i
        for j in range(i + 1, n):
            if probs.load(j) > probs.load(best):
                best = j
        # Swap
        var tmp_idx = indices.load(i)
        var tmp_prob = probs.load(i)
        indices.store(i, indices.load(best))
        probs.store(i, probs.load(best))
        indices.store(best, tmp_idx)
        probs.store(best, tmp_prob)
    
    # Find nucleus cutoff
    var cumsum: Float32 = 0.0
    var cutoff = n
    for i in range(n):
        cumsum += probs.load(i)
        if cumsum >= p:
            cutoff = i + 1
            break
    
    # Set all tokens outside nucleus to -inf
    # First, build a set of nucleus token IDs
    for i in range(n):
        logits.store(i, -1e30)  # zero all
    
    for i in range(cutoff):
        var idx = indices.load(i)
        logits.store(idx, probs.load(i))
    
    # Re-normalize (don't need full softmax, just divide by sum)
    var nucleus_sum: Float32 = 0.0
    for i in range(n):
        if logits.load(i) > -1e29:
            nucleus_sum += logits.load(i)
    
    var inv_sum = 1.0 / nucleus_sum
    for i in range(n):
        if logits.load(i) > -1e29:
            logits.store(i, logits.load(i) * inv_sum)
        else:
            logits.store(i, 0.0)
    
    return cutoff


# ─── Min-P Filtering ────────────────────────────────────────────────────

fn min_p_filter(
    logits: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    min_p: Float32,
) -> Int:
    """Filter tokens with probability < min_p * max_probability.
    min_p=0.05 means: remove all tokens with prob < 5% of the top token's prob.
    Must be called AFTER softmax.
    """
    if min_p <= 0.0 or min_p >= 1.0:
        return n
    
    # Find max probability
    softmax_inplace(logits, n)
    var max_prob: Float32 = 0.0
    for i in range(n):
        var p = logits.load(i)
        if p > max_prob: max_prob = p
    
    var threshold = max_prob * min_p
    var count = 0
    
    # Zero out tokens below threshold
    for i in range(n):
        if logits.load(i) >= threshold:
            count += 1
        else:
            logits.store(i, 0.0)
    
    # Re-normalize
    var sum: Float32 = 0.0
    for i in range(n):
        sum += logits.load(i)
    var inv_sum = 1.0 / sum
    for i in range(n):
        if logits.load(i) > 0.0:
            logits.store(i, logits.load(i) * inv_sum)
    
    return count


# ─── Greedy Sampling (argmax) ───────────────────────────────────────────

fn sample_greedy(
    logits: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
) -> Int:
    """Return the token with the highest logit (greedy/argmax decoding)."""
    var best_id = 0
    var best_score = logits.load(0)
    for i in range(1, n):
        var s = logits.load(i)
        if s > best_score:
            best_score = s
            best_id = i
    return best_id


# ─── Random Sampling ────────────────────────────────────────────────────

fn sample_random(
    probs: UnsafePointer[Float32, MutAnyOrigin],  # must be softmaxed
    n: Int,
    rng_state: UInt64,
) -> (Int, UInt64):
    """Sample from probability distribution using PCG-style LCG.
    Returns (token_id, new_rng_state).
    """
    # Simple LCG: state = state * 6364136223846793005 + 1442695040888963407
    rng_state = rng_state * 6364136223846793005 + 1442695040888963407
    # Convert to float in [0, 1)
    var rand_val = Float32((rng_state >> 11) & 0x1FFFFF) / Float32(0x1FFFFF)
    
    # Cumulative sampling
    var cumsum: Float32 = 0.0
    for i in range(n):
        cumsum += probs.load(i)
        if cumsum >= rand_val:
            return (i, rng_state)
    
    # Fallback: last token
    return (n - 1, rng_state)


# ─── Full Sampling Pipeline ────────────────────────────────────────────

struct SamplingConfig:
    var temperature: Float32
    var top_k: Int
    var top_p: Float32
    var min_p: Float32
    var repetition_penalty: Float32
    var repeat_last_n: Int      # how many recent tokens to penalize
    var seed: UInt64            # RNG seed (0 = random)

struct SamplingResult:
    var token_id: Int
    var prob: Float32           # probability of chosen token
    var n_candidates: Int      # tokens considered after filtering


fn sample(
    logits: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    config: SamplingConfig,
    recent_tokens: UnsafePointer[Int32, MutAnyOrigin],
    n_recent: Int,
) -> SamplingResult:
    """Complete sampling pipeline matching llama.cpp's sampling order:
    1. Repetition penalty
    2. Temperature scaling
    3. Top-k filtering
    4. Softmax
    5. Top-p (nucleus) filtering
    6. Min-p filtering
    7. Sample (greedy if temperature=0, random otherwise)
    """
    var result = SamplingResult(0, 0.0, n)
    
    # 1. Repetition penalty
    apply_repetition_penalty(logits, n, recent_tokens, n_recent, config.repetition_penalty)
    
    # 2. Temperature
    if config.temperature < 1e-10:
        # Greedy: just argmax
        result.token_id = sample_greedy(logits, n)
        result.prob = 1.0
        result.n_candidates = 1
        return result
    
    apply_temperature(logits, n, config.temperature)
    
    # 3. Top-k (before softmax for efficiency)
    var candidates = alloc[TokenScore](n)
    var n_valid = top_k_filter(logits, candidates, n, config.top_k)
    result.n_candidates = n_valid
    
    # 4. Softmax
    softmax_inplace(logits, n)
    
    # 5. Top-p (nucleus)
    if config.top_p < 1.0:
        n_valid = top_p_filter(logits, n, config.top_p)
        result.n_candidates = n_valid
    
    # 6. Min-p
    if config.min_p > 0.0:
        n_valid = min_p_filter(logits, n, config.min_p)
        result.n_candidates = n_valid
    
    # 7. Sample
    var rng = config.seed
    if rng == 0:
        rng = 42  # Default seed
    
    var (token_id, new_rng) = sample_random(logits, n, rng)
    result.token_id = token_id
    result.prob = logits.load(token_id)
    
    return result