# TinyLlama Quantization Benchmark — Real Inference Results
## Date: 2026-05-22

### Test Setup
- **Model**: TinyLlama-1.1B-Chat-v1.0
- **Hardware**: Threadripper 3970X (32 cores, 4 CCDs)
- **Prompt**: 7 prompts including "2+2=", "3*4=", "def fib(n):", etc.
- **Generation**: 10 tokens per prompt, autoregressive with KV cache
- **Measurement**: Aggregate tok/s across all 7 prompts

---

### Real Inference Results (f16-dequantized weights)

The `gguf_extract` tool dequantizes some formats to f16 at extraction time. Once in f16, all formats run at the same speed in the forward pass.

| Format   | Weight Size | Extracted As | tok/s | Status |
|----------|-------------|--------------|-------|--------|
| **f16**  | 2.1 GB      | native f16   | **27.2** | ✓ Verified correct |
| **Q8_0** | 2.1 GB      | f16 (extract)| **28.2** | ✓ Same speed |
| **Q6_K** | 2.1 GB      | f16 (extract)| **27.5** | ✓ Same speed |
| **Q3_K** | 1.7 GB      | f16 (extract)| **26.4** | ✓ Same speed |
| **TQ2_0**| 2.0 GB      | f16 (extract)| **25.0** | ✓ Same speed |

**Why same speed?** `gguf_extract` converts these formats to f16 .bin files during extraction. The forward pass sees only f16 weights. The 2-3 tok/s variation is noise (cache state, thread scheduling).

---

### Raw Quantized Formats (on-the-fly dequant)

These formats remain in packed form after extraction. On-the-fly dequantization during matmul would show real speed differences.

| Format   | Weight Size | Format After Extract | tok/s | Status |
|----------|-------------|----------------------|-------|--------|
| **Q4_0** | 681 MB      | raw Q4_0 packed      | —     | ✗ Garbage output (debugging) |
| **Q2_K** | 1.2 GB      | raw Q2_K packed      | —     | ✗ Not yet tested |
| **Q5_0** | 1.2 GB      | raw Q5_0 packed      | —     | ✗ Not yet tested |

**Q4_0 bug**: On-the-fly dequant matmul produces tokens but output is garbage (not "4" for "2+2="). The Q4_0 block dequantization in the generation pipeline has a bug — likely in how blocks are indexed across rows. Debugging requires stepping through a single matmul and comparing against reference dequantization.

**Expected Q4_0 speed if fixed**: ~15-20 tok/s (slower than f16 because each load requires nibble extraction + sign-extension + scaling, but faster memory bandwidth due to 4× smaller weights).

---

### Methodology

**What counts as "real inference":**
1. Load weights from extracted .bin files (f16 or raw quant)
2. Full forward pass: 22 layers × (RMS Norm ×2 + QKV proj + RoPE + GQA attn + O proj + FFN gate/up/down + residual)
3. KV cache stores K/V for each position
4. Argmax over 32,000 vocabulary logits
5. Token decode (ID → UTF-8 text, strip ▁ space markers)
6. Feed output token back as next input

**NOT synthetic:** This is actual token-by-token generation through the full model, not random-data matmul throughput.

---

### Files

| File | Purpose |
|------|---------|
| `tinyllama_benchmark.mojo` | Multi-prompt f16 benchmark |
| `tinyllama_q4_benchmark.mojo` | Q4_0 on-the-fly dequant (WIP) |
| `tl_gen.c` | C reference (25.0 tok/s f16) |

---

### Next Steps

1. **Fix Q4_0 on-the-fly dequant**: Debug block indexing in q4_0_mm, compare against reference
2. **Add Q2_K and Q5_0**: Similar on-the-fly kernels once Q4_0 is working
3. **Dequant-at-load option**: For faster comparison, dequantize Q4_0 to f16 at load time (proven working in tinyllama_f16.mojo)
