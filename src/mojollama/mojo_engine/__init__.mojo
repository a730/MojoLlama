"""MojoLlama Mojo Engine — top-level module.
Systematic port of turbo_engine_v7_moe.py to Mojo.

Porting order:
  1. mojo_engine.mojo     — Main engine (Python interop bridge)
  2. matmul_kernels.mojo  — MXFP4/Q8_0 matmul kernels (pure Mojo)
  3. forward/              — Per-architecture forward passes
  4. server.mojo           — HTTP inference server

Rules:
  - Python files are NOT modified — Mojo versions replace them
  - Unportable Python parts use @python_call or Python interop
  - C kernels stay as C (called via @extern) — they're already optimized
"""
