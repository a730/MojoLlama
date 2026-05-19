#!/usr/bin/env python3
"""MojoLlama bridge — llama.cpp backend integration.

Uses llama.cpp server as the high-performance compute backend while
preserving the MojoLlama op graph architecture.
"""

import json
import urllib.request
import urllib.error
import numpy as np
from typing import Optional


class LlamaCppBackend:
    """Inference backend using llama.cpp server."""
    
    def __init__(self, base_url: str = "http://localhost:9000"):
        self.base_url = base_url.rstrip("/")
        self._model_info = None
    
    def _request(self, endpoint: str, data: dict) -> dict:
        """Send request to llama.cpp server."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    
    def _get(self, endpoint: str) -> dict:
        """GET request."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read())
    
    @property
    def model_info(self) -> dict:
        if self._model_info is None:
            models = self._get("v1/models")
            self._model_info = models
        return self._model_info
    
    @property
    def is_healthy(self) -> bool:
        try:
            info = self._get("health") if self.base_url else self._get("v1/models")
            return True
        except Exception:
            return False
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: int = 42,
    ) -> dict:
        """Generate text using llama.cpp server."""
        result = self._request("completion", {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "cache_prompt": True,
        })
        return {
            "text": result.get("content", ""),
            "tokens_generated": result.get("tokens_predicted", 0),
            "tokens_evaluated": result.get("tokens_evaluated", 0),
            "timings": result.get("timings", {}),
            "model": result.get("model", ""),
        }
    
    def chat(
        self,
        messages: list,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> dict:
        """Chat completion via llama.cpp's OpenAI-compatible endpoint."""
        result = self._request("v1/chat/completions", {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        choice = result.get("choices", [{}])[0]
        usage = result.get("usage", {})
        return {
            "text": choice.get("message", {}).get("content", ""),
            "tokens_generated": usage.get("completion_tokens", 0),
            "tokens_prompt": usage.get("prompt_tokens", 0),
        }


# Quick benchmark
if __name__ == "__main__":
    import time
    
    backend = LlamaCppBackend()
    print(f"Server healthy: {backend.is_healthy}")
    
    if backend.is_healthy:
        # Benchmark generation
        prompt = "The meaning of life is"
        n_warmup = 2
        n_bench = 5
        
        print(f"\nWarmup ({n_warmup})...")
        for i in range(n_warmup):
            backend.generate(prompt, max_tokens=32)
        
        print(f"Benchmark ({n_bench})...")
        times = []
        for i in range(n_bench):
            t0 = time.time()
            result = backend.generate(prompt, max_tokens=128)
            elapsed = time.time() - t0
            tok = result.get("tokens_generated", 0)
            tps = tok / elapsed if elapsed > 0 else 0
            times.append(tps)
            print(f"  [{i+1}] {tok} tok in {elapsed:.2f}s = {tps:.1f} tok/s")
        
        avg_tps = sum(times) / len(times)
        print(f"\nAverage: {avg_tps:.1f} tok/s")
        print(f"Sample: {result['text'][:100]}...")
