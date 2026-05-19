"""Base evaluation harness for MojoLlama benchmarks.

Defines the abstract interface for all benchmarks and the core evaluation
loop that interacts with llama.cpp server or any backend.
"""

import json
import os
import re
import time
import sys
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from typing import Optional


class BaseBenchmark(ABC):
    """Abstract base for all benchmarks.

    Subclasses must implement:
    - name: str — benchmark name
    - run(self, backend_url, model_name, **kwargs) -> dict — execute evaluation

    The run() method returns:
    {
        'name': str,
        'accuracy': float,  # 0.0 to 1.0
        'correct': int,
        'total': int,
        'per_category': {category: {correct: int, total: int, accuracy: float}},
        'metrics': {...},  # additional metrics
        'samples': int
    }
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def run(self, backend_url: str, model_name: str,
            max_samples: int = 0, **kwargs) -> dict:
        ...


class LlamaCppEvaluator:
    """Core evaluator that calls llama.cpp server API for completions."""

    def __init__(self, backend_url: str = "http://127.0.0.1:8081",
                 temperature: float = 0.0, max_tokens: int = 256,
                 timeout: int = 60):
        self.backend_url = backend_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._stats = {"api_calls": 0, "total_tokens": 0, "api_errors": 0}

    def complete(self, prompt: str, max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 stop: Optional[list] = None) -> str:
        """Send a completion request and return the generated text."""
        data = {
            "prompt": prompt,
            "n_predict": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "stream": False,
        }
        if stop:
            data["stop"] = stop

        req = urllib.request.Request(
            f"{self.backend_url}/completion",
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        self._stats["api_calls"] += 1
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                text = result.get("content", "")
                tokens = result.get("tokens", 0) or result.get("tokens_evaluated", 0)
                self._stats["total_tokens"] += tokens
                return text
        except Exception as e:
            self._stats["api_errors"] += 1
            raise RuntimeError(f"API call failed: {e}")

    def chat_complete(self, messages: list, max_tokens: Optional[int] = None,
                      temperature: Optional[float] = None) -> str:
        """Send a chat completion request and return the response text."""
        data = {
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "stream": False,
        }

        req = urllib.request.Request(
            f"{self.backend_url}/v1/chat/completions",
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        self._stats["api_calls"] += 1
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                usage = result.get("usage", {})
                self._stats["total_tokens"] += usage.get("total_tokens", 0)
                return text
        except Exception as e:
            self._stats["api_errors"] += 1
            raise RuntimeError(f"Chat API call failed: {e}")

    def get_stats(self) -> dict:
        """Return evaluation statistics."""
        return dict(self._stats)


def normalize_answer(answer: str) -> str:
    """Normalize answer for exact match comparison."""
    answer = answer.strip().lower()
    # Remove punctuation except period (for single letters)
    answer = answer.strip(".,;:!?\"'()[]{}")
    answer = answer.strip()
    return answer


def extract_letter_answer(text: str) -> Optional[str]:
    """Extract letter answer (A, B, C, D) from generated text."""
    text = text.strip()
    # Look for patterns like "Answer: A", "The answer is B", "A."
    patterns = [
        r'(?:^|\n)\s*(?:answer|the answer is|so the answer is|correct answer is|option|choice|answer:\s*)\s*[:\s]*([A-D])\b',
        r'(?:^|\n)\s*([A-D])\s*[.:)]?\s*(?:$|\n)',
        r'\(([A-D])\)',
        r'\b([A-D])\s*(?:is correct|is the answer|is right)\b',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


def extract_number_answer(text: str) -> Optional[str]:
    """Extract numeric answer from generated text (for GSM8K)."""
    text = text.strip()
    # Look for patterns like "#### 42", "Answer: 42", final number
    patterns = [
        r'####\s*(-?\d+(?:[,\.]\d+)?)',
        r'(?:answer|the answer is|result|so the answer is)\s*[:\s]*(-?\d+(?:[,\.]\d+)?)',
        r'(?:^|\n)\s*(-?\d+(?:[,\.]\d+)?)\s*(?:$|\n)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).replace(",", "")
    return None


def check_backend_alive(url: str) -> bool:
    """Check if a llama.cpp backend is running."""
    try:
        req = urllib.request.Request(f"{url}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def format_results(results: list) -> str:
    """Format benchmark results for display."""
    lines = []
    total_correct = sum(r["correct"] for r in results)
    total_all = sum(r["total"] for r in results)
    overall_acc = total_correct / total_all if total_all > 0 else 0

    lines.append(f"{'═'*60}")
    lines.append(f"  Overall Accuracy: {overall_acc*100:.2f}% ({total_correct}/{total_all})")
    lines.append(f"{'═'*60}")

    for r in results:
        acc = r["correct"] / r["total"] if r["total"] > 0 else 0
        lines.append(f"  {r['name']:30s} {acc*100:6.2f}%  ({r['correct']:4d}/{r['total']:<4d})")

        per_cat = r.get("per_category", {})
        if per_cat:
            for cat, stats in sorted(per_cat.items()):
                cat_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
                lines.append(f"    ├─ {cat:40s} {cat_acc*100:6.2f}% ({stats['correct']}/{stats['total']})")

    lines.append(f"{'═'*60}")
    metrics = {}
    for r in results:
        if "metrics" in r:
            for k, v in r["metrics"].items():
                metrics[k] = metrics.get(k, 0) + (v if isinstance(v, (int, float)) else 0)
    if metrics:
        lines.append(f"  Runtime: {metrics.get('elapsed', 0):.1f}s, "
                      f"API calls: {metrics.get('api_calls', 0)}, "
                      f"Errors: {metrics.get('api_errors', 0)}")

    return "\n".join(lines)
