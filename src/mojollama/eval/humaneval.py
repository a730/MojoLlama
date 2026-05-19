"""HumanEval — Code generation benchmark.

OpenAI's HumanEval: write function from docstring, pass/fail by unit test.
Evaluates functional correctness (pass@k).
"""

import json
import re
import time
from typing import Optional

from .base import BaseBenchmark, LlamaCppEvaluator
from .dataset import get_dataset_stream


FEW_SHOT_EXAMPLES = [
    {
        "prompt": 'def add(a, b):\n    """Return the sum of a and b."""',
        "result": '    return a + b'
    },
    {
        "prompt": 'def is_even(n):\n    """Return True if n is even, False otherwise."""',
        "result": '    return n % 2 == 0'
    },
    {
        "prompt": 'def factorial(n):\n    """Return n! for n >= 0."""',
        "result": '    if n <= 1:\n        return 1\n    return n * factorial(n - 1)'
    }
]


def _build_humaneval_prompt(prompt_text: str) -> str:
    """Build HumanEval prompt with few-shot examples."""
    parts = [
        "Complete the following Python functions. Return only the function body with proper indentation.",
        ""
    ]
    for ex in FEW_SHOT_EXAMPLES:
        parts.append(ex["prompt"])
        parts.append(ex["result"])
        parts.append("")

    parts.append(prompt_text)
    return "\n".join(parts)


def _extract_code(text: str) -> str:
    """Extract code from model output, removing markdown fences."""
    # Remove markdown code blocks
    text = re.sub(r'```(?:python)?\n?', '', text)
    # Remove trailing explanations
    text = re.split(r'\n(?:# |"""|\'\'\'|Explanation)', text)[0]
    # Get indented body
    lines = text.strip().split('\n')
    body = []
    for line in lines:
        if line.startswith('    ') or line.startswith('\t') or line.strip() == '':
            body.append(line)
        elif not body and line.strip():
            # First line might be the signature, skip it
            continue
        elif body and not line.startswith('    ') and line.strip():
            # We've moved past the function body
            break
    return '\n'.join(body).strip()


class HumanEvalBenchmark(BaseBenchmark):
    """HumanEval — code generation, functional correctness."""

    @property
    def name(self) -> str:
        return "HumanEval"

    def run(self, backend_url: str, model_name: str,
            max_samples: int = 0, **kwargs) -> dict:
        evaluator = LlamaCppEvaluator(backend_url, temperature=0.2, max_tokens=512)
        t0 = time.time()

        records = get_dataset_stream("humaneval")
        if not records:
            return {"name": "HumanEval", "accuracy": 0, "correct": 0, "total": 0,
                    "per_category": {}, "metrics": {"elapsed": time.time() - t0},
                    "samples": 0, "error": "No dataset. Run download first."}

        if max_samples > 0:
            records = records[:max_samples]

        correct = 0
        total = 0

        for record in records:
            prompt_text = record.get("prompt", "")
            entry_point = record.get("entry_point", "")
            test_code = record.get("test", "")
            canonical = record.get("canonical_solution", "")

            prompt = _build_humaneval_prompt(prompt_text)

            try:
                generated = evaluator.complete(prompt, max_tokens=384, temperature=0.2)
                body = _extract_code(generated)

                if body:
                    # Build complete function and run test
                    full_code = f"{prompt_text}\n{body}\n\n{test_code}\ncheck({entry_point})"
                    try:
                        exec_globals = {}
                        exec(full_code, exec_globals)
                        correct += 1
                    except Exception:
                        pass
            except Exception:
                pass

            total += 1

        elapsed = time.time() - t0
        stats = evaluator.get_stats()
        acc = correct / total if total > 0 else 0
        print(f"  HumanEval: {acc*100:.2f}% pass@1 ({correct}/{total})")

        return {
            "name": "HumanEval",
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "per_category": {"python_code_gen": {"correct": correct, "total": total, "accuracy": acc}},
            "metrics": {"elapsed": elapsed, "api_calls": stats["api_calls"], "api_errors": stats["api_errors"]},
            "samples": total
        }
