"""BBH — Big-Bench Hard benchmark.

27 challenging reasoning tasks (subset of BIG-Bench).
Few-shot CoT evaluation.
"""

import json
import re
import time
from typing import Optional

from .base import BaseBenchmark, LlamaCppEvaluator, normalize_answer
from .dataset import get_dataset_stream, DATASET_SOURCES


# Example for each BBH task
TASK_EXAMPLES = {
    "boolean_expressions": {
        "ex": "Question: (not True or False) and (True or False)\nAnswer: Let's think step by step. (not True) is False. (False or False) is False. (True or False) is True. False and True is False. So the answer is False.\nTherefore, the answer is False.",
    },
    "causal_judgment": {
        "ex": "Question: ...\nAnswer: Let's think step by step. ... Therefore, the answer is No.",
    },
}


def _build_bbh_prompt(task_name: str, question: str, cot: bool = True) -> str:
    """Build BBH prompt with optional chain-of-thought."""
    examples = TASK_EXAMPLES.get(task_name, {})

    if cot and examples:
        prompt = f"{examples['ex']}\n\nQuestion: {question}\nAnswer: Let's think step by step."
    else:
        prompt = f"Question: {question}\nAnswer:"
    return prompt


def _extract_bbh_answer(text: str) -> str:
    """Extract final answer from BBH output."""
    text = text.strip()
    # Look for "the answer is X" patterns
    patterns = [
        r'(?:answer is|so the answer is|therefore,?\s*(?:the\s+)?answer\s+is)\s*[:.]?\s*["\']?(\w+)["\']?',
        r'(?:^|\n)\s*answer\s*:\s*["\']?(\w+)["\']?',
        r'(\w+)\s*$',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip().strip('.,;:!?')
    # Last word as fallback
    words = text.split()
    return words[-1].strip('.,;:!?') if words else ""


class BBHBenchmark(BaseBenchmark):
    """BBH — Big-Bench Hard, 27 challenging reasoning tasks."""

    @property
    def name(self) -> str:
        return "BBH"

    @property
    def tasks(self) -> list:
        return list(DATASET_SOURCES.get("bbh", {}).get("url_templates", {}).keys())

    def run(self, backend_url: str, model_name: str,
            max_samples: int = 0, tasks: Optional[list] = None,
            use_cot: bool = True,
            **kwargs) -> dict:
        evaluator = LlamaCppEvaluator(backend_url, temperature=0.0, max_tokens=256)
        t0 = time.time()

        all_tasks = tasks or self.tasks
        per_category = {}
        total_correct = 0
        total_all = 0

        for task_name in all_tasks:
            records = get_dataset_stream("bbh", task_name)
            if not records:
                print(f"  ⚠ No data for {task_name}, skipping")
                per_category[task_name] = {"correct": 0, "total": 0, "accuracy": 0}
                continue

            if max_samples > 0:
                records = records[:max_samples]

            task_correct = 0
            task_total = 0

            for record in records:
                question = record.get("input", record.get("question", ""))
                target = record.get("target", record.get("answer", ""))

                prompt = _build_bbh_prompt(task_name, question, use_cot)

                try:
                    max_tok = 128 if not use_cot else 256
                    generated = evaluator.complete(prompt, max_tokens=max_tok)
                    predicted = _extract_bbh_answer(generated)
                    if normalize_answer(predicted) == normalize_answer(target):
                        task_correct += 1
                except Exception:
                    pass

                task_total += 1

            acc = task_correct / task_total if task_total > 0 else 0
            per_category[task_name] = {"correct": task_correct, "total": task_total, "accuracy": acc}
            total_correct += task_correct
            total_all += task_total
            print(f"  {task_name:40s} {acc*100:6.2f}% ({task_correct}/{task_total})")

        elapsed = time.time() - t0
        stats = evaluator.get_stats()
        overall_acc = total_correct / total_all if total_all > 0 else 0

        return {
            "name": "BBH",
            "accuracy": overall_acc,
            "correct": total_correct,
            "total": total_all,
            "per_category": per_category,
            "metrics": {"elapsed": elapsed, "api_calls": stats["api_calls"], "api_errors": stats["api_errors"],
                        "tasks_evaluated": len([t for t in per_category if per_category[t]["total"] > 0])},
            "samples": total_all
        }
