"""HellaSwag — Commonsense Reasoning benchmark.

Multiple choice sentence completion. Choose the most plausible ending.
Uses rowan/hellaswag dataset.
"""

import json
import re
import time
from typing import Optional

from .base import BaseBenchmark, LlamaCppEvaluator, extract_letter_answer, normalize_answer
from .dataset import get_dataset_stream


FEW_SHOT_EXAMPLES = [
    {
        "ctx": "A woman is sitting on a bench. She",
        "endings": ["A. stands up and walks away.", "B. flies into the sky.", "C. turns into a dragon.", "D. disappears."],
        "answer": "A"
    },
    {
        "ctx": "A man is cooking in the kitchen. He",
        "endings": ["A. starts singing opera loudly.", "B. chops vegetables for the soup.", "C. dissolves into thin air.", "D. forgets how to cook."],
        "answer": "B"
    },
    {
        "ctx": "A child is playing in the park. The child",
        "endings": ["A. grows wings and flies away.", "B. starts levitating.", "C. runs toward the swings.", "D. turns into a tree."],
        "answer": "C"
    },
    {
        "ctx": "A person is driving a car. They approach a red light. They",
        "endings": ["A. accelerate through the intersection.", "B. stop the car and wait.", "C. close their eyes.", "D. get out and run."],
        "answer": "B"
    },
    {
        "ctx": "A student is taking an exam. The student",
        "endings": ["A. reads the questions carefully.", "B. eats the exam paper.", "C. falls asleep on the floor.", "D. turns into a bird."],
        "answer": "A"
    }
]


def _build_hellaswag_prompt(ctx, endings):
    """Build a HellaSwag prompt with few-shot examples."""
    prompt_lines = [
        "Choose the most plausible ending for each situation.",
        ""
    ]
    for ex in FEW_SHOT_EXAMPLES:
        prompt_lines.append(ex["ctx"])
        for e in ex["endings"]:
            prompt_lines.append(e)
        prompt_lines.append(f"Answer: {ex['answer']}")
        prompt_lines.append("")

    prompt_lines.append(ctx)
    for e in endings:
        prompt_lines.append(e)
    prompt_lines.append("Answer:")

    return "\n".join(prompt_lines)


class HellaSwagBenchmark(BaseBenchmark):
    """HellaSwag — commonsense reasoning, multiple choice."""

    @property
    def name(self) -> str:
        return "HellaSwag"

    def run(self, backend_url: str, model_name: str,
            max_samples: int = 0, **kwargs) -> dict:
        evaluator = LlamaCppEvaluator(backend_url, temperature=0.0, max_tokens=32)
        t0 = time.time()

        records = get_dataset_stream("hellaswag")
        if not records:
            return {"name": "HellaSwag", "accuracy": 0, "correct": 0, "total": 0,
                    "per_category": {}, "metrics": {"elapsed": time.time() - t0},
                    "samples": 0, "error": "No dataset. Run download first."}

        if max_samples > 0:
            records = records[:max_samples]

        correct = 0
        total = 0
        labels = ["A", "B", "C", "D"]

        for record in records:
            ctx = record.get("ctx", "")
            endings = record.get("endings", [])
            label = record.get("label", "")

            # Map label to index
            label_idx = None
            if label in "0123":
                label_idx = int(label)
            elif label.isdigit() and 0 <= int(label) < len(endings):
                label_idx = int(label)
            elif label in labels:
                label_idx = labels.index(label)
            else:
                continue

            answer_letter = labels[label_idx] if 0 <= label_idx < len(labels) else None
            if answer_letter is None:
                continue

            # Format endings with letter labels
            formatted = [f"{labels[i]}. {endings[i]}" for i in range(len(endings))]
            prompt = _build_hellaswag_prompt(ctx, formatted)

            try:
                generated = evaluator.complete(prompt, max_tokens=16)
                predicted = extract_letter_answer(generated)
                if predicted is None:
                    m = re.search(r'\b([A-D])\b', generated)
                    if m:
                        predicted = m.group(1)
                if predicted == answer_letter:
                    correct += 1
            except Exception:
                pass

            total += 1

        elapsed = time.time() - t0
        stats = evaluator.get_stats()
        acc = correct / total if total > 0 else 0

        print(f"  HellaSwag: {acc*100:.2f}% ({correct}/{total})")

        return {
            "name": "HellaSwag",
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "per_category": {"commonsense_reasoning": {"correct": correct, "total": total, "accuracy": acc}},
            "metrics": {"elapsed": elapsed, "api_calls": stats["api_calls"], "api_errors": stats["api_errors"]},
            "samples": total
        }
