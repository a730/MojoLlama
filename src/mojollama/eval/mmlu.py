"""MMLU (Massive Multitask Language Understanding) benchmark.

57 subjects, 5-shot, multiple-choice evaluation.
Uses the lukaemon/mmlu dataset format from Hugging Face.
"""

import json
import os
import re
import time
from typing import Optional

from .base import BaseBenchmark, LlamaCppEvaluator, extract_letter_answer, normalize_answer
from .dataset import get_dataset_stream, get_subjects, get_dataset_path, DATASET_SOURCES


# Few-shot examples for MMLU (5-shot)
FEW_SHOT_EXAMPLES = {
    "abstract_algebra": [
        {
            "question": "Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(5)) over Q.",
            "choices": ["A. 4", "B. 6", "C. 8", "D. 2"],
            "answer": "C"
        },
        {
            "question": "Let V be the set of all solutions to the differential equation y'' + 4y' + 3y = 0. What is the dimension of V as a real vector space?",
            "choices": ["A. 0", "B. 1", "C. 2", "D. 3"],
            "answer": "C"
        },
        {
            "question": "How many elements of order 5 are in S_7?",
            "choices": ["A. 504", "B. 126", "C. 21", "D. 252"],
            "answer": "D"
        },
        {
            "question": "What is the value of the infinite sum 1 + 1/4 + 1/9 + 1/16 + ...?",
            "choices": ["A. pi^2/4", "B. pi^2/6", "C. pi^2/8", "D. pi^2/2"],
            "answer": "B"
        },
        {
            "question": "How many ring homomorphisms are there from Z/5Z to Z/20Z?",
            "choices": ["A. 1", "B. 2", "C. 3", "D. 5"],
            "answer": "A"
        }
    ]
}

_DEFAULT_FEW_SHOT = [
    {"question": "What is the capital of France?", "choices": ["A. London", "B. Paris", "C. Berlin", "D. Madrid"], "answer": "B"},
    {"question": "Which planet is known as the Red Planet?", "choices": ["A. Venus", "B. Jupiter", "C. Mars", "D. Saturn"], "answer": "C"},
    {"question": "What is 2 + 2?", "choices": ["A. 3", "B. 4", "C. 5", "D. 6"], "answer": "B"},
]


def _build_mmlu_prompt(question, choices, subject=None, few_shot=None):
    """Build a prompt for MMLU with few-shot examples."""
    if few_shot is None:
        few_shot = FEW_SHOT_EXAMPLES.get(subject, _DEFAULT_FEW_SHOT)

    prompt_lines = [
        "The following are multiple choice questions (with answers).",
        ""
    ]
    for ex in few_shot:
        prompt_lines.append(ex["question"])
        for ch in ex["choices"]:
            prompt_lines.append(ch)
        prompt_lines.append(f"Answer: {ex['answer']}")
        prompt_lines.append("")

    # The actual question
    prompt_lines.append(question)
    for ch in choices:
        prompt_lines.append(ch)
    prompt_lines.append("Answer:")

    return "\n".join(prompt_lines)


class MMLUBenchmark(BaseBenchmark):
    """MMLU benchmark — 57 subjects, 5-shot multiple choice."""

    @property
    def name(self) -> str:
        return "MMLU"

    def run(self, backend_url: str, model_name: str,
            max_samples: int = 0, subjects: Optional[list] = None,
            **kwargs) -> dict:
        evaluator = LlamaCppEvaluator(backend_url, temperature=0.0, max_tokens=50)
        t0 = time.time()

        all_subjects = subjects or get_subjects("mmlu")
        per_category = {}
        total_correct = 0
        total_all = 0

        for subject in all_subjects:
            records = get_dataset_stream("mmlu", subject)
            if not records:
                print(f"  ⚠ No data for {subject}, skipping")
                continue

            if max_samples > 0:
                records = records[:max_samples]

            subject_correct = 0
            subject_total = 0

            for record in records:
                question = record.get("question", "")
                choices = record.get("choices", [])
                answer_idx = record.get("answer", -1)

                # Convert answer index to letter
                labels = ["A", "B", "C", "D"]
                answer_letter = labels[answer_idx] if 0 <= answer_idx < len(labels) else None
                if answer_letter is None:
                    continue

                # Format choices as A. text, B. text, etc.
                formatted_choices = [
                    f"{labels[i]}. {choices[i]}"
                    for i in range(len(choices))
                ]

                prompt = _build_mmlu_prompt(question, formatted_choices, subject)

                try:
                    generated = evaluator.complete(prompt, max_tokens=32)
                    predicted = extract_letter_answer(generated)
                    if predicted is None:
                        # Try first letter in output
                        m = re.search(r'\b([A-D])\b', generated)
                        if m:
                            predicted = m.group(1)
                    if predicted == answer_letter:
                        subject_correct += 1
                except Exception as e:
                    print(f"  ⚠ Error on {subject}: {e}")

                subject_total += 1

            per_category[subject] = {
                "correct": subject_correct,
                "total": subject_total,
                "accuracy": subject_correct / subject_total if subject_total > 0 else 0
            }
            total_correct += subject_correct
            total_all += subject_total

            pct = (subject_correct / subject_total * 100) if subject_total > 0 else 0
            print(f"  {subject:40s} {pct:6.2f}% ({subject_correct}/{subject_total})")

        elapsed = time.time() - t0
        stats = evaluator.get_stats()

        return {
            "name": "MMLU",
            "accuracy": total_correct / total_all if total_all > 0 else 0,
            "correct": total_correct,
            "total": total_all,
            "per_category": per_category,
            "metrics": {
                "elapsed": elapsed,
                "api_calls": stats["api_calls"],
                "api_errors": stats["api_errors"],
                "total_tokens": stats["total_tokens"],
                "subjects_evaluated": len([s for s in per_category if per_category[s]["total"] > 0])
            },
            "samples": total_all
        }
