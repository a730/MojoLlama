"""ARC — AI2 Reasoning Challenge benchmark.

Multiple choice science questions (Challenge and Easy sets).
Uses ai2_arc dataset.
"""

import json
import re
import time
from typing import Optional

from .base import BaseBenchmark, LlamaCppEvaluator, extract_letter_answer
from .dataset import get_dataset_path


FEW_SHOT_EXAMPLES = [
    {
        "question": "Which of the following is an example of a chemical change?",
        "choices": ["A. Ice melting", "B. Paper burning", "C. Water boiling", "D. Sugar dissolving"],
        "answer": "B"
    },
    {
        "question": "What force keeps planets in orbit around the Sun?",
        "choices": ["A. Magnetism", "B. Friction", "C. Gravity", "D. Electricity"],
        "answer": "C"
    },
    {
        "question": "Which organ pumps blood through the human body?",
        "choices": ["A. Lungs", "B. Liver", "C. Heart", "D. Brain"],
        "answer": "C"
    },
    {
        "question": "What is the largest planet in our solar system?",
        "choices": ["A. Mars", "B. Venus", "C. Saturn", "D. Jupiter"],
        "answer": "D"
    },
    {
        "question": "Which of these is a renewable energy source?",
        "choices": ["A. Coal", "B. Natural gas", "C. Solar power", "D. Oil"],
        "answer": "C"
    }
]


def _build_prompt(question, choices):
    prompt_lines = [
        "The following are science questions (with answers).",
        ""
    ]
    for ex in FEW_SHOT_EXAMPLES:
        prompt_lines.append(ex["question"])
        for ch in ex["choices"]:
            prompt_lines.append(ch)
        prompt_lines.append(f"Answer: {ex['answer']}")
        prompt_lines.append("")

    prompt_lines.append(question)
    for ch in choices:
        prompt_lines.append(ch)
    prompt_lines.append("Answer:")
    return "\n".join(prompt_lines)


def _load_arc_jsonl(path: str) -> list:
    """Load ARC dataset from JSONL format."""
    records = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception:
        pass
    return records


class ARCBenchmark(BaseBenchmark):
    """ARC benchmark — AI2 Reasoning Challenge (Challenge + Easy)."""

    @property
    def name(self) -> str:
        return "ARC"

    def run(self, backend_url: str, model_name: str,
            max_samples: int = 0, variant: str = "challenge",
            **kwargs) -> dict:
        evaluator = LlamaCppEvaluator(backend_url, temperature=0.0, max_tokens=32)
        t0 = time.time()

        total_correct = 0
        total_all = 0
        per_category = {}

        variants_to_run = ["challenge", "easy"] if variant == "all" else [variant]

        for v in variants_to_run:
            # Try multiple possible file names
            possible = [
                get_dataset_path("arc", subject=f"ARC-{v.capitalize()}-Test"),
                None
            ]
            import os
            base = os.path.expanduser("~/.mojollama/eval_datasets")
            for fname in [
                f"ARC-{v.capitalize()}-Test.jsonl",
                f"arc_{v}_test.jsonl",
                f"ARC-{v.capitalize()}-Test.json"
            ]:
                p = os.path.join(base, fname)
                if os.path.exists(p):
                    possible.append(p)

            records = []
            for p in possible:
                if p and os.path.exists(p):
                    records = _load_arc_jsonl(p)
                    if records:
                        break

            if not records:
                print(f"  ⚠ ARC-{v.capitalize()} data not found, download first: mojollama-studio evaluate download")
                per_category[f"arc_{v}"] = {"correct": 0, "total": 0, "accuracy": 0}
                continue

            if max_samples > 0:
                records = records[:max_samples]

            correct = 0
            total = 0
            labels = ["A", "B", "C", "D"]

            for record in records:
                question = record.get("question", "")
                choices_raw = record.get("choices", {})
                answer_key = record.get("answerKey", "")

                # Handle different ARC formats
                if isinstance(choices_raw, dict):
                    choice_labels = choices_raw.get("label", [])
                    choice_texts = choices_raw.get("text", [])
                    choices = [f"{l}. {t}" for l, t in zip(choice_labels, choice_texts)]
                elif isinstance(choices_raw, list):
                    choices = []
                    for i, c in enumerate(choices_raw):
                        if isinstance(c, dict):
                            label = c.get("label", labels[i] if i < len(labels) else chr(65+i))
                            text = c.get("text", str(c))
                            choices.append(f"{label}. {text}")
                        else:
                            choices.append(f"{labels[i]}. {c}")
                else:
                    continue

                prompt = _build_prompt(question, choices)

                try:
                    generated = evaluator.complete(prompt, max_tokens=16)
                    predicted = extract_letter_answer(generated)
                    if predicted is None:
                        m = re.search(r'\b([A-D])\b', generated)
                        if m:
                            predicted = m.group(1)
                    if predicted and predicted == answer_key:
                        correct += 1
                except Exception:
                    pass

                total += 1

            acc = correct / total if total > 0 else 0
            per_category[f"arc_{v}"] = {"correct": correct, "total": total, "accuracy": acc}
            total_correct += correct
            total_all += total
            print(f"  ARC-{v.capitalize()}: {acc*100:.2f}% ({correct}/{total})")

        elapsed = time.time() - t0
        stats = evaluator.get_stats()
        overall_acc = total_correct / total_all if total_all > 0 else 0

        return {
            "name": "ARC",
            "accuracy": overall_acc,
            "correct": total_correct,
            "total": total_all,
            "per_category": per_category,
            "metrics": {"elapsed": elapsed, "api_calls": stats["api_calls"], "api_errors": stats["api_errors"]},
            "samples": total_all
        }
