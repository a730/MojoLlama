"""GSM8K — Grade School Math benchmark.

8-shot chain-of-thought evaluation. Exact match accuracy on final answer.
Uses openai/gsm8k dataset.
"""

import json
import re
import time
from typing import Optional

from .base import BaseBenchmark, LlamaCppEvaluator, extract_number_answer, normalize_answer
from .dataset import get_dataset_stream


# 8-shot CoT examples for GSM8K
COT_EXAMPLES = [
    {
        "question": "Beth has 4 bags of marbles. Each bag has 24 marbles. She gives away 3 marbles. How many marbles does she have left?",
        "answer": "Beth has 4 bags * 24 marbles = 96 marbles total. She gives away 3, so she has 96 - 3 = 93 marbles. #### 93"
    },
    {
        "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins with four every day. She sells the rest at the farmers' market daily for $2 per fresh egg. How much in dollars does she make every day at the farmers' market?",
        "answer": "She uses 3 + 4 = 7 eggs each day. She has 16 - 7 = 9 eggs left. She makes 9 * $2 = $18 each day. #### 18"
    },
    {
        "question": "A robe takes 2 1/4 yards of material and a jacket takes 4 3/4 yards of material. How much more material does a jacket require than a robe?",
        "answer": "4 3/4 - 2 1/4 = 2 1/2. The jacket requires 2 1/2 yards more. #### 2.5"
    },
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "answer": "There are 15 trees initially and 21 after planting. So they planted 21 - 15 = 6 trees. #### 6"
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "answer": "There are originally 3 cars. 2 more arrive, so 3 + 2 = 5 cars. #### 5"
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "answer": "Originally, Leah had 32 and her sister had 42, so 32 + 42 = 74. They ate 35, so 74 - 35 = 39 pieces left. #### 39"
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "answer": "Shawn started with 5 toys. He got 2 from mom and 2 from dad, so 5 + 2 + 2 = 9 toys. #### 9"
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from Monday to Thursday. How many computers are now in the server room?",
        "answer": "There were originally 9 computers. From Monday to Thursday is 4 days. Each day 5 computers are installed, so 4 * 5 = 20 computers. Total: 9 + 20 = 29 computers. #### 29"
    }
]


def _build_gsm8k_prompt(question, use_cot=True):
    """Build a GSM8K prompt with 8-shot CoT examples."""
    prompt_lines = []

    for ex in COT_EXAMPLES:
        prompt_lines.append(f"Question: {ex['question']}")
        prompt_lines.append(f"Answer: {ex['answer']}")
        prompt_lines.append("")

    prompt_lines.append(f"Question: {question}")
    prompt_lines.append("Answer:")

    return "\n".join(prompt_lines)


def _extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract the final numeric answer from GSM8K output."""
    # Look for #### N pattern
    m = re.search(r'####\s*(-?\d+(?:[.,]\d+)?)', text)
    if m:
        return m.group(1).replace(",", "")

    # Try to get the last number mentioned
    numbers = re.findall(r'-?\d+(?:[.,]\d+)?', text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


class GSM8KBenchmark(BaseBenchmark):
    """GSM8K benchmark — 8-shot chain-of-thought math word problems."""

    @property
    def name(self) -> str:
        return "GSM8K"

    def run(self, backend_url: str, model_name: str,
            max_samples: int = 0, use_cot: bool = True,
            **kwargs) -> dict:
        evaluator = LlamaCppEvaluator(backend_url, temperature=0.0, max_tokens=512)
        t0 = time.time()

        records = get_dataset_stream("gsm8k")
        if not records:
            return {
                "name": "GSM8K",
                "accuracy": 0, "correct": 0, "total": 0,
                "per_category": {},
                "metrics": {"elapsed": time.time() - t0, "api_calls": 0, "api_errors": 0},
                "samples": 0,
                "error": "No dataset found. Run `mojollama-studio evaluate download` first."
            }

        if max_samples > 0:
            records = records[:max_samples]

        correct = 0
        total = 0
        errors = 0

        for i, record in enumerate(records):
            question = record.get("question", "")
            answer = record.get("answer", "")

            # Extract ground truth answer (after ####)
            gt = _extract_gsm8k_answer(answer)
            if gt is None:
                continue

            prompt = _build_gsm8k_prompt(question, use_cot)

            try:
                generated = evaluator.complete(prompt, max_tokens=256,
                                               stop=["Question:"])
                predicted = _extract_gsm8k_answer(generated)
                if predicted and normalize_answer(predicted) == normalize_answer(gt):
                    correct += 1
                elif predicted:
                    # Try numeric comparison
                    try:
                        if abs(float(predicted) - float(gt)) < 0.01:
                            correct += 1
                    except (ValueError, TypeError):
                        pass
            except Exception as e:
                errors += 1

            total += 1

            if (i + 1) % 20 == 0 or (i + 1) == len(records):
                pct = (correct / total * 100) if total > 0 else 0
                print(f"  GSM8K: {pct:.1f}% ({correct}/{total}) — {i+1}/{len(records)}")

        elapsed = time.time() - t0
        stats = evaluator.get_stats()

        return {
            "name": "GSM8K",
            "accuracy": correct / total if total > 0 else 0,
            "correct": correct,
            "total": total,
            "per_category": {"math": {"correct": correct, "total": total, "accuracy": correct / total if total > 0 else 0}},
            "metrics": {
                "elapsed": elapsed,
                "api_calls": stats["api_calls"],
                "api_errors": stats["api_errors"],
                "total_tokens": stats["total_tokens"],
            },
            "samples": total
        }
