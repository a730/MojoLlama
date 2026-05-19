"""CEval — Chinese Evaluation benchmark.

52 subjects, multiple choice evaluation in Chinese.
Uses ceval/ceval-exam dataset format.
"""

import json
import re
import time
from typing import Optional

from .base import BaseBenchmark, LlamaCppEvaluator, extract_letter_answer
from .dataset import get_dataset_stream, get_subjects


# Few-shot examples for CEval
CEVAL_FEW_SHOT = [
    {
        "question": "中国的首都是哪个城市？",
        "choices": ["A. 上海", "B. 北京", "C. 广州", "D. 深圳"],
        "answer": "B"
    },
    {
        "question": "太阳从哪个方向升起？",
        "choices": ["A. 西方", "B. 南方", "C. 北方", "D. 东方"],
        "answer": "D"
    },
    {
        "question": "一年有多少个月？",
        "choices": ["A. 10个月", "B. 11个月", "C. 12个月", "D. 13个月"],
        "answer": "C"
    },
    {
        "question": "以下哪个是中国的四大发明？",
        "choices": ["A. 造纸术", "B. 蒸汽机", "C. 电灯", "D. 手机"],
        "answer": "A"
    },
    {
        "question": "水的化学式是什么？",
        "choices": ["A. CO2", "B. H2O", "C. NaCl", "D. O2"],
        "answer": "B"
    }
]


def _build_ceval_prompt(question, choices, few_shot=None):
    """Build a CEval prompt in Chinese with few-shot examples."""
    if few_shot is None:
        few_shot = CEVAL_FEW_SHOT

    prompt_lines = [
        "以下是选择题（请选择正确的答案）。",
        ""
    ]
    for ex in few_shot:
        prompt_lines.append(ex["question"])
        for ch in ex["choices"]:
            prompt_lines.append(ch)
        prompt_lines.append(f"答案: {ex['answer']}")
        prompt_lines.append("")

    prompt_lines.append(question)
    for ch in choices:
        prompt_lines.append(ch)
    prompt_lines.append("答案:")

    return "\n".join(prompt_lines)


class CEvalBenchmark(BaseBenchmark):
    """CEval benchmark — 52 subjects, Chinese multiple choice."""

    @property
    def name(self) -> str:
        return "CEval"

    def run(self, backend_url: str, model_name: str,
            max_samples: int = 0, subjects: Optional[list] = None,
            **kwargs) -> dict:
        evaluator = LlamaCppEvaluator(backend_url, temperature=0.0, max_tokens=50)
        t0 = time.time()

        all_subjects = subjects or get_subjects("ceval")
        per_category = {}
        total_correct = 0
        total_all = 0

        for subject in all_subjects:
            records = get_dataset_stream("ceval", subject)
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

                labels = ["A", "B", "C", "D"]
                answer_letter = labels[answer_idx] if 0 <= answer_idx < len(labels) else None
                if answer_letter is None:
                    continue

                formatted_choices = [
                    f"{labels[i]}. {choices[i]}"
                    for i in range(len(choices))
                ]

                prompt = _build_ceval_prompt(question, formatted_choices)

                try:
                    generated = evaluator.complete(prompt, max_tokens=32)
                    predicted = extract_letter_answer(generated)
                    if predicted is None:
                        m = re.search(r'\b([A-D])\b', generated)
                        if m:
                            predicted = m.group(1)
                    if predicted == answer_letter:
                        subject_correct += 1
                except Exception as e:
                    pass

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
            "name": "CEval",
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
