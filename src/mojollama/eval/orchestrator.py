"""Evaluation orchestrator — runs benchmarks and aggregates results.

This module provides the main entry point for running evaluations against
a llama.cpp backend, either from the CLI or the REST API.
"""

import json
import os
import sys
import time
import urllib.request
from typing import Optional

from .base import BaseBenchmark, LlamaCppEvaluator, check_backend_alive, format_results
from .dataset import get_dataset_path, get_dataset_stream, get_subjects, list_available_datasets, DATASET_SOURCES
from .mmlu import MMLUBenchmark
from .gsm8k import GSM8KBenchmark
from .ceval import CEvalBenchmark
from .hellaswag import HellaSwagBenchmark
from .arc import ARCBenchmark
from .bbh import BBHBenchmark
from .humaneval import HumanEvalBenchmark


BENCHMARKS = {
    "mmlu": MMLUBenchmark,
    "gsm8k": GSM8KBenchmark,
    "ceval": CEvalBenchmark,
    "hellaswag": HellaSwagBenchmark,
    "arc": ARCBenchmark,
    "bbh": BBHBenchmark,
    "humaneval": HumanEvalBenchmark,
}


def get_benchmark(name: str) -> Optional[BaseBenchmark]:
    """Get benchmark instance by name."""
    cls = BENCHMARKS.get(name.lower())
    if cls:
        return cls()
    return None


def list_benchmarks() -> list:
    """List available benchmarks with descriptions."""
    return [
        {"id": "mmlu", "name": "MMLU", "description": "57 subjects, 5-shot multiple choice"},
        {"id": "gsm8k", "name": "GSM8K", "description": "Grade school math, 8-shot CoT"},
        {"id": "ceval", "name": "CEval", "description": "Chinese evaluation, 52 subjects"},
        {"id": "hellaswag", "name": "HellaSwag", "description": "Commonsense reasoning"},
        {"id": "arc", "name": "ARC", "description": "AI2 Reasoning Challenge (Challenge + Easy)"},
        {"id": "bbh", "name": "BBH", "description": "Big-Bench Hard, 27 reasoning tasks"},
        {"id": "humaneval", "name": "HumanEval", "description": "Code generation (pass@1)"},
    ]


def run_benchmark(name: str, backend_url: str, model_name: str,
                  max_samples: int = 0, **kwargs) -> dict:
    """Run a single benchmark.

    Args:
        name: Benchmark name (mmlu, gsm8k, etc.)
        backend_url: llama.cpp server URL
        model_name: Model name/path for logging
        max_samples: Max samples per subject (0 = all)
        **kwargs: Passed to benchmark.run()

    Returns:
        Result dict with accuracy, per_category, etc.
    """
    bench = get_benchmark(name)
    if not bench:
        return {"name": name, "accuracy": 0, "correct": 0, "total": 0,
                "error": f"Unknown benchmark: {name}"}

    print(f"\n{'─'*60}")
    print(f"  Running {bench.name} benchmark...")
    print(f"  Backend: {backend_url}")
    print(f"  Model:   {model_name}")
    if max_samples > 0:
        print(f"  Samples: {max_samples} per category")
    print(f"{'─'*60}")

    return bench.run(backend_url, model_name, max_samples=max_samples, **kwargs)


def run_benchmarks(benchmarks: list, backend_url: str, model_name: str,
                   max_samples: int = 0) -> list:
    """Run multiple benchmarks and return aggregated results."""
    results = []
    for name in benchmarks:
        try:
            result = run_benchmark(name, backend_url, model_name, max_samples)
            results.append(result)
        except Exception as e:
            print(f"\n  ❌ {name} benchmark failed: {e}")
            results.append({
                "name": name, "accuracy": 0, "correct": 0, "total": 0,
                "error": str(e)
            })
    return results


def download_all_datasets(progress_callback=None) -> dict:
    """Download all benchmark datasets.

    Returns:
        {name: True/False} for each dataset
    """
    results = {}

    # MMLU — one file per subject
    subjects = get_subjects("mmlu")
    print(f"\n  Downloading MMLU ({len(subjects)} subjects)...")
    mmlu_ok = 0
    for i, subj in enumerate(subjects):
        path = get_dataset_path("mmlu", subj)
        if path:
            mmlu_ok += 1
        if progress_callback:
            progress_callback("mmlu", i + 1, len(subjects))
    results["mmlu"] = mmlu_ok > 0
    print(f"  → {mmlu_ok}/{len(subjects)} subjects cached")

    # GSM8K
    print("  Downloading GSM8K...")
    path = get_dataset_path("gsm8k")
    results["gsm8k"] = path is not None

    # CEval
    ceval_subjects = get_subjects("ceval")
    print(f"  Downloading CEval ({len(ceval_subjects)} subjects)...")
    ceval_ok = 0
    for subj in ceval_subjects:
        path = get_dataset_path("ceval", subj)
        if path:
            ceval_ok += 1
    results["ceval"] = ceval_ok > 0
    print(f"  → {ceval_ok}/{len(ceval_subjects)} subjects cached")

    # HellaSwag
    print("  Downloading HellaSwag...")
    path = get_dataset_path("hellaswag")
    results["hellaswag"] = path is not None

    # ARC — try specific file names
    print("  Downloading ARC (Challenge + Easy)...")
    arc_ok = False
    for v in ["challenge", "easy"]:
        for fname in [
            f"ARC-{v.capitalize()}-Test.jsonl",
            f"arc_{v}_test.jsonl"
        ]:
            import os
            dest = os.path.join(os.path.expanduser("~/.mojollama/eval_datasets"), fname)
            url = DATASET_SOURCES["arc"].get(f"{v}_url", "")
            if url:
                from .dataset import _download_file
                if _download_file(url, dest):
                    arc_ok = True
    results["arc"] = arc_ok

    # BBH
    print("  Downloading BBH (27 tasks)...")
    bbh_tasks = list(DATASET_SOURCES.get("bbh", {}).get("url_templates", {}).keys())
    bbh_ok = 0
    for task in bbh_tasks:
        path = get_dataset_path("bbh", task)
        if path:
            bbh_ok += 1
    results["bbh"] = bbh_ok > 0
    print(f"  → {bbh_ok}/{len(bbh_tasks)} tasks cached")

    # HumanEval
    print("  Downloading HumanEval...")
    path = get_dataset_path("humaneval")
    results["humaneval"] = path is not None

    print(f"\n  {'─'*40}")
    for name, ok in results.items():
        print(f"  {name:20s} {'✅' if ok else '❌'}")
    print(f"  {'─'*40}")

    return results


def get_dataset_info() -> list:
    """Get list of cached dataset info."""
    available = list_available_datasets()
    info = []
    for name, data in available.items():
        info.append({
            "name": name,
            "cached": True,
            "size_bytes": data.get("size", 0),
            "subjects": data.get("subjects", 0),
        })
    return info


def serve_stream_results(results: list, wfile) -> None:
    """Stream benchmark results line by line for the web UI.

    Each line is a JSON dict with a 'type' field:
      - progress: {type:'progress', message:str}
      - result: {type:'result', ...benchmark_result}
      - done: {type:'done', summary:str}
    """
    for r in results:
        name = r.get("name", "?")
        acc = r.get("accuracy", 0) * 100
        correct = r.get("correct", 0)
        total = r.get("total", 0)

        wfile.write(json.dumps({
            "type": "progress",
            "name": name,
            "message": f"{name}: {acc:.2f}% ({correct}/{total})"
        }).encode() + b"\n")
        wfile.flush()

        wfile.write(json.dumps({
            "type": "result",
            **r
        }).encode() + b"\n")
        wfile.flush()

    total_correct = sum(r.get("correct", 0) for r in results)
    total_all = sum(r.get("total", 0) for r in results)
    overall = total_correct / total_all * 100 if total_all > 0 else 0

    wfile.write(json.dumps({
        "type": "done",
        "overall_accuracy": overall,
        "total_correct": total_correct,
        "total_samples": total_all,
        "benchmarks_run": len(results),
    }).encode() + b"\n")
    wfile.flush()
