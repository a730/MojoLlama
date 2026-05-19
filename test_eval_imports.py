"""Quick smoke test for eval module imports."""
import sys
sys.path.insert(0, "src")

from mojollama.eval.orchestrator import list_benchmarks, get_benchmark, BENCHMARKS
print("=== Benchmarks ===")
for b in list_benchmarks():
    print(f"  {b['id']:15s} — {b['description']}")
print()
print(f"Total benchmark classes: {len(BENCHMARKS)}")

from mojollama.eval.mmlu import MMLUBenchmark
m = MMLUBenchmark()
print(f"MMLU name: {m.name}")

from mojollama.eval.gsm8k import GSM8KBenchmark
g = GSM8KBenchmark()
print(f"GSM8K name: {g.name}")

from mojollama.eval.ceval import CEvalBenchmark
c = CEvalBenchmark()
print(f"CEval name: {c.name}")

from mojollama.eval.hellaswag import HellaSwagBenchmark
h = HellaSwagBenchmark()
print(f"HellaSwag name: {h.name}")

from mojollama.eval.arc import ARCBenchmark
a = ARCBenchmark()
print(f"ARC name: {a.name}")

from mojollama.eval.bbh import BBHBenchmark
b2 = BBHBenchmark()
print(f"BBH name: {b2.name}, tasks: {len(b2.tasks)}")

from mojollama.eval.humaneval import HumanEvalBenchmark
he = HumanEvalBenchmark()
print(f"HumanEval name: {he.name}")

from mojollama.eval.dataset import list_available_datasets, DATASET_SOURCES
cached = list_available_datasets()
print(f"\nCached datasets: {list(cached.keys()) or 'none'}")
print(f"MMLU subjects: {len(DATASET_SOURCES['mmlu']['subjects'])}")
print(f"CEval subjects: {len(DATASET_SOURCES['ceval']['subjects'])}")
print(f"BBH tasks: {len(DATASET_SOURCES['bbh']['url_templates'])}")
print()
print("All imports OK!")
