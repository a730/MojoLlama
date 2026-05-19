"""Dataset downloader — Caches evaluation datasets locally.

Downloads from Hugging Face datasets or mirrors. Supports MMLU, GSM8K,
CEval, HellaSwag, ARC, BBH, HumanEval.
"""

import json
import os
import urllib.request
import urllib.error
import gzip
import shutil
import hashlib
import re
from pathlib import Path
from typing import Optional

CACHE_DIR = os.path.expanduser("~/.mojollama/eval_datasets")
os.makedirs(CACHE_DIR, exist_ok=True)

DATASET_SOURCES = {
    "mmlu": {
        "url": "https://huggingface.co/datasets/lukaemon/mmlu/resolve/main/data/{subject}_test.json",
        "fallback_url": "https://huggingface.co/datasets/lukaemon/mmlu/raw/main/data/{subject}_test.json",
        "subjects": [
            "abstract_algebra", "anatomy", "astronomy", "business_ethics",
            "clinical_knowledge", "college_biology", "college_chemistry",
            "college_computer_science", "college_mathematics", "college_medicine",
            "college_physics", "computer_security", "conceptual_physics",
            "econometrics", "electrical_engineering", "elementary_mathematics",
            "formal_logic", "global_facts", "high_school_biology",
            "high_school_chemistry", "high_school_computer_science",
            "high_school_european_history", "high_school_geography",
            "high_school_government_and_politics", "high_school_macroeconomics",
            "high_school_mathematics", "high_school_microeconomics",
            "high_school_physics", "high_school_psychology",
            "high_school_statistics", "high_school_us_history",
            "high_school_world_history", "human_aging", "human_sexuality",
            "international_law", "jurisprudence", "logical_fallacies",
            "machine_learning", "management", "marketing", "medical_genetics",
            "miscellaneous", "moral_disputes", "moral_scenarios",
            "nutrition", "philosophy", "prehistory", "professional_accounting",
            "professional_law", "professional_medicine", "professional_psychology",
            "public_relations", "security_studies", "sociology",
            "us_foreign_policy", "virology", "world_religions"
        ]
    },
    "gsm8k": {
        "url": "https://huggingface.co/datasets/openai/gsm8k/resolve/main/data/test.jsonl",
        "fallback_url": "https://huggingface.co/datasets/openai/gsm8k/raw/main/data/test.jsonl"
    },
    "ceval": {
        "url": "https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/data/{subject}_test.json",
        "subjects": [
            "accountant", "advanced_mathematics", "art_studies", "basic_medicine",
            "business_administration", "chinese_language_and_literature",
            "college_chemistry", "college_physics", "computer_network",
            "computer_organization_and_architecture", "criminal_law", "criminal_procedure_law",
            "discrete_mathematics", "education_science", "electrical_engineer",
            "environmental_impact_assessment_engineer", "fire_fighting",
            "high_school_biology", "high_school_chemistry", "high_school_chinese",
            "high_school_geography", "high_school_history", "high_school_mathematics",
            "high_school_physics", "high_school_politics", "ideological_and_moral_cultivation",
            "internet_information_service_engineer", "law", "lawyer_qualification",
            "legal_professional", "legal_system_and_legal_institutions",
            "logistics_engineering_and_management", "mao_zedong_thought",
            "marxism", "metrology_engineer", "middle_school_biology",
            "middle_school_chemistry", "middle_school_geography", "middle_school_history",
            "middle_school_mathematics", "middle_school_physics", "middle_school_politics",
            "modern_chinese_history", "operating_system", "pharmacy",
            "principles_of_computer_composition", "probability_and_statistics",
            "professional_geography_of_civil_engineering", "professional_knowledge_of_civil_engineering",
            "sports_training", "tax_law_agent", "teacher_qualification",
            "tour_guide", "urban_and_rural_planning", "veterinary_medicine",
            "water_resources", "writer"
        ]
    },
    "hellaswag": {
        "url": "https://huggingface.co/datasets/rowan/hellaswag/resolve/main/data/hellaswag_test.jsonl",
        "fallback_url": "https://huggingface.co/datasets/rowan/hellaswag/raw/main/data/hellaswag_test.jsonl"
    },
    "arc": {
        "challenge_url": "https://huggingface.co/datasets/ai2_arc/resolve/main/data/ARC-Challenge-Test.jsonl",
        "easy_url": "https://huggingface.co/datasets/ai2_arc/resolve/main/data/ARC-Easy-Test.jsonl",
        "challenge_fallback": "https://huggingface.co/datasets/ai2_arc/raw/main/data/ARC-Challenge-Test.jsonl",
        "easy_fallback": "https://huggingface.co/datasets/ai2_arc/raw/main/data/ARC-Easy-Test.jsonl"
    },
    "bbh": {
        "url_templates": {
            "boolean_expressions": "https://huggingface.co/datasets/lukaemon/bbh/resolve/main/data/boolean_expressions/test.jsonl",
            "causal_judgment": "https://huggingface.co/datasets/lukaemon/bbh/resolve/main/data/causal_judgment/test.jsonl",
            "date_understanding": "https://huggingface.co/datasets/lukaemon/bbh/resolve/main/data/date_understanding/test.jsonl",
            "disambiguation_qa": "https://huggingface.co/datasets/lukaemon/bbh/resolve/main/data/disambiguation_qa/test.jsonl",
            "dyck_languages": "https://huggingface.co/datasets/lukaemon/bbh/resolve/main/data/dyck_languages/test.jsonl",
            "formal_fallacies": "https://huggingface.co/datasets/lukaemon/bbh/resolve/main/data/formal_fallacies/test.jsonl",
            "geometric_shapes": "https://huggingface.co/datasets/lukaemon/bbh/resolve/main/data/geometric_shapes/test.jsonl",
            "hyperbaton": "https://huggingface.co/datasets/lukaemon/bbh/resolve/main/data/hyperbaton/test.jsonl",
            "logical_deduction_five_objects": "https://huggingface.co/datasets/lukaemon/bbh/resolve/main/data/logical_deduction_five_objects/test.jsonl",
            "logical_deduction_seven_objects": "https://huggingface.co/datasets/lukaemon/bbh/resolve/main/data/logical_deduction_seven_objects/test.jsonl",
            "logical_deduction_three_objects": "https://huggingface.co/datasets/lukaemon/bbh/resolve/main/data/logical_deduction_three_objects/test.jsonl",
            "movie_recommendation": "https://huggingface.co/datasets/lukaemon/bbh/resolve/main/data/movie_recommendation/test.jsonl",
            "multistep_arithmetic_two": "https://huggingface.co/datasets/lukaemon/bbh/resolve/main/data/multistep_arithmetic_two/test.jsonl",
            "navigate": "https://huggingface.co/datasets/lukaemon/bbh/resolve/main/data/navigate/test.jsonl",
            "object_counting": "https://huggingface.co/datasets/lukaemon/bbh/resolve/main/data/object_counting/test.jsonl",
            "penguins_in_a_table": "https://huggingface.co/datasets/lukaemon/bbh/resolve/main/data/penguins_in_a_table/test.jsonl",
            "reasoning_about_colored_objects": "https://huggingface.co/datasets/lukaemon/bbh/resolve/main/data/reasoning_about_colored_objects/test.jsonl",
            "ruin_names": "https://huggingface.co/datasets/lukaemon/bbh/resolve/main/data/ruin_names/test.jsonl",
            "salient_translation_error_detection": "https://huggingface.co/datasets/lukaemon/bbh/resolve/main/data/salient_translation_error_detection/test.jsonl",
            "snarks": "https://huggingface.co/datasets/lukaemon/bbh/resolve/main/data/snarks/test.jsonl",
            "sports_understanding": "https://huggingface.co/datasets/lukaemon/bbh/resolve/main/data/sports_understanding/test.jsonl",
            "temporal_sequences": "https://huggingface.co/datasets/lukaemon/bbh/resolve/main/data/temporal_sequences/test.jsonl",
            "tracking_shuffled_objects_five_objects": "https://huggingface.co/datasets/lukaemon/bbh/resolve/main/data/tracking_shuffled_objects_five_objects/test.jsonl",
            "tracking_shuffled_objects_seven_objects": "https://huggingface.co/datasets/lukaemon/bbh/resolve/main/data/tracking_shuffled_objects_seven_objects/test.jsonl",
            "tracking_shuffled_objects_three_objects": "https://huggingface.co/datasets/lukaemon/bbh/resolve/main/data/tracking_shuffled_objects_three_objects/test.jsonl",
            "web_of_lies": "https://huggingface.co/datasets/lukaemon/bbh/resolve/main/data/web_of_lies/test.jsonl",
        }
    },
    "humaneval": {
        "url": "https://huggingface.co/datasets/openai/openai_humaneval/resolve/main/data/test.jsonl",
        "fallback_url": "https://huggingface.co/datasets/openai/openai_humaneval/raw/main/data/test.jsonl"
    }
}

# Embedded mini test data for benchmarks that don't need full download
EMBEDDED_SAMPLES = {}


def _download_file(url: str, dest: str) -> bool:
    """Download a file with progress. Returns True on success."""
    try:
        print(f"  Downloading {url}...")
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def get_dataset_path(name: str, subject: Optional[str] = None, force_download: bool = False) -> Optional[str]:
    """Get cached path for a dataset. Downloads if needed.

    Args:
        name: Dataset name (mmlu, gsm8k, ceval, hellaswag, arc, bbh, humaneval)
        subject: Subject name for multi-subject datasets (mmlu, ceval)
        force_download: Re-download even if cached

    Returns:
        Path to cached file, or None if download failed
    """
    source = DATASET_SOURCES.get(name)
    if not source:
        return None

    if subject:
        cache_file = os.path.join(CACHE_DIR, name, f"{subject}.json")
        os.makedirs(os.path.join(CACHE_DIR, name), exist_ok=True)
    else:
        cache_file = os.path.join(CACHE_DIR, f"{name}.jsonl")
        os.makedirs(CACHE_DIR, exist_ok=True)

    if os.path.exists(cache_file) and not force_download:
        return cache_file

    # Build URL
    if subject:
        url = source.get("url", "").replace("{subject}", subject)
        fallback = source.get("fallback_url", "").replace("{subject}", subject)
    else:
        url = source.get("url", "")
        fallback = source.get("fallback_url", "")

    if not url:
        return None

    if _download_file(url, cache_file):
        return cache_file
    if fallback and _download_file(fallback, cache_file):
        return cache_file

    return None


def get_dataset_stream(name: str, subject: Optional[str] = None):
    """Get a stream of JSON records for a dataset.

    Usage: for record in get_dataset_stream('gsm8k'):
               question = record['question']

    Returns an iterable of parsed JSON dicts, or empty list on failure.
    """
    path = get_dataset_path(name, subject)
    if not path:
        return []

    records = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except json.JSONDecodeError:
        # Try as JSON array
        try:
            with open(path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    records = data
        except Exception:
            pass
    except Exception:
        pass

    return records


def get_subjects(name: str) -> list:
    """Get list of subjects for multi-subject datasets."""
    source = DATASET_SOURCES.get(name, {})
    return source.get("subjects", [])


def list_available_datasets() -> dict:
    """List which datasets are already cached."""
    available = {}
    for name in DATASET_SOURCES:
        path = os.path.join(CACHE_DIR, f"{name}.jsonl")
        if os.path.exists(path):
            available[name] = {
                "path": path,
                "size": os.path.getsize(path)
            }
        else:
            subj_dir = os.path.join(CACHE_DIR, name)
            if os.path.isdir(subj_dir):
                files = os.listdir(subj_dir)
                available[name] = {
                    "path": subj_dir,
                    "subjects": len(files),
                    "size": sum(os.path.getsize(os.path.join(subj_dir, f)) for f in files if os.path.isfile(os.path.join(subj_dir, f)))
                }
    return available
