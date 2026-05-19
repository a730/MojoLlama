"""Test HF URL patterns."""
import urllib.request, json, sys

URLS = [
    # Try different URL patterns for MMLU
    "https://huggingface.co/datasets/lukaemon/mmlu/resolve/main/data/abstract_algebra_test.json",
    "https://huggingface.co/datasets/lukaemon/mmlu/raw/main/data/abstract_algebra_test.json",
    "https://huggingface.co/datasets/lukaemon/mmlu/resolve/main/abstract_algebra_test.json",
    "https://huggingface.co/datasets/lukaemon/mmlu/raw/main/abstract_algebra_test.json",
    "https://huggingface.co/datasets/lukaemon/mmlu/resolve/main/data/test/abstract_algebra_test.json",
    "https://huggingface.co/api/datasets/lukaemon/mmlu",
]

for url in URLS:
    try:
        req = urllib.request.Request(url, method="HEAD")
        resp = urllib.request.urlopen(req, timeout=10)
        print(f"✅ {url[:80]:80s} {resp.status}")
    except Exception as e:
        print(f"❌ {url[:80]:80s} {e}")
