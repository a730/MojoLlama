#!/usr/bin/env python3
"""MojoLlama Dataset Pipeline — formats, streaming, auto-labeling, stats.

Supports:
  - Alpaca (instruction/input/output)
  - ShareGPT (conversations array)
  - OpenAI (messages format)
  - Preference (chosen/rejected pairs for DPO/ORPO/KTO)
  - Streaming (generator-based, memory-mapped, HF datasets)
  - Auto-labeling with confidence scoring and batch processing
  - Format detection, conversion, and statistics
"""

import os
import json
import gzip
import math
import time
import queue
import struct
import hashlib
import threading
import collections
from pathlib import Path
from typing import (
    Any, Callable, Dict, Generator, Iterator, List, Optional,
    Tuple, Union,
)

import numpy as np

# ─── Format Constants ───────────────────────────────────────────────────

FORMAT_ALPACA = "alpaca"
FORMAT_SHAREGPT = "sharegpt"
FORMAT_OPENAI = "openai"
FORMAT_PREFERENCE = "preference"
FORMAT_JSONL = "jsonl"

ALL_FORMATS = [FORMAT_ALPACA, FORMAT_SHAREGPT, FORMAT_OPENAI, FORMAT_PREFERENCE, FORMAT_JSONL]
TRAINING_FORMATS = [FORMAT_ALPACA, FORMAT_SHAREGPT, FORMAT_OPENAI, FORMAT_PREFERENCE]

FORMAT_DESCRIPTIONS = {
    FORMAT_ALPACA: "Alpaca (instruction/input/output)",
    FORMAT_SHAREGPT: "ShareGPT (conversations array with roles)",
    FORMAT_OPENAI: "OpenAI (messages with system/user/assistant)",
    FORMAT_PREFERENCE: "Preference (chosen/rejected for DPO/ORPO/KTO)",
    FORMAT_JSONL: "Simple JSONL (key-value pairs)",
}


# ─── Dataset Entry Types ────────────────────────────────────────────────

class DatasetEntry:
    """Normalized dataset entry — the canonical internal format."""
    
    def __init__(
        self,
        source_format: str = FORMAT_JSONL,
        prompt: str = "",
        completion: str = "",
        instruction: str = "",
        input_text: str = "",
        output: str = "",
        system: str = "",
        messages: Optional[List[Dict[str, str]]] = None,
        conversations: Optional[List[Dict[str, str]]] = None,
        chosen: Optional[List[Dict[str, str]]] = None,
        rejected: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.source_format = source_format
        self.prompt = prompt
        self.completion = completion
        self.instruction = instruction
        self.input_text = input_text
        self.output = output
        self.system = system
        self.messages = messages or []
        self.conversations = conversations or []
        self.chosen = chosen or []
        self.rejected = rejected or []
        self.metadata = metadata or {}
    
    def token_count(self, tokenizer: Optional[Callable[[str], int]] = None) -> int:
        """Estimate token count. If tokenizer provided, use it; else char-based estimate."""
        text = self.get_text()
        if tokenizer:
            return tokenizer(text)
        return len(text) // 4  # rough estimate: ~4 chars per token
    
    def get_text(self) -> str:
        """Get the full text of this entry for tokenization."""
        parts = []
        if self.system:
            parts.append(self.system)
        if self.instruction:
            parts.append(self.instruction)
        if self.input_text:
            parts.append(self.input_text)
        if self.prompt:
            parts.append(self.prompt)
        if self.output:
            parts.append(self.output)
        if self.completion:
            parts.append(self.completion)
        for m in self.messages:
            parts.append(m.get("content", ""))
        for c in self.conversations:
            parts.append(c.get("value", ""))
        for m in self.chosen:
            parts.append(m.get("content", ""))
        for m in self.rejected:
            parts.append(m.get("content", ""))
        return "\n".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON output."""
        d = {"source_format": self.source_format}
        for k in ["prompt", "completion", "instruction", "input_text",
                   "output", "system"]:
            v = getattr(self, k)
            if v:
                d[k] = v
        if self.messages:
            d["messages"] = self.messages
        if self.conversations:
            d["conversations"] = self.conversations
        if self.chosen:
            d["chosen"] = self.chosen
        if self.rejected:
            d["rejected"] = self.rejected
        if self.metadata:
            d["metadata"] = self.metadata
        return d


# ─── Format Detection ──────────────────────────────────────────────────

def detect_format(path: str, sample_lines: int = 5) -> Optional[str]:
    """Detect dataset format from file extension and content."""
    path_lower = path.lower()
    
    # Extension-based hints
    if path_lower.endswith(".parquet"):
        return FORMAT_JSONL  # treat as generic; needs special reader
    
    # Read first few lines
    try:
        lines = _read_head(path, sample_lines)
    except (FileNotFoundError, IOError, json.JSONDecodeError) as e:
        raise ValueError(f"Cannot read dataset {path}: {e}")
    
    if not lines:
        return FORMAT_JSONL
    
    # Analyze content patterns across lines
    formats = {}
    for line in lines:
        try:
            data = json.loads(line) if isinstance(line, str) else line
        except json.JSONDecodeError:
            continue
        
        for fmt in TRAINING_FORMATS:
            score = _detect_format_in_entry(data, fmt)
            if fmt not in formats:
                formats[fmt] = 0
            formats[fmt] += score
    
    if not formats:
        return FORMAT_JSONL
    
    # Return format with highest total score
    best = max(formats, key=formats.get)
    return best if formats[best] > 0 else FORMAT_JSONL


def _read_head(path: str, n: int = 5) -> List[Union[str, Dict]]:
    """Read first n lines/entries from a dataset file."""
    entries = []
    
    if path.endswith(".gz"):
        import gzip
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                line = line.strip()
                if line:
                    entries.append(line)
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                line = line.strip()
                if line:
                    entries.append(line)
    
    return entries


def _detect_format_in_entry(data: Dict, fmt: str) -> int:
    """Score an entry for a given format. Higher = more likely."""
    score = 0
    
    if fmt == FORMAT_ALPACA:
        if "instruction" in data:
            score += 3
        if "output" in data and "instruction" in data:
            score += 2
        if "input" in data and "output" in data:
            score += 1
    
    elif fmt == FORMAT_SHAREGPT:
        if "conversations" in data:
            convs = data["conversations"]
            if isinstance(convs, list) and len(convs) > 0:
                score += 3
                if isinstance(convs[0], dict):
                    if "from" in convs[0] or "role" in convs[0]:
                        score += 2
                    if "value" in convs[0] or "content" in convs[0]:
                        score += 1
    
    elif fmt == FORMAT_OPENAI:
        if "messages" in data:
            msgs = data["messages"]
            if isinstance(msgs, list) and len(msgs) > 0:
                score += 3
                if isinstance(msgs[0], dict):
                    if "role" in msgs[0] and "content" in msgs[0]:
                        score += 3
                    # Check for system role
                    roles = {m.get("role", "") for m in msgs}
                    if roles & {"system", "user", "assistant"}:
                        score += 2
    
    elif fmt == FORMAT_PREFERENCE:
        if "chosen" in data:
            chosen = data["chosen"]
            if isinstance(chosen, list) and len(chosen) > 0:
                score += 3
        if "rejected" in data:
            rejected = data["rejected"]
            if isinstance(rejected, list) and len(rejected) > 0:
                score += 3
        if "chosen" in data and "rejected" in data:
            score += 2
        # Multi-turn preference (list of messages in chosen/rejected)
        if "chosen" in data and isinstance(data["chosen"], list):
            if len(data["chosen"]) > 0 and isinstance(data["chosen"][0], dict):
                if "content" in data["chosen"][0]:
                    score += 1
    
    return score


# ─── Readers ────────────────────────────────────────────────────────────

class BaseReader:
    """Base class for format-specific readers."""
    
    def __init__(self, path: str, stream: bool = False, max_samples: Optional[int] = None):
        self.path = path
        self.stream = stream
        self.max_samples = max_samples
    
    def read(self) -> List[DatasetEntry]:
        """Read all entries into memory."""
        raise NotImplementedError
    
    def stream_entries(self) -> Generator[DatasetEntry, None, None]:
        """Stream entries one by one (memory efficient)."""
        raise NotImplementedError


class JSONLReader(BaseReader):
    """Reads simple JSONL (key-value pairs)."""
    
    def read(self) -> List[DatasetEntry]:
        entries = []
        for entry in self.stream_entries():
            entries.append(entry)
            if self.max_samples and len(entries) >= self.max_samples:
                break
        return entries
    
    def stream_entries(self) -> Generator[DatasetEntry, None, None]:
        open_fn = gzip.open if self.path.endswith(".gz") else open
        with open_fn(self.path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                entry = DatasetEntry(
                    source_format=FORMAT_JSONL,
                    prompt=data.get("prompt", ""),
                    completion=data.get("completion", ""),
                    instruction=data.get("instruction", ""),
                    input_text=data.get("input", ""),
                    output=data.get("output", ""),
                    system=data.get("system", ""),
                    messages=data.get("messages", []),
                    conversations=data.get("conversations", []),
                    chosen=data.get("chosen", []),
                    rejected=data.get("rejected", []),
                    metadata={k: v for k, v in data.items()
                              if k not in ("prompt", "completion", "instruction",
                                           "input", "output", "system", "messages",
                                           "conversations", "chosen", "rejected")},
                )
                yield entry
                if self.max_samples:
                    self.max_samples -= 1
                    if self.max_samples <= 0:
                        break


class AlpacaReader(BaseReader):
    """Reads Alpaca format (instruction/input/output)."""
    
    def read(self) -> List[DatasetEntry]:
        entries = []
        for entry in self.stream_entries():
            entries.append(entry)
            if self.max_samples and len(entries) >= self.max_samples:
                break
        return entries
    
    def stream_entries(self) -> Generator[DatasetEntry, None, None]:
        open_fn = gzip.open if self.path.endswith(".gz") else open
        with open_fn(self.path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                entry = DatasetEntry(
                    source_format=FORMAT_ALPACA,
                    instruction=data.get("instruction", ""),
                    input_text=data.get("input", ""),
                    output=data.get("output", ""),
                )
                yield entry
                if self.max_samples:
                    self.max_samples -= 1
                    if self.max_samples <= 0:
                        break


class ShareGPTReader(BaseReader):
    """Reads ShareGPT format (conversations array)."""
    
    def read(self) -> List[DatasetEntry]:
        entries = []
        for entry in self.stream_entries():
            entries.append(entry)
            if self.max_samples and len(entries) >= self.max_samples:
                break
        return entries
    
    def stream_entries(self) -> Generator[DatasetEntry, None, None]:
        open_fn = gzip.open if self.path.endswith(".gz") else open
        with open_fn(self.path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                conversations = data.get("conversations", [])
                entry = DatasetEntry(
                    source_format=FORMAT_SHAREGPT,
                    conversations=conversations,
                )
                yield entry
                if self.max_samples:
                    self.max_samples -= 1
                    if self.max_samples <= 0:
                        break


class OpenAIReader(BaseReader):
    """Reads OpenAI messages format."""
    
    def read(self) -> List[DatasetEntry]:
        entries = []
        for entry in self.stream_entries():
            entries.append(entry)
            if self.max_samples and len(entries) >= self.max_samples:
                break
        return entries
    
    def stream_entries(self) -> Generator[DatasetEntry, None, None]:
        open_fn = gzip.open if self.path.endswith(".gz") else open
        with open_fn(self.path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                messages = data.get("messages", [])
                # Extract system message if present
                system = ""
                for m in messages:
                    if m.get("role") == "system":
                        system = m.get("content", "")
                        break
                
                entry = DatasetEntry(
                    source_format=FORMAT_OPENAI,
                    messages=messages,
                    system=system,
                )
                yield entry
                if self.max_samples:
                    self.max_samples -= 1
                    if self.max_samples <= 0:
                        break


class PreferenceReader(BaseReader):
    """Reads preference format (chosen/rejected for DPO/ORPO/KTO)."""
    
    def read(self) -> List[DatasetEntry]:
        entries = []
        for entry in self.stream_entries():
            entries.append(entry)
            if self.max_samples and len(entries) >= self.max_samples:
                break
        return entries
    
    def stream_entries(self) -> Generator[DatasetEntry, None, None]:
        open_fn = gzip.open if self.path.endswith(".gz") else open
        with open_fn(self.path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                chosen = data.get("chosen", [])
                rejected = data.get("rejected", [])
                
                entry = DatasetEntry(
                    source_format=FORMAT_PREFERENCE,
                    chosen=chosen,
                    rejected=rejected,
                )
                yield entry
                if self.max_samples:
                    self.max_samples -= 1
                    if self.max_samples <= 0:
                        break


# ─── Reader Factory ────────────────────────────────────────────────────

def get_reader(
    path: str,
    format: Optional[str] = None,
    stream: bool = False,
    max_samples: Optional[int] = None,
) -> BaseReader:
    """Get the appropriate reader for a dataset."""
    if format is None:
        format = detect_format(path)
    
    readers = {
        FORMAT_ALPACA: AlpacaReader,
        FORMAT_SHAREGPT: ShareGPTReader,
        FORMAT_OPENAI: OpenAIReader,
        FORMAT_PREFERENCE: PreferenceReader,
        FORMAT_JSONL: JSONLReader,
    }
    
    reader_cls = readers.get(format)
    if reader_cls is None:
        raise ValueError(f"Unknown format: {format}. Supported: {list(readers.keys())}")
    
    return reader_cls(path, stream=stream, max_samples=max_samples)


# ─── Converters ────────────────────────────────────────────────────────

def convert_dataset(
    input_path: str,
    output_path: str,
    target_format: str = FORMAT_OPENAI,
    input_format: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Convert a dataset from one format to another."""
    reader = get_reader(input_path, format=input_format)
    entries = reader.read()
    
    if max_samples and len(entries) > max_samples:
        entries = entries[:max_samples]
    
    converted = []
    for entry in entries:
        converted.append(_convert_entry(entry, target_format))
    
    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    return {
        "input_path": input_path,
        "output_path": output_path,
        "input_format": input_format or detect_format(input_path),
        "output_format": target_format,
        "samples": len(converted),
    }


def _convert_entry(entry: DatasetEntry, target_format: str) -> Dict[str, Any]:
    """Convert a single entry to target format."""
    if target_format == FORMAT_ALPACA:
        if entry.messages:
            return _messages_to_alpaca(entry.messages)
        if entry.conversations:
            return _conversations_to_alpaca(entry.conversations)
        return {
            "instruction": entry.instruction or entry.prompt,
            "input": entry.input_text,
            "output": entry.output or entry.completion,
        }
    
    elif target_format == FORMAT_SHAREGPT:
        if entry.messages:
            return _messages_to_sharegpt(entry.messages)
        if entry.conversations:
            return {"conversations": entry.conversations}
        # Default: wrap as single-turn
        return {
            "conversations": [
                {"from": "human", "value": entry.instruction or entry.prompt},
                {"from": "gpt", "value": entry.output or entry.completion},
            ]
        }
    
    elif target_format == FORMAT_OPENAI:
        if entry.messages:
            return {"messages": entry.messages}
        if entry.conversations:
            return _conversations_to_openai(entry.conversations)
        # Build from fields
        msgs = []
        if entry.system:
            msgs.append({"role": "system", "content": entry.system})
        msgs.append({
            "role": "user",
            "content": entry.instruction or entry.prompt,
        })
        if entry.input_text:
            msgs[-1]["content"] += "\n\n" + entry.input_text
        if entry.output or entry.completion:
            msgs.append({
                "role": "assistant",
                "content": entry.output or entry.completion,
            })
        return {"messages": msgs}
    
    elif target_format == FORMAT_PREFERENCE:
        if entry.chosen and entry.rejected:
            return {"chosen": entry.chosen, "rejected": entry.rejected}
        # Can't convert without preference data
        return {"chosen": [], "rejected": []}
    
    else:
        # Default: raw dict
        return entry.to_dict()


def _messages_to_alpaca(messages: List[Dict[str, str]]) -> Dict[str, str]:
    """Convert OpenAI messages to Alpaca format."""
    instr = ""
    output = ""
    input_text = ""
    for m in messages:
        if m.get("role") == "system":
            instr = m.get("content", "")
        elif m.get("role") == "user":
            if not instr:
                instr = m.get("content", "")
            else:
                input_text = m.get("content", "")
        elif m.get("role") == "assistant":
            output = m.get("content", "")
    return {"instruction": instr, "input": input_text, "output": output}


def _conversations_to_alpaca(convs: List[Dict[str, str]]) -> Dict[str, str]:
    """Convert ShareGPT conversations to Alpaca format."""
    instr = ""
    output = ""
    for c in convs:
        role = c.get("from", c.get("role", ""))
        val = c.get("value", c.get("content", ""))
        if role in ("human", "user"):
            instr = val
        elif role in ("gpt", "assistant"):
            output = val
    return {"instruction": instr, "input": "", "output": output}


def _messages_to_sharegpt(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Convert OpenAI messages to ShareGPT format."""
    role_map = {"user": "human", "assistant": "gpt", "system": "system"}
    convs = []
    for m in messages:
        role = role_map.get(m.get("role", ""), m.get("role", ""))
        convs.append({"from": role, "value": m.get("content", "")})
    return {"conversations": convs}


def _conversations_to_openai(convs: List[Dict[str, str]]) -> Dict[str, Any]:
    """Convert ShareGPT conversations to OpenAI messages format."""
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    msgs = []
    for c in convs:
        role = role_map.get(c.get("from", ""), c.get("role", "user"))
        msgs.append({"role": role, "content": c.get("value", c.get("content", ""))})
    return {"messages": msgs}


# ─── Streaming Dataset ─────────────────────────────────────────────────

class StreamingDataset:
    """Lazy-load datasets larger than RAM using generator-based iteration.
    
    Supports:
    - JSONL streaming (line-by-line)
    - GZip compressed JSONL
    - HuggingFace datasets streaming
    - Memory-mapped reading for fixed-size records
    """
    
    def __init__(
        self,
        path: str,
        format: Optional[str] = None,
        buffer_size: int = 1000,
    ):
        self.path = path
        self.format = format or detect_format(path)
        self.buffer_size = buffer_size
        self._reader = get_reader(path, format=self.format)
        self._cache: List[DatasetEntry] = []
        self._generator = self._reader.stream_entries()
        self._exhausted = False
    
    def __len__(self) -> int:
        """Return total sample count (reads entire file if not cached)."""
        if hasattr(self, "_total_samples"):
            return self._total_samples
        count = 0
        for _ in self.stream_entries():
            count += 1
            if count > 100_000:
                # Don't count beyond 100K for large datasets
                self._total_samples_approx = True
                return count
        self._total_samples = count
        self._total_samples_approx = False
        return count
    
    def __iter__(self) -> Iterator[DatasetEntry]:
        return self.stream_entries()
    
    def __getitem__(self, idx: int) -> DatasetEntry:
        """Random access by index (reads up to index if needed)."""
        if idx < len(self._cache):
            return self._cache[idx]
        
        # Fast-forward generator
        for entry in self._generator:
            self._cache.append(entry)
            if len(self._cache) > idx:
                return entry
        
        raise IndexError(f"Index {idx} out of range ({len(self._cache)} entries)")
    
    def stream_entries(self) -> Generator[DatasetEntry, None, None]:
        """Stream all entries, yielding cached ones first."""
        # Yield cached entries
        yield from self._cache
        
        # Stream new entries
        for entry in self._generator:
            self._cache.append(entry)
            yield entry
        
        self._exhausted = True
    
    def batch(self, batch_size: int) -> Generator[List[DatasetEntry], None, None]:
        """Yield batches of entries."""
        batch = []
        for entry in self.stream_entries():
            batch.append(entry)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
    
    def count_lines(self) -> int:
        """Quick line count without parsing JSON."""
        count = 0
        open_fn = gzip.open if self.path.endswith(".gz") else open
        with open_fn(self.path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count
    
    def reset(self):
        """Reset the stream (re-reads from file)."""
        self._cache = []
        self._reader = get_reader(self.path, format=self.format)
        self._generator = self._reader.stream_entries()
        self._exhausted = False


def load_hf_dataset(
    path: str,
    split: str = "train",
    streaming: bool = True,
    max_samples: Optional[int] = None,
) -> "StreamingDataset":
    """Load a dataset from HuggingFace hub or local path."""
    try:
        import datasets
    except ImportError:
        raise ImportError(
            "HuggingFace datasets library required. Install: pip install datasets"
        )
    
    ds = datasets.load_dataset(
        path,
        split=split,
        streaming=streaming,
    )
    
    if max_samples:
        ds = ds.take(max_samples)
    
    # Convert to streaming JSONL adapter
    hf_path = _hf_to_temp_jsonl(ds, max_samples=max_samples)
    return StreamingDataset(hf_path, format=FORMAT_JSONL)


def _hf_to_temp_jsonl(ds, max_samples: Optional[int] = None) -> str:
    """Write HF dataset samples to a temporary JSONL for the reader pipeline."""
    import tempfile
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    count = 0
    for sample in ds:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        count += 1
        if max_samples and count >= max_samples:
            break
    f.close()
    return f.name


# ─── Auto-Labeling ─────────────────────────────────────────────────────

class AutoLabeler:
    """Auto-generate completions using a language model.
    
    Supports:
    - Single-prompt completion
    - Batch processing with configurable concurrency
    - Confidence scoring based on logprobs
    - Multi-model comparison
    """
    
    def __init__(
        self,
        api_base: str = "http://127.0.0.1:8080",
        model: str = "",
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        timeout: int = 120,
    ):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.timeout = timeout
    
    def generate(self, prompt: str, system: str = "") -> Dict[str, Any]:
        """Generate a completion for a single prompt."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        return self._call_api(messages)
    
    def _call_api(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Call the OpenAI-compatible API."""
        import urllib.request
        
        data = json.dumps({
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
        }).encode()
        
        req = urllib.request.Request(
            f"{self.api_base}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read())
            
            choice = result.get("choices", [{}])[0]
            message = choice.get("message", {})
            logprobs = choice.get("logprobs", None)
            
            return {
                "text": message.get("content", ""),
                "finish_reason": choice.get("finish_reason", ""),
                "logprobs": logprobs,
                "usage": result.get("usage", {}),
                "success": True,
            }
        except Exception as e:
            return {
                "text": "",
                "finish_reason": "error",
                "error": str(e),
                "success": False,
            }
    
    def label_entry(self, entry: DatasetEntry) -> DatasetEntry:
        """Generate completion for an entry."""
        prompt = entry.instruction or entry.prompt
        if not prompt and entry.conversations:
            # Take the last user message
            for c in reversed(entry.conversations):
                if c.get("from") in ("human", "user"):
                    prompt = c.get("value", "")
                    break
        
        if not prompt and entry.messages:
            for m in reversed(entry.messages):
                if m.get("role") == "user":
                    prompt = m.get("content", "")
                    break
        
        if not prompt:
            return entry
        
        result = self.generate(prompt, system=entry.system)
        entry.completion = result.get("text", "")
        entry.metadata["labeling"] = {
            "success": result.get("success", False),
            "finish_reason": result.get("finish_reason", ""),
            "tokens": result.get("usage", {}).get("completion_tokens", 0),
            "error": result.get("error"),
        }
        return entry
    
    def label_batch(
        self,
        entries: List[DatasetEntry],
        batch_size: int = 4,
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[DatasetEntry]:
        """Label entries in batches, optionally with parallelism."""
        results = []
        total = len(entries)
        
        for i in range(0, total, batch_size):
            batch = entries[i:i + batch_size]
            thread_results = [None] * len(batch)
            threads = []
            
            def _label(idx: int, entry: DatasetEntry, result_list: List, pos: int):
                result_list[pos] = self.label_entry(entry)
            
            for j, entry in enumerate(batch):
                t = threading.Thread(
                    target=_label,
                    args=(j, entry, thread_results, j),
                    daemon=True,
                )
                t.start()
                threads.append(t)
            
            for t in threads:
                t.join(timeout=self.timeout + 10)
            
            results.extend(thread_results)
            
            if callback:
                callback(i + len(batch), total)
        
        return results


def compute_confidence(entry: DatasetEntry) -> float:
    """Compute a confidence score for an auto-labeled entry.
    
    Uses logprobs if available, otherwise heuristics based on
    completion length and content diversity.
    """
    logprobs = entry.metadata.get("labeling", {}).get("logprobs")
    if logprobs:
        # Average log probability as confidence
        tokens = logprobs.get("tokens", [])
        token_logprobs = logprobs.get("token_logprobs", [])
        if token_logprobs:
            avg_logprob = sum(token_logprobs) / len(token_logprobs)
            # Convert to 0-1 scale: exp(avg_logprob / ln(2)) normalized
            # At -0.5 nats → ~0.6, at -2.0 nats → ~0.13
            return max(0.0, min(1.0, math.exp(avg_logprob)))
    
    # Heuristic: length diversity
    completion = entry.completion
    if not completion:
        return 0.0
    
    # Unique token-ratio heuristic
    words = completion.split()
    if len(words) < 2:
        return 0.3
    
    unique_ratio = len(set(words)) / max(len(words), 1)
    length_score = min(1.0, len(completion) / 500)
    
    return 0.3 + 0.4 * unique_ratio + 0.3 * length_score


# ─── Dataset Statistics ─────────────────────────────────────────────────

class DatasetStats:
    """Compute and store dataset statistics."""
    
    def __init__(self, entries: List[DatasetEntry]):
        self.entries = entries
        self.total_samples = len(entries)
        self.total_tokens = 0
        self.vocab = set()
        self.lengths: List[int] = []
        self.prompt_lengths: List[int] = []
        self.completion_lengths: List[int] = []
        self.format_counts: Dict[str, int] = {}
        self._compute()
    
    def _compute(self):
        """Compute all statistics."""
        for entry in self.entries:
            text = entry.get_text()
            length = len(text)
            self.lengths.append(length)
            self.prompt_lengths.append(len(entry.prompt or entry.instruction or ""))
            self.completion_lengths.append(len(entry.completion or entry.output or ""))
            self.vocab.update(text.lower().split())
            
            fmt = entry.source_format
            self.format_counts[fmt] = self.format_counts.get(fmt, 0) + 1
    
    @property
    def avg_length(self) -> float:
        return np.mean(self.lengths) if self.lengths else 0.0
    
    @property
    def median_length(self) -> float:
        return float(np.median(self.lengths)) if self.lengths else 0.0
    
    @property
    def max_length(self) -> int:
        return max(self.lengths) if self.lengths else 0
    
    @property
    def min_length(self) -> int:
        return min(self.lengths) if self.lengths else 0
    
    @property
    def std_length(self) -> float:
        return float(np.std(self.lengths)) if self.lengths else 0.0
    
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
    
    @property
    def estimated_tokens(self) -> int:
        return sum(max(1, l // 4) for l in self.lengths)
    
    @property
    def format_distribution(self) -> Dict[str, int]:
        return self.format_counts
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "estimated_tokens": self.estimated_tokens,
            "vocab_size": self.vocab_size,
            "length": {
                "mean": round(self.avg_length, 1),
                "median": round(self.median_length, 1),
                "std": round(self.std_length, 1),
                "min": self.min_length,
                "max": self.max_length,
            },
            "prompt_length": {
                "mean": round(np.mean(self.prompt_lengths), 1) if self.prompt_lengths else 0,
                "median": round(float(np.median(self.prompt_lengths)), 1) if self.prompt_lengths else 0,
            },
            "completion_length": {
                "mean": round(np.mean(self.completion_lengths), 1) if self.completion_lengths else 0,
                "median": round(float(np.median(self.completion_lengths)), 1) if self.completion_lengths else 0,
            },
            "format_distribution": self.format_distribution,
        }
    
    def length_distribution(self, bins: int = 10) -> Dict[str, Any]:
        """Get histogram of sample lengths."""
        if not self.lengths:
            return {"bins": [], "counts": []}
        
        hist, edges = np.histogram(self.lengths, bins=bins)
        return {
            "bins": [float(e) for e in edges],
            "counts": [int(c) for c in hist],
        }


def compute_stats(entries: List[DatasetEntry]) -> DatasetStats:
    """Convenience function to compute dataset stats."""
    return DatasetStats(entries)


# ─── Dataset Metadata ─────────────────────────────────────────────────

def get_dataset_info(path: str) -> Dict[str, Any]:
    """Get comprehensive info about a dataset file."""
    if not os.path.exists(path):
        return {"error": f"File not found: {path}"}
    
    info = {
        "path": path,
        "name": os.path.basename(path),
        "size_bytes": os.path.getsize(path),
        "size_display": _format_size(os.path.getsize(path)),
        "modified": os.path.getmtime(path),
    }
    
    # Detect format
    try:
        detected = detect_format(path)
        info["format"] = detected
        info["format_description"] = FORMAT_DESCRIPTIONS.get(detected, detected)
    except Exception as e:
        info["format"] = "unknown"
        info["format_error"] = str(e)
    
    # Read samples
    reader = get_reader(path, format=info.get("format"))
    try:
        sample_entries = reader.read()[:3]
        info["samples_preview"] = [
            entry.to_dict() for entry in sample_entries
        ]
    except Exception as e:
        info["samples_error"] = str(e)
    
    # Count lines (quick)
    try:
        info["line_count"] = _count_lines(path)
    except Exception as e:
        info["line_count"] = 0
    
    return info


def _count_lines(path: str) -> int:
    """Quick line count."""
    count = 0
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _format_size(size: int) -> str:
    """Format file size in human-readable form."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


# ─── File Discovery ────────────────────────────────────────────────────

def find_datasets(directory: str = ".") -> List[Dict[str, Any]]:
    """Find all dataset files in a directory."""
    datasets = []
    dir_path = Path(directory).resolve()
    
    patterns = ["*.jsonl", "*.jsonl.gz", "*.json", "*.json.gz"]
    
    for pattern in patterns:
        for f in sorted(dir_path.glob(pattern)):
            # Filter out non-dataset files
            name = f.name.lower()
            if any(x in name for x in ("config", "tokenizer", "model", "training_args")):
                continue
            
            try:
                fmt = detect_format(str(f))
            except Exception:
                fmt = "unknown"
            
            datasets.append({
                "name": f.name,
                "path": str(f),
                "size_bytes": f.stat().st_size,
                "size_display": _format_size(f.stat().st_size),
                "format": fmt,
                "format_description": FORMAT_DESCRIPTIONS.get(fmt, fmt),
            })
    
    return datasets


# ─── Dataset Splitting ─────────────────────────────────────────────────

def split_dataset(
    entries: List[DatasetEntry],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
) -> Dict[str, List[DatasetEntry]]:
    """Split dataset into train/val/test sets."""
    total = len(entries)
    indices = list(range(total))
    
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
    
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    return {
        "train": [entries[i] for i in indices[:train_end]],
        "val": [entries[i] for i in indices[train_end:val_end]],
        "test": [entries[i] for i in indices[val_end:]],
    }


# ─── Utility Functions ─────────────────────────────────────────────────

def write_jsonl(entries: List[DatasetEntry], path: str, format: str = FORMAT_OPENAI):
    """Write entries to a JSONL file in the specified format."""
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            converted = _convert_entry(entry, format)
            f.write(json.dumps(converted, ensure_ascii=False) + "\n")


def load_dataset(
    path: str,
    format: Optional[str] = None,
    max_samples: Optional[int] = None,
    stream: bool = False,
) -> Union[List[DatasetEntry], StreamingDataset]:
    """Load a dataset, optionally streaming."""
    if stream:
        return StreamingDataset(path, format=format)
    
    reader = get_reader(path, format=format, max_samples=max_samples)
    return reader.read()
