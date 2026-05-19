#!/usr/bin/env python3
"""Run standalone test under GDB to get backtrace on segfault."""
import subprocess, sys

cmd = [
    "gdb", "-batch", "-ex", "run", "-ex", "bt",
    "--args", "python3", "-u",
    "/onedev-workspace/work/src/mojollama/test_forward_standalone.py",
    "/tmp/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
]

# GDB needs the environment
env = {"OMP_NUM_THREADS": "32", "OMP_PROC_BIND": "close", "OMP_PLACES": "{0}:32:1"}
result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
print("STDOUT:", result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout)
print("STDERR:", result.stderr[-5000:] if len(result.stderr) > 5000 else result.stderr)
print("Return code:", result.returncode)
