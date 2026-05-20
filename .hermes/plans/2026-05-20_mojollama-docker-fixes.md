# MojoLlama Docker Fix Plan

**Date:** 2026-05-20

---

## 1. Fix engine script path resolution

**`docker-entrypoint.sh:171`** — Add `benchmarks/` subdirectory:
```bash
ENGINE_PATH="/app/src/mojollama/benchmarks/${ENGINE_SCRIPT}"
```

## 2. Fix healthcheck port to respect `$PORT`

**`Dockerfile:144-145`**:
```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:${PORT:-8080}/health || exit 1
```

**`docker-compose.yml:43`** — Same fix:
```yaml
test: ["CMD", "curl", "-sf", "http://localhost:${PORT:-8080}/health"]
```

## 3. Remove `deploy.resources` for Zima OS compatibility

**`docker-compose.yml:51-58`** — `deploy.resources` is Swarm-only, silently ignored by `docker compose up`. Replace with compose-v2-compatible approach:

```yaml
# Deploy block removed — resource limits are Swarm-only
# Use `docker compose run --cpus=N --memory=N` instead
# For CasaOS / Zima OS: configure resource limits in the CasaOS UI
```

## 4. Fix named volume permission clash for Zima OS

**`docker-compose.yml:31`** — Named volume owned by root clashes with non-root `mojollama` user. Zima OS CasaOS expects root-owned containers. Either:

- (Option A) Change config to bind mount:
  ```yaml
  - "${CONFIG_VOLUME:-./config}:/home/mojollama/.mojollama"
  ```

- (Option B) Create a separate mojollama-dedicated Dockerfile variant that runs as root for CasaOS

**`Dockerfile:153`** — Run as root for Zima OS compatibility (CasaOS standard):
```dockerfile
# USER mojollama   ← remove or make conditional
```

## 5. Add HuggingFace model download at startup

**`Dockerfile:99-103`** — Add `huggingface_hub` to pip install list:
```dockerfile
RUN pip install --no-cache-dir gguf numpy transformers requests huggingface_hub
```

**`docker-entrypoint.sh`** — Before engine start, add auto-download logic:
```bash
# ── Auto-download model from HuggingFace if not found locally ──
if [[ -n "$MODEL_PATH" && ! -f "$MODEL_PATH" ]]; then
    if [[ -n "$HF_REPO" && -n "$HF_FILE" ]]; then
        log "Downloading ${HF_REPO}/${HF_FILE}..."
        python3 -c "
from huggingface_hub import hf_hub_download
import os, sys
path = hf_hub_download(
    repo_id='${HF_REPO}',
    filename='${HF_FILE}',
    local_dir='/models'
)
print(f'Downloaded to {path}')
" || warn "HuggingFace download failed; server may error if model missing."
    fi
fi
```

**`docker-compose.yml`** — Add HuggingFace env vars:
```yaml
environment:
  - HF_REPO=${HF_REPO:-}
  - HF_FILE=${HF_FILE:-}
```

## 6. Fix `backends.py` default model path

**`src/mojollama/backends.py:290`** — Change default from `/tmp/models/` to `/models/`:
```python
self.model_path = model_path or "/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
```

## 7. Default model for Zima OS (low-resource)

Since Qwen3-30B is too large for low-resource devices, document that Zima OS users should use:

```bash
# Use TinyLlama (~0.7 GB) instead of Qwen3-30B (~18 GB)
HF_REPO=bartowski/TinyLlama-1.1B-GGUF
HF_FILE=tinyllama-1.1b.Q4_K_M.gguf
MODEL_PATH=/models/tinyllama-1.1b.Q4_K_M.gguf
```

Or mount a local GGUF model:
```bash
docker compose run -v /path/to/model.gguf:/models/model.gguf:ro -e MODEL_PATH=/models/model.gguf mojollama
```
