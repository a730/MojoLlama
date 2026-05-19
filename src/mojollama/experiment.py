#!/usr/bin/env python3
"""MojoLlama Experiment Tracking — W&B + MLflow integration.

Provides a unified API for experiment tracking that gracefully degrades
when external services aren't available. All runs are persisted locally
in a SQLite database for offline viewing and comparison.

Usage:
    from mojollama.experiment import ExperimentTracker
    tracker = ExperimentTracker()
    run = tracker.start_run("my-experiment", tags={"model": "Llama-3.2-1B"})
    tracker.log_metric("loss", 0.5, step=10)
    tracker.log_params({"lr": 1e-4, "batch_size": 4})
    tracker.stop_run()
"""

import os
import sys
import json
import time
import uuid
import threading
import sqlite3
import atexit
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

# ─── Global state ─────────────────────────────────────────────────────

EXPERIMENTS_DB = os.environ.get(
    "MOJOLLAMA_EXPERIMENTS_DB",
    str(Path(__file__).parent.parent.parent / "experiments.db")
)

# Thread-local so multiple server threads don't conflict
_local = threading.local()

# Global lock for DB writes
_db_lock = threading.Lock()

# ─── DB Schema ─────────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    status TEXT DEFAULT 'running',  -- running, completed, failed, cancelled
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    finished_at REAL,
    tags TEXT DEFAULT '{}',          -- JSON dict
    params TEXT DEFAULT '{}',        -- JSON dict
    metrics TEXT DEFAULT '{}',        -- JSON dict of metric_name -> [values]
    artifacts TEXT DEFAULT '[]',      -- JSON array of artifact paths
    metadata TEXT DEFAULT '{}'        -- JSON dict of extra info
);

CREATE TABLE IF NOT EXISTS metric_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    step INTEGER NOT NULL,
    value REAL NOT NULL,
    timestamp REAL NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

CREATE INDEX IF NOT EXISTS idx_metric_points_exp
    ON metric_points(experiment_id, metric_name);

CREATE TABLE IF NOT EXISTS model_cards (
    id TEXT PRIMARY KEY,
    experiment_id TEXT,
    model_name TEXT NOT NULL,
    architecture TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    description TEXT DEFAULT '',
    base_model TEXT DEFAULT '',
    dataset TEXT DEFAULT '',
    training_config TEXT DEFAULT '{}',
    metrics TEXT DEFAULT '{}',
    quant_type TEXT DEFAULT '',
    params TEXT DEFAULT '{}',
    tags TEXT DEFAULT '[]',
    license TEXT DEFAULT 'MIT',
    readme TEXT DEFAULT ''
);
"""

# ─── Helpers ──────────────────────────────────────────────────────────

def _get_db() -> sqlite3.Connection:
    """Get thread-local DB connection."""
    if not hasattr(_local, 'conn') or _local.conn is None:
        _local.conn = sqlite3.connect(EXPERIMENTS_DB)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA busy_timeout=5000")
    return _local.conn

def _init_db():
    """Initialize the database schema."""
    with _db_lock:
        db = _get_db()
        db.executescript(_SCHEMA_SQL)
        db.commit()

def _gen_id() -> str:
    return uuid.uuid4().hex[:12]

def _now() -> float:
    return time.time()

def _run_number(name: str) -> int:
    """Get the next run number for a given experiment name."""
    with _db_lock:
        db = _get_db()
        row = db.execute(
            "SELECT COUNT(*) FROM experiments WHERE name = ?",
            (name,)
        ).fetchone()
        return (row[0] if row else 0) + 1


# ─── W&B Integration ──────────────────────────────────────────────────

_wandb_available = None

def _check_wandb():
    global _wandb_available
    if _wandb_available is None:
        try:
            import wandb
            _wandb_available = True
        except ImportError:
            _wandb_available = False
    return _wandb_available

_wandb_runs = {}  # local_experiment_id -> wandb_run


# ─── MLflow Integration ───────────────────────────────────────────────

_mlflow_available = None

def _check_mlflow():
    global _mlflow_available
    if _mlflow_available is None:
        try:
            import mlflow
            _mlflow_available = True
        except ImportError:
            _mlflow_available = False
    return _mlflow_available

_mlflow_runs = {}  # local_experiment_id -> mlflow_run_id


# ─── ExperimentTracker ────────────────────────────────────────────────

class ExperimentTracker:
    """Main experiment tracking interface.

    Thread-safe. Designed to work both from the CLI and the API server.

    Examples:
        tracker = ExperimentTracker()

        # Start a run
        run = tracker.start_run("qwen-finetune-v1", tags={"model": "Qwen3-30B"})

        # Log metrics during training
        for step in range(100):
            loss = compute_loss()
            tracker.log_metric("loss", loss, step=step)
            tracker.log_metric("accuracy", acc, step=step)

        # Log hyperparams once
        tracker.log_params({"lr": 0.001, "batch_size": 4, "lora_rank": 16})

        # Save artifacts
        tracker.log_artifact("lora-adapter.gguf", artifact_type="model")

        # Finish
        tracker.stop_run()
    """

    def __init__(self, db_path: Optional[str] = None):
        global EXPERIMENTS_DB
        if db_path:
            EXPERIMENTS_DB = db_path
        _init_db()
        self._current_run_id: Optional[str] = None
        self._lock = threading.Lock()

    # ── Run Management ───────────────────────────────────────────────

    def start_run(
        self,
        name: str,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_wandb: bool = True,
        auto_mlflow: bool = True,
    ) -> Dict[str, Any]:
        """Start a new experiment run.

        Returns the run record as a dict.
        """
        run_id = _gen_id()
        run_num = _run_number(name)
        display_name = f"{name}-run-{run_num}"
        now = _now()
        tags_dict = tags or {}
        tags_dict["run_number"] = str(run_num)

        record = {
            "id": run_id,
            "name": name,
            "display_name": display_name,
            "description": description,
            "status": "running",
            "created_at": now,
            "updated_at": now,
            "finished_at": None,
            "tags": json.dumps(tags_dict),
            "params": json.dumps(params or {}),
            "metrics": "{}",
            "artifacts": "[]",
            "metadata": json.dumps(metadata or {}),
        }

        with _db_lock:
            db = _get_db()
            db.execute(
                """INSERT INTO experiments
                   (id, name, description, status, created_at, updated_at,
                    finished_at, tags, params, metrics, artifacts, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (record["id"], name, description, "running", now, now, None,
                 record["tags"], record["params"], record["metrics"],
                 record["artifacts"], record["metadata"])
            )
            db.commit()

        with self._lock:
            self._current_run_id = run_id

        # Optional W&B integration
        if auto_wandb and _check_wandb():
            try:
                import wandb
                wandb_run = wandb.init(
                    project=name,
                    name=display_name,
                    config=params or {},
                    tags=list(tags_dict.values()) if tags_dict else [],
                    reinit=True,
                )
                if wandb_run:
                    _wandb_runs[run_id] = wandb_run
            except Exception as e:
                print(f"[Experiment] W&B init failed (non-fatal): {e}", file=sys.stderr)

        # Optional MLflow integration
        if auto_mlflow and _check_mlflow():
            try:
                import mlflow
                mlflow.set_experiment(name)
                with mlflow.start_run(run_name=display_name):
                    mlflow_run = mlflow.active_run()
                    if mlflow_run:
                        _mlflow_runs[run_id] = mlflow_run.info.run_id
                    if params:
                        mlflow.log_params(params)
                    if tags_dict:
                        mlflow.set_tags(tags_dict)
            except Exception as e:
                print(f"[Experiment] MLflow init failed (non-fatal): {e}", file=sys.stderr)

        return self.get_run(run_id)

    def stop_run(
        self,
        status: str = "completed",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Stop the current run. Status: completed, failed, cancelled."""
        with self._lock:
            run_id = self._current_run_id
            if not run_id:
                return None
            self._current_run_id = None

        now = _now()
        with _db_lock:
            db = _get_db()
            db.execute(
                "UPDATE experiments SET status=?, finished_at=?, updated_at=? WHERE id=?",
                (status, now, now, run_id)
            )
            if metadata:
                db.execute(
                    "UPDATE experiments SET metadata=? WHERE id=?",
                    (json.dumps(metadata), run_id)
                )
            db.commit()

        # Finalize W&B run
        if run_id in _wandb_runs:
            try:
                _wandb_runs[run_id].finish()
            except Exception:
                pass
            del _wandb_runs[run_id]

        # Finalize MLflow run
        if run_id in _mlflow_runs:
            try:
                import mlflow
                mlflow.end_run()
            except Exception:
                pass
            del _mlflow_runs[run_id]

        return self.get_run(run_id)

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get a single experiment run."""
        with _db_lock:
            db = _get_db()
            row = db.execute(
                "SELECT * FROM experiments WHERE id = ?", (run_id,)
            ).fetchone()
            if not row:
                return None
            return self._row_to_dict(row)

    def list_runs(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List experiment runs, newest first."""
        query = "SELECT * FROM experiments"
        params = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with _db_lock:
            db = _get_db()
            rows = db.execute(query, params).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def delete_run(self, run_id: str) -> bool:
        """Delete an experiment run and its metric points."""
        with _db_lock:
            db = _get_db()
            db.execute("DELETE FROM metric_points WHERE experiment_id = ?", (run_id,))
            db.execute("DELETE FROM experiments WHERE id = ?", (run_id,))
            db.commit()
            return db.total_changes > 0

    # ── Metrics ──────────────────────────────────────────────────────

    def log_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> bool:
        """Log a single metric point."""
        rid = run_id or self._current_run_id
        if not rid:
            return False
        if step is None:
            step = self._next_step(rid, name)

        now = _now()
        with _db_lock:
            db = _get_db()
            db.execute(
                "INSERT INTO metric_points (experiment_id, metric_name, step, value, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (rid, name, step, float(value), now)
            )
            # Update the metrics summary in experiments table
            metrics = self._get_metrics_summary(rid)
            db.execute(
                "UPDATE experiments SET metrics=?, updated_at=? WHERE id=?",
                (json.dumps(metrics), now, rid)
            )
            db.commit()

        # Log to W&B
        if rid in _wandb_runs:
            try:
                _wandb_runs[rid].log({name: value, "step": step})
            except Exception:
                pass

        # Log to MLflow
        if rid in _mlflow_runs:
            try:
                import mlflow
                with mlflow.start_run(run_id=_mlflow_runs[rid], nested=True):
                    mlflow.log_metric(name, value, step=step or 0)
            except Exception:
                pass

        return True

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            self.log_metric(name, value, step=step)

    def get_metrics(
        self,
        run_id: str,
        metric_name: Optional[str] = None,
        limit: int = 10000,
    ) -> List[Dict[str, Any]]:
        """Get all metric points for a run, optionally filtered by name."""
        query = "SELECT * FROM metric_points WHERE experiment_id = ?"
        params = [run_id]
        if metric_name:
            query += " AND metric_name = ?"
            params.append(metric_name)
        query += " ORDER BY step ASC, timestamp ASC LIMIT ?"
        params.append(limit)

        with _db_lock:
            db = _get_db()
            rows = db.execute(query, params).fetchall()
            return [
                {
                    "id": r["id"],
                    "metric_name": r["metric_name"],
                    "step": r["step"],
                    "value": r["value"],
                    "timestamp": r["timestamp"],
                }
                for r in rows
            ]

    def get_metrics_chart_data(
        self,
        run_id: str,
        metric_name: str,
    ) -> Dict[str, Any]:
        """Get chart-ready data for a metric across all steps."""
        points = self.get_metrics(run_id, metric_name=metric_name)
        return {
            "name": metric_name,
            "run_id": run_id,
            "steps": [p["step"] for p in points],
            "values": [p["value"] for p in points],
            "min": min(p["value"] for p in points) if points else 0,
            "max": max(p["value"] for p in points) if points else 0,
            "last": points[-1]["value"] if points else 0,
            "count": len(points),
        }

    def compare_metrics(
        self,
        run_ids: List[str],
        metric_name: str,
    ) -> Dict[str, Any]:
        """Compare a specific metric across multiple runs."""
        series = {}
        for rid in run_ids:
            run = self.get_run(rid)
            data = self.get_metrics_chart_data(rid, metric_name)
            series[rid] = {
                "run_name": run["name"] if run else rid,
                "display_name": run.get("display_name", rid) if run else rid,
                **data,
            }
        return {
            "metric": metric_name,
            "run_ids": run_ids,
            "series": series,
        }

    # ── Params ───────────────────────────────────────────────────────

    def log_params(
        self,
        params: Dict[str, Any],
        run_id: Optional[str] = None,
    ) -> bool:
        """Log hyperparameters for the current run."""
        rid = run_id or self._current_run_id
        if not rid:
            return False

        with _db_lock:
            db = _get_db()
            row = db.execute(
                "SELECT params FROM experiments WHERE id = ?", (rid,)
            ).fetchone()
            if not row:
                return False
            existing = json.loads(row["params"] or "{}")
            existing.update(params)
            db.execute(
                "UPDATE experiments SET params=?, updated_at=? WHERE id=?",
                (json.dumps(existing), _now(), rid)
            )
            db.commit()

        # Sync to W&B
        if rid in _wandb_runs:
            try:
                _wandb_runs[rid].config.update(params)
            except Exception:
                pass

        # Sync to MLflow
        if rid in _mlflow_runs:
            try:
                import mlflow
                with mlflow.start_run(run_id=_mlflow_runs[rid], nested=True):
                    mlflow.log_params(params)
            except Exception:
                pass

        return True

    def get_params(self, run_id: str) -> Dict[str, Any]:
        """Get logged parameters for a run."""
        with _db_lock:
            db = _get_db()
            row = db.execute(
                "SELECT params FROM experiments WHERE id = ?", (run_id,)
            ).fetchone()
            if row:
                return json.loads(row["params"] or "{}")
            return {}

    # ── Artifacts ────────────────────────────────────────────────────

    def log_artifact(
        self,
        path: str,
        artifact_type: str = "model",
        description: str = "",
        run_id: Optional[str] = None,
    ) -> bool:
        """Log an artifact file path."""
        rid = run_id or self._current_run_id
        if not rid:
            return False

        artifact = {
            "path": path,
            "type": artifact_type,
            "description": description,
            "timestamp": _now(),
            "size_bytes": os.path.getsize(path) if os.path.exists(path) else 0,
        }

        with _db_lock:
            db = _get_db()
            row = db.execute(
                "SELECT artifacts FROM experiments WHERE id = ?", (rid,)
            ).fetchone()
            if not row:
                return False
            existing = json.loads(row["artifacts"] or "[]")
            existing.append(artifact)
            db.execute(
                "UPDATE experiments SET artifacts=?, updated_at=? WHERE id=?",
                (json.dumps(existing), _now(), rid)
            )
            db.commit()

        # Log to W&B
        if rid in _wandb_runs and os.path.exists(path):
            try:
                _wandb_runs[rid].log_model(
                    path=path,
                    model_name=os.path.basename(path),
                )
            except Exception:
                pass

        return True

    def list_artifacts(self, run_id: str) -> List[Dict[str, Any]]:
        """List artifacts for a run."""
        with _db_lock:
            db = _get_db()
            row = db.execute(
                "SELECT artifacts FROM experiments WHERE id = ?", (run_id,)
            ).fetchone()
            if row:
                return json.loads(row["artifacts"] or "[]")
            return []

    # ── Tags ─────────────────────────────────────────────────────────

    def set_tags(
        self,
        tags: Dict[str, str],
        run_id: Optional[str] = None,
    ) -> bool:
        """Set tags for a run."""
        rid = run_id or self._current_run_id
        if not rid:
            return False

        with _db_lock:
            db = _get_db()
            row = db.execute(
                "SELECT tags FROM experiments WHERE id = ?", (rid,)
            ).fetchone()
            if not row:
                return False
            existing = json.loads(row["tags"] or "{}")
            existing.update(tags)
            db.execute(
                "UPDATE experiments SET tags=?, updated_at=? WHERE id=?",
                (json.dumps(existing), _now(), rid)
            )
            db.commit()
        return True

    def get_tags(self, run_id: str) -> Dict[str, str]:
        """Get tags for a run."""
        with _db_lock:
            db = _get_db()
            row = db.execute(
                "SELECT tags FROM experiments WHERE id = ?", (run_id,)
            ).fetchone()
            if row:
                return json.loads(row["tags"] or "{}")
            return {}

    # ── Model Cards ──────────────────────────────────────────────────

    def create_model_card(
        self,
        model_name: str,
        architecture: str,
        experiment_id: Optional[str] = None,
        description: str = "",
        base_model: str = "",
        dataset: str = "",
        training_config: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        quant_type: str = "",
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        license: str = "MIT",
    ) -> Dict[str, Any]:
        """Create a model card for documentation/sharing."""
        card_id = _gen_id()
        now = _now()
        card = {
            "id": card_id,
            "experiment_id": experiment_id,
            "model_name": model_name,
            "architecture": architecture,
            "created_at": now,
            "updated_at": now,
            "description": description,
            "base_model": base_model,
            "dataset": dataset,
            "training_config": json.dumps(training_config or {}),
            "metrics": json.dumps(metrics or {}),
            "quant_type": quant_type,
            "params": json.dumps(params or {}),
            "tags": json.dumps(tags or []),
            "license": license,
            "readme": "",
        }

        with _db_lock:
            db = _get_db()
            db.execute(
                """INSERT INTO model_cards
                   (id, experiment_id, model_name, architecture, created_at, updated_at,
                    description, base_model, dataset, training_config, metrics,
                    quant_type, params, tags, license, readme)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (card_id, experiment_id, model_name, architecture, now, now,
                 description, base_model, dataset, card["training_config"],
                 card["metrics"], quant_type, card["params"], card["tags"],
                 license, "")
            )
            db.commit()

        return self.get_model_card(card_id)

    def get_model_card(self, card_id: str) -> Optional[Dict[str, Any]]:
        """Get a model card."""
        with _db_lock:
            db = _get_db()
            row = db.execute(
                "SELECT * FROM model_cards WHERE id = ?", (card_id,)
            ).fetchone()
            if not row:
                return None
            return self._card_row_to_dict(row)

    def list_model_cards(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List all model cards."""
        with _db_lock:
            db = _get_db()
            rows = db.execute(
                "SELECT * FROM model_cards ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [self._card_row_to_dict(r) for r in rows]

    def update_model_card(
        self,
        card_id: str,
        **updates,
    ) -> Optional[Dict[str, Any]]:
        """Update fields on a model card."""
        allowed = {"description", "dataset", "metrics", "tags", "license", "readme",
                   "training_config", "params", "quant_type", "base_model"}
        now = _now()
        fields = {}
        for k, v in updates.items():
            if k in allowed:
                if isinstance(v, (dict, list)):
                    fields[k] = json.dumps(v)
                else:
                    fields[k] = str(v)

        if not fields:
            return self.get_model_card(card_id)

        fields["updated_at"] = now
        set_clause = ", ".join(f"{k}=?" for k in fields)
        values = list(fields.values()) + [card_id]

        with _db_lock:
            db = _get_db()
            db.execute(
                f"UPDATE model_cards SET {set_clause} WHERE id=?",
                values
            )
            db.commit()

        return self.get_model_card(card_id)

    def generate_model_card_readme(self, card_id: str) -> Optional[str]:
        """Generate a HuggingFace-style README markdown from model card data."""
        card = self.get_model_card(card_id)
        if not card:
            return None

        lines = [
            f"---",
            f"license: {card.get('license', 'MIT')}",
            f"language: ['en']",
            f"tags: {card.get('tags', '[]')}",
            f"datasets: [{card.get('dataset', '')}]",
            f"base_model: {card.get('base_model', '')}",
            f"---",
            f"",
            f"# {card['model_name']}",
            f"",
            f"{card['description']}",
            f"",
            f"## Model Details",
            f"",
            f"- **Architecture:** {card['architecture']}",
            f"- **Base Model:** {card.get('base_model', 'N/A')}",
            f"- **Quantization:** {card.get('quant_type', 'N/A')}",
            f"- **Parameters:** {card.get('params', '{}')}",
            f"- **License:** {card.get('license', 'MIT')}",
            f"",
        ]

        metrics_raw = card.get("metrics", "{}")
        if isinstance(metrics_raw, str):
            try:
                metrics = json.loads(metrics_raw)
            except json.JSONDecodeError:
                metrics = {}
        else:
            metrics = metrics_raw

        if metrics:
            lines.append(f"## Performance Metrics")
            lines.append(f"")
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            for k, v in metrics.items():
                lines.append(f"| {k} | {v} |")
            lines.append(f"")

        dataset = card.get("dataset", "")
        if dataset:
            lines.append(f"## Training Data")
            lines.append(f"")
            lines.append(f"- **Dataset:** {dataset}")
            lines.append(f"")

        training_config = card.get("training_config", "{}")
        if isinstance(training_config, str):
            try:
                tc = json.loads(training_config)
            except json.JSONDecodeError:
                tc = {}
        else:
            tc = training_config
        if tc:
            lines.append(f"## Training Configuration")
            lines.append(f"")
            for k, v in tc.items():
                lines.append(f"- **{k}:** {v}")
            lines.append(f"")

        lines.append(f"---")
        lines.append(f"*Generated by MojoLlama Studio*")

        readme = "\n".join(lines)

        # Save the generated readme
        with _db_lock:
            db = _get_db()
            db.execute(
                "UPDATE model_cards SET readme=?, updated_at=? WHERE id=?",
                (readme, _now(), card_id)
            )
            db.commit()

        return readme

    def delete_model_card(self, card_id: str) -> bool:
        """Delete a model card."""
        with _db_lock:
            db = _get_db()
            db.execute("DELETE FROM model_cards WHERE id = ?", (card_id,))
            db.commit()
            return db.total_changes > 0

    # ── Internal ─────────────────────────────────────────────────────

    def _next_step(self, run_id: str, metric_name: str) -> int:
        """Get the next step number for a metric."""
        with _db_lock:
            db = _get_db()
            row = db.execute(
                "SELECT COALESCE(MAX(step), 0) as max_step FROM metric_points "
                "WHERE experiment_id = ? AND metric_name = ?",
                (run_id, metric_name)
            ).fetchone()
            return (row[0] if row else 0) + 1

    def _get_metrics_summary(self, run_id: str) -> Dict[str, Any]:
        """Build a summary dict of all metrics for a run (latest values)."""
        db = _get_db()
        rows = db.execute(
            """SELECT metric_name, value FROM metric_points
               WHERE (experiment_id, metric_name, step) IN (
                   SELECT experiment_id, metric_name, MAX(step)
                   FROM metric_points
                   WHERE experiment_id = ?
                   GROUP BY metric_name
               )""",
            (run_id,)
        ).fetchall()
        return {r["metric_name"]: r["value"] for r in rows}

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        # Parse JSON fields
        for field in ["tags", "params", "metrics", "artifacts", "metadata"]:
            if isinstance(d.get(field), str):
                try:
                    d[field] = json.loads(d[field])
                except (json.JSONDecodeError, TypeError):
                    pass
        d["display_name"] = f"{d['name']}-run-{d.get('tags', {}).get('run_number', '?')}"
        return d

    def _card_row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        d: Dict[str, Any] = dict(row)
        for field in ["training_config", "metrics", "params", "tags"]:
            if isinstance(d.get(field), str):
                try:
                    d[field] = json.loads(d[field])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d


# ─── Singleton for server use ─────────────────────────────────────────

_tracker = None

def get_tracker() -> ExperimentTracker:
    global _tracker
    if _tracker is None:
        _tracker = ExperimentTracker()
    return _tracker


# ─── Training Metrics Integration ──────────────────────────────────────

class TrainingSession:
    """Wrapper for a training session that auto-logs metrics.

    Example:
        with TrainingSession("qwen-lora", model="Qwen3-30B", lr=1e-4) as session:
            for epoch in range(10):
                loss = train_one_epoch()
                session.log(epoch=epoch, loss=loss, lr=1e-4)
    """

    def __init__(self, name: str, tracker: Optional[ExperimentTracker] = None, **params):
        self.tracker = tracker or get_tracker()
        self.name = name
        self.params = params
        self.run_id: Optional[str] = None
        self.step = 0

    def __enter__(self):
        run = self.tracker.start_run(
            name=self.name,
            tags={"type": "training"},
            params=self.params,
        )
        self.run_id = run["id"]
        return self

    def __exit__(self, *args):
        status = "completed" if not any(args) else "failed"
        self.tracker.stop_run(status=status)

    def log(self, step: Optional[int] = None, **metrics):
        self.step = step if step is not None else self.step + 1
        self.tracker.log_metrics(metrics, step=self.step)


# ─── Cleanup ──────────────────────────────────────────────────────────

@atexit.register
def _cleanup():
    """Clean up any lingering W&B runs when the Python process exits."""
    for run_id, wandb_run in list(_wandb_runs.items()):
        try:
            wandb_run.finish()
        except Exception:
            pass
    _wandb_runs.clear()


# ─── CLI ──────────────────────────────────────────────────────────────

def main():
    """CLI for experiment tracking."""
    import argparse
    parser = argparse.ArgumentParser(description="MojoLlama Experiment Tracker")
    sub = parser.add_subparsers(dest="command")

    # list
    p_list = sub.add_parser("list", help="List experiment runs")
    p_list.add_argument("--status", help="Filter by status")
    p_list.add_argument("--limit", type=int, default=20)

    # start
    p_start = sub.add_parser("start", help="Start a new experiment run")
    p_start.add_argument("name", help="Experiment name")
    p_start.add_argument("--desc", help="Description")
    p_start.add_argument("--tag", action="append", help="Tags (key=val)")

    # log
    p_log = sub.add_parser("log", help="Log a metric")
    p_log.add_argument("name", help="Metric name")
    p_log.add_argument("value", type=float, help="Metric value")
    p_log.add_argument("--step", type=int, help="Step number")
    p_log.add_argument("--run", help="Run ID (default: current)")

    # params
    p_params = sub.add_parser("params", help="Log hyperparameters")
    p_params.add_argument("params", nargs="+", help="key=val pairs")

    # stop
    p_stop = sub.add_parser("stop", help="Stop current run")
    p_stop.add_argument("--status", default="completed")

    # delete
    p_delete = sub.add_parser("delete", help="Delete a run")
    p_delete.add_argument("run_id", help="Run ID to delete")

    # show
    p_show = sub.add_parser("show", help="Show run details")
    p_show.add_argument("run_id", help="Run ID")

    # model-card
    p_mc = sub.add_parser("model-card", help="Create a model card")
    p_mc.add_argument("model_name", help="Model name")
    p_mc.add_argument("--arch", default="llama", help="Architecture")
    p_mc.add_argument("--desc", help="Description")
    p_mc.add_argument("--base", help="Base model")
    p_mc.add_argument("--dataset", help="Training dataset")
    p_mc.add_argument("--quant", default="", help="Quantization type")
    p_mc.add_argument("--license", default="MIT", help="License")

    p_mc_list = sub.add_parser("model-cards", help="List model cards")

    p_mc_gen = sub.add_parser("gen-readme", help="Generate README for a model card")
    p_mc_gen.add_argument("card_id", help="Model card ID")

    args = parser.parse_args()

    tracker = get_tracker()

    if args.command == "list":
        runs = tracker.list_runs(status=args.status, limit=args.limit)
        if not runs:
            print("No experiment runs found.")
            return
        print(f"{'ID':<14} {'Name':<24} {'Status':<12} {'Created':<20} {'Metrics'}")
        print("-" * 90)
        for r in runs:
            created = datetime.fromtimestamp(r["created_at"]).strftime("%Y-%m-%d %H:%M")
            n_metrics = len(r.get("metrics", {}))
            print(f"{r['id']:<14} {r['display_name']:<24} {r['status']:<12} {created:<20} {n_metrics} metrics")

    elif args.command == "start":
        tags = {}
        if args.tag:
            for t in args.tag:
                if "=" in t:
                    k, v = t.split("=", 1)
                    tags[k] = v
        run = tracker.start_run(args.name, description=args.desc or "", tags=tags)
        print(f"Started run: {run['id']}")
        print(f"  Name: {run['display_name']}")
        print(f"  Export MOJOLLAMA_RUN_ID={run['id']}")

    elif args.command == "log":
        rid = args.run or os.environ.get("MOJOLLAMA_RUN_ID")
        tracker.log_metric(args.name, args.value, step=args.step, run_id=rid)
        print(f"Logged {args.name}={args.value} (step={args.step or 'auto'})")

    elif args.command == "params":
        params = {}
        for p in args.params:
            if "=" in p:
                k, v = p.split("=", 1)
                params[k] = v
        tracker.log_params(params)
        print(f"Logged {len(params)} parameters")

    elif args.command == "stop":
        run = tracker.stop_run(status=args.status)
        if run:
            print(f"Run {run['id']} stopped: {args.status}")
        else:
            print("No active run to stop.")

    elif args.command == "delete":
        if tracker.delete_run(args.run_id):
            print(f"Deleted run {args.run_id}")
        else:
            print(f"Run {args.run_id} not found")

    elif args.command == "show":
        run = tracker.get_run(args.run_id)
        if not run:
            print(f"Run {args.run_id} not found")
            return
        print(f"Run: {run['id']}")
        print(f"  Name: {run['display_name']}")
        print(f"  Status: {run['status']}")
        print(f"  Created: {datetime.fromtimestamp(run['created_at'])}")
        print(f"  Tags: {json.dumps(run.get('tags', {}), indent=2)}")
        print(f"  Params: {json.dumps(run.get('params', {}), indent=2)}")
        print(f"  Metrics: {json.dumps(run.get('metrics', {}), indent=2)}")
        if run.get("artifacts"):
            print(f"  Artifacts: {json.dumps(run['artifacts'], indent=2)}")

    elif args.command == "model-card":
        card = tracker.create_model_card(
            model_name=args.model_name,
            architecture=args.arch,
            description=args.desc or "",
            base_model=args.base or "",
            dataset=args.dataset or "",
            quant_type=args.quant,
            license=args.license,
        )
        print(f"Created model card: {card['id']}")
        print(f"  Model: {card['model_name']} ({card['architecture']})")

    elif args.command == "model-cards":
        cards = tracker.list_model_cards()
        if not cards:
            print("No model cards.")
            return
        for c in cards:
            print(f"{c['id']:<14} {c['model_name']:<24} {c['architecture']:<12} {c.get('license', 'MIT'):<8}")

    elif args.command == "gen-readme":
        readme = tracker.generate_model_card_readme(args.card_id)
        if readme:
            print(readme)
        else:
            print(f"Model card {args.card_id} not found")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
