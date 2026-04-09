"""A/B experiment management — SQLite-backed experiment store + FastAPI routes.

Enables creating, tracking, and analyzing reward policy experiments without
modifying production grading code. Routing adds <5ms latency per episode
via in-memory caching + consistent hashing.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# ─── Database ────────────────────────────────────────────────────────────────

DB_PATH = os.environ.get(
    "EXPERIMENTS_DB",
    str(Path(__file__).resolve().parent.parent / "experiments.db"),
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS experiments (
    id            TEXT    PRIMARY KEY,
    name          TEXT    NOT NULL,
    policy_type   TEXT    NOT NULL,
    traffic_split REAL    NOT NULL DEFAULT 0.1,
    target_scenarios TEXT  DEFAULT NULL,
    status        TEXT    NOT NULL DEFAULT 'active',
    created_at    TEXT    NOT NULL,
    updated_at    TEXT    NOT NULL,
    metadata      TEXT    DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS episodes (
    id            TEXT    PRIMARY KEY,
    experiment_id TEXT    NOT NULL,
    scenario_id   TEXT    NOT NULL,
    variant       TEXT    NOT NULL,
    step_rewards  TEXT    NOT NULL DEFAULT '[]',
    total_reward  REAL    NOT NULL DEFAULT 0.0,
    policy_type   TEXT    NOT NULL,
    created_at    TEXT    NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

CREATE INDEX IF NOT EXISTS idx_episodes_experiment ON episodes(experiment_id);
CREATE INDEX IF NOT EXISTS idx_episodes_variant ON episodes(variant);
CREATE INDEX IF NOT EXISTS idx_episodes_scenario ON episodes(scenario_id);
"""


@contextmanager
def _get_db(db_path: str = DB_PATH):
    """Thread-safe SQLite connection with WAL mode."""
    conn = sqlite3.connect(db_path, timeout=5)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str = DB_PATH) -> None:
    """Create tables if they don't exist."""
    with _get_db(db_path) as conn:
        conn.executescript(SCHEMA_SQL)


# Auto-init on import
init_db()


# ─── Models ──────────────────────────────────────────────────────────────────

class ExperimentStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABORTED = "aborted"


class CreateExperimentRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Experiment name")
    policy_type: str = Field(..., description="Variant policy: equal, escalation_first, reply_quality")
    traffic_split: float = Field(0.1, ge=0.0, le=1.0, description="Fraction of traffic to variant (0.0-1.0)")
    target_scenarios: Optional[List[str]] = Field(None, description="Limit to specific scenario IDs")


class ExperimentResponse(BaseModel):
    id: str
    name: str
    policy_type: str
    traffic_split: float
    target_scenarios: Optional[List[str]]
    status: str
    created_at: str
    updated_at: str
    metrics: Optional[Dict[str, Any]] = None


class RecordEpisodeRequest(BaseModel):
    experiment_id: str
    scenario_id: str
    variant: str  # "control" or "variant"
    step_rewards: List[float]
    total_reward: float
    policy_type: str


# ─── Experiment Store ────────────────────────────────────────────────────────

class ExperimentStore:
    """SQLite-backed experiment + episode store with in-memory cache.

    The cache avoids DB round-trips on the hot path (variant routing),
    keeping latency < 5ms per episode.
    """

    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path
        self._cache: Dict[str, Dict] = {}
        self._cache_ts: float = 0
        self._cache_ttl: float = 5.0  # seconds
        init_db(db_path)

    def _refresh_cache(self) -> None:
        """Refresh active experiments from DB every cache_ttl seconds."""
        now = time.monotonic()
        if now - self._cache_ts < self._cache_ttl and self._cache:
            return
        with _get_db(self._db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM experiments WHERE status = 'active'"
            ).fetchall()
            self._cache = {r["id"]: dict(r) for r in rows}
            self._cache_ts = now

    def create_experiment(self, req: CreateExperimentRequest) -> ExperimentResponse:
        """Create a new experiment. Max 2 concurrent active experiments."""
        with _get_db(self._db_path) as conn:
            active_count = conn.execute(
                "SELECT COUNT(*) FROM experiments WHERE status = 'active'"
            ).fetchone()[0]
            if active_count >= 2:
                raise HTTPException(
                    status_code=409,
                    detail="Max 2 concurrent experiments. Complete or abort an existing one first.",
                )

            # Validate policy type
            from core.rewards.experimental_policies import POLICY_CLASSES
            if req.policy_type not in POLICY_CLASSES and req.policy_type != "control":
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown policy: {req.policy_type}. Available: {list(POLICY_CLASSES.keys())}",
                )

            exp_id = str(uuid.uuid4())[:8]
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """INSERT INTO experiments (id, name, policy_type, traffic_split,
                   target_scenarios, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, 'active', ?, ?)""",
                (exp_id, req.name, req.policy_type, req.traffic_split,
                 json.dumps(req.target_scenarios) if req.target_scenarios else None,
                 now, now),
            )

        self._cache_ts = 0  # Force cache refresh
        return self.get_experiment(exp_id)

    def get_experiment(self, exp_id: str) -> ExperimentResponse:
        with _get_db(self._db_path) as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE id = ?", (exp_id,)
            ).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"Experiment {exp_id} not found")

            # Compute metrics
            metrics = self._compute_metrics(conn, exp_id)

            return ExperimentResponse(
                id=row["id"],
                name=row["name"],
                policy_type=row["policy_type"],
                traffic_split=row["traffic_split"],
                target_scenarios=(
                    json.loads(row["target_scenarios"])
                    if row["target_scenarios"] else None
                ),
                status=row["status"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                metrics=metrics,
            )

    def list_experiments(
        self, status_filter: Optional[str] = None
    ) -> List[ExperimentResponse]:
        with _get_db(self._db_path) as conn:
            if status_filter:
                rows = conn.execute(
                    "SELECT id FROM experiments WHERE status = ? ORDER BY created_at DESC",
                    (status_filter,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id FROM experiments ORDER BY created_at DESC"
                ).fetchall()
        return [self.get_experiment(r["id"]) for r in rows]

    def update_status(self, exp_id: str, status: ExperimentStatus) -> ExperimentResponse:
        with _get_db(self._db_path) as conn:
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "UPDATE experiments SET status = ?, updated_at = ? WHERE id = ?",
                (status.value, now, exp_id),
            )
        self._cache_ts = 0
        return self.get_experiment(exp_id)

    def record_episode(self, req: RecordEpisodeRequest) -> str:
        """Record an episode result for an experiment."""
        ep_id = str(uuid.uuid4())[:12]
        now = datetime.now(timezone.utc).isoformat()
        with _get_db(self._db_path) as conn:
            conn.execute(
                """INSERT INTO episodes (id, experiment_id, scenario_id, variant,
                   step_rewards, total_reward, policy_type, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (ep_id, req.experiment_id, req.scenario_id, req.variant,
                 json.dumps(req.step_rewards), req.total_reward,
                 req.policy_type, now),
            )
        return ep_id

    def _compute_metrics(self, conn: sqlite3.Connection, exp_id: str) -> Dict[str, Any]:
        """Compute per-variant metrics for an experiment."""
        metrics: Dict[str, Any] = {}
        for variant in ["control", "variant"]:
            rows = conn.execute(
                """SELECT total_reward, step_rewards FROM episodes
                   WHERE experiment_id = ? AND variant = ?""",
                (exp_id, variant),
            ).fetchall()

            if not rows:
                metrics[variant] = {"count": 0}
                continue

            rewards = [r["total_reward"] for r in rows]
            step_data = [json.loads(r["step_rewards"]) for r in rows]

            # Per-step means
            step_means = {}
            for step_idx, step_name in enumerate(["classify", "reply", "escalate"]):
                step_vals = [s[step_idx] for s in step_data if len(s) > step_idx]
                step_means[step_name] = round(sum(step_vals) / max(len(step_vals), 1), 4)

            import statistics
            metrics[variant] = {
                "count": len(rows),
                "mean_reward": round(statistics.mean(rewards), 4),
                "std_reward": round(statistics.stdev(rewards), 4) if len(rewards) > 1 else 0.0,
                "min_reward": round(min(rewards), 4),
                "max_reward": round(max(rewards), 4),
                "step_means": step_means,
            }

        return metrics

    # ── Hot-path: variant routing ────────────────────────────────────────

    def route_episode(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """Determine if episode should use variant policy. O(1) cached lookup.

        Returns None if no experiment matches, or a dict with:
          - experiment_id, policy_type, variant ("control" or "variant")
        """
        self._refresh_cache()

        for exp_id, exp in self._cache.items():
            # Check scenario filter
            targets = exp.get("target_scenarios")
            if targets:
                target_list = json.loads(targets) if isinstance(targets, str) else targets
                if scenario_id not in target_list:
                    continue

            # Consistent hashing: same scenario always goes to same variant
            hash_input = f"{exp_id}:{scenario_id}"
            hash_val = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            bucket = (hash_val % 1000) / 1000.0

            if bucket < exp["traffic_split"]:
                return {
                    "experiment_id": exp_id,
                    "policy_type": exp["policy_type"],
                    "variant": "variant",
                }
            else:
                return {
                    "experiment_id": exp_id,
                    "policy_type": "control",
                    "variant": "control",
                }

        return None


# ─── Singleton ───────────────────────────────────────────────────────────────

_store: Optional[ExperimentStore] = None


def get_store() -> ExperimentStore:
    global _store
    if _store is None:
        _store = ExperimentStore()
    return _store


# ─── FastAPI Router ──────────────────────────────────────────────────────────

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.post("", response_model=ExperimentResponse, status_code=201)
async def create_experiment(req: CreateExperimentRequest):
    """Create a new A/B experiment. Max 2 concurrent experiments."""
    return get_store().create_experiment(req)


@router.get("", response_model=List[ExperimentResponse])
async def list_experiments(status: Optional[str] = Query(None)):
    """List all experiments, optionally filtered by status."""
    return get_store().list_experiments(status)


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(experiment_id: str):
    """Get experiment details with live metrics."""
    return get_store().get_experiment(experiment_id)


@router.post("/{experiment_id}/status")
async def update_experiment_status(experiment_id: str, status: ExperimentStatus):
    """Update experiment status (pause, complete, abort)."""
    return get_store().update_status(experiment_id, status)


@router.post("/{experiment_id}/episodes")
async def record_episode(experiment_id: str, req: RecordEpisodeRequest):
    """Record an episode result for an experiment."""
    req.experiment_id = experiment_id
    ep_id = get_store().record_episode(req)
    return {"episode_id": ep_id, "recorded": True}


@router.get("/{experiment_id}/route")
async def route_check(experiment_id: str, scenario_id: str):
    """Check which variant a scenario would be routed to."""
    result = get_store().route_episode(scenario_id)
    if result and result["experiment_id"] == experiment_id:
        return result
    return {"variant": "control", "policy_type": "control"}
