# Experimentation Framework

> A/B test reward function variants in production without modifying the core grading pipeline.

---

## Overview

The experimentation framework enables testing alternative reward weightings against the production baseline. Each variant differs **only** in how per-step rewards are weighted — the underlying grading logic (accuracy, semantic similarity, rule-based evaluation) remains identical.

### Available Policies

| Policy | Classify | Reply | Escalate | Use Case |
|---|---|---|---|---|
| `control` | 40% | 35% | 25% | Production baseline |
| `equal` | 33% | 33% | 33% | Unbiased skill assessment |
| `escalation_first` | 25% | 25% | 50% | Safety-critical environments |
| `reply_quality` | 30% | 50% | 20% | Customer satisfaction focus |

---

## Quick Start

### 1. Create an Experiment

```bash
curl -X POST http://localhost:8000/experiments \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test-escalation-50pct",
    "policy_type": "escalation_first",
    "traffic_split": 0.2
  }'
```

Response:
```json
{
  "id": "a1b2c3d4",
  "name": "test-escalation-50pct",
  "policy_type": "escalation_first",
  "traffic_split": 0.2,
  "status": "active",
  "metrics": null
}
```

### 2. Monitor In-Flight

```bash
# Check metrics via API
curl http://localhost:8000/experiments/a1b2c3d4

# Check routing for a specific scenario
curl "http://localhost:8000/experiments/a1b2c3d4/route?scenario_id=E1"
```

### 3. Analyze Results

```bash
# Full analysis with recommendation
python tools/analyze_experiment.py a1b2c3d4

# Output as JSON
python tools/analyze_experiment.py a1b2c3d4 --json

# List all experiments
python tools/analyze_experiment.py --list
```

### 4. Act on Recommendation

```bash
# Deploy variant (if recommended)
curl -X POST "http://localhost:8000/experiments/a1b2c3d4/status?status=completed"

# Abort experiment
curl -X POST "http://localhost:8000/experiments/a1b2c3d4/status?status=aborted"
```

---

## Architecture

```
Episode Request
       │
       ▼
┌──────────────────┐
│  ExperimentStore  │ ── cached active experiments (refreshed every 5s)
│  .route_episode() │
└──────┬───────────┘
       │ consistent hash(experiment_id:scenario_id)
       ▼
  bucket < split? ──── YES ──→ Variant policy (e.g., EscalationFirstPolicy)
       │                                    │
       NO                                   │
       │                                    │
       ▼                                    ▼
  Control policy ──────────────────→ Record episode to experiments.db
  (RuleBasedRewardPolicy)           with variant label + step rewards
```

### Routing Guarantees

- **Deterministic**: Same `(experiment_id, scenario_id)` always routes to the same variant
- **Low latency**: In-memory cache, <5ms overhead per episode
- **No cross-contamination**: Scenarios are assigned to exactly one variant

### Database Schema

```sql
-- experiments.db (SQLite with WAL mode)

experiments:
  id            TEXT PRIMARY KEY
  name          TEXT NOT NULL
  policy_type   TEXT NOT NULL        -- "equal", "escalation_first", "reply_quality"
  traffic_split REAL DEFAULT 0.1     -- fraction routed to variant
  target_scenarios TEXT              -- JSON array or NULL (all scenarios)
  status        TEXT DEFAULT 'active' -- active, paused, completed, aborted
  created_at    TEXT NOT NULL
  updated_at    TEXT NOT NULL

episodes:
  id            TEXT PRIMARY KEY
  experiment_id TEXT REFERENCES experiments(id)
  scenario_id   TEXT NOT NULL
  variant       TEXT NOT NULL         -- "control" or "variant"
  step_rewards  TEXT NOT NULL          -- JSON array: [classify, reply, escalate]
  total_reward  REAL NOT NULL
  policy_type   TEXT NOT NULL
  created_at    TEXT NOT NULL
```

---

## Guardrails

| Rule | Value | Rationale |
|---|---|---|
| Max concurrent experiments | **2** | Prevent combinatorial explosion |
| Min episodes per variant | **100** | Statistical power for t-test |
| Significance level | **p < 0.05** | Standard threshold |
| Max traffic split | **50%** | Protect production quality |

---

## Analysis Output

```
═══════════════════════════════════════════════════════════════════════
  EXPERIMENT ANALYSIS: test-escalation-50pct
  ID: a1b2c3d4  |  Policy: escalation_first
═══════════════════════════════════════════════════════════════════════

  Metric                    │      Control │      Variant │       Diff
  ──────────────────────────┼──────────────┼──────────────┼───────────
  Episodes                  │          500 │          125 │       -375
  Mean Reward               │       0.6800 │       0.7200 │    +0.0400
  Escalation Acc (%)        │         65.0 │         82.0 │      +17.0

  Statistics:
    t-statistic:    3.2100
    p-value:        0.0014  ✅ significant
    Lift:           +5.88%

  ═══════════════════════════════════════════════════════════════
  🚀 RECOMMENDATION: DEPLOY_VARIANT
     Variant is significantly better: +5.9% lift, p=0.0014 < 0.05
  ═══════════════════════════════════════════════════════════════
```

### Recommendation Logic

| Condition | Recommendation |
|---|---|
| < 100 episodes per variant | ⏳ `CONTINUE_TEST` |
| p < 0.05, positive lift | 🚀 `DEPLOY_VARIANT` |
| p < 0.05, negative lift | 🛑 `ABORT` |
| p ≥ 0.05 | ⏳ `CONTINUE_TEST` |

---

## Extending Policies

To add a new reward weighting:

```python
# core/rewards/experimental_policies.py

POLICY_WEIGHTS["my_custom"] = PolicyWeights(0.20, 0.60, 0.20)

class MyCustomPolicy(ConfigurableRewardPolicy):
    def __init__(self, **kwargs):
        super().__init__(POLICY_WEIGHTS["my_custom"], **kwargs)

POLICY_CLASSES["my_custom"] = MyCustomPolicy
```

Then create an experiment:
```bash
curl -X POST http://localhost:8000/experiments \
  -d '{"name": "custom-test", "policy_type": "my_custom", "traffic_split": 0.1}'
```
