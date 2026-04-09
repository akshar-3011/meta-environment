# Architecture

> System design and component responsibilities for meta-environment.

---

## System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        Client Layer                              │
│  RL Agent (SB3/Custom) ──── HTTP Client ──── Gym Wrapper         │
└──────────────────────────────┬───────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────┐
│                       API Layer (FastAPI)                         │
│                                                                  │
│  ┌──────────────────── Middleware Stack ────────────────────────┐ │
│  │ API Key Auth → CORS → Rate Limit → Size Limit → Error Sani │ │
│  │ → Request ID → Security Headers → Prometheus Metrics        │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌────────────┐   │
│  │ /reset   │  │ /step    │  │ /experiments │  │ /metrics   │   │
│  │ /state   │  │ /health  │  │ (CRUD + route)│  │ /docs      │   │
│  └────┬─────┘  └────┬─────┘  └──────┬───────┘  └────────────┘   │
└───────┼──────────────┼───────────────┼───────────────────────────┘
        │              │               │
┌───────▼──────────────▼───────────────▼───────────────────────────┐
│                    Environment Layer                              │
│                                                                  │
│  WorkplaceEnvironment                                            │
│  ├── reset() → select scenario → return observation              │
│  ├── step()  → validate action → grade → update state → return   │
│  └── state   → episode metadata                                  │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐   │
│  │ ScenarioRepository│  │  RewardPolicy    │  │ GymWrapper    │   │
│  │ (100 scenarios)  │  │ (configurable)   │  │ (SB3-ready)   │   │
│  └──────────────────┘  └──────────────────┘  └───────────────┘   │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                     Grading Layer                                 │
│                                                                  │
│  WeightedParallelGradingEngine                                   │
│  ├── ClassificationGrader (exact + adjacent match)               │
│  ├── ReplyQualityGrader  (length, keywords, empathy, solution)   │
│  └── EscalationGrader    (correctness + trajectory bonus)        │
│                                                                  │
│  Experimental Policies:                                          │
│  ├── EqualWeightPolicy        (33/33/33)                         │
│  ├── EscalationFirstPolicy    (25/25/50)                         │
│  └── ReplyQualityPolicy       (30/50/20)                         │
└──────────────────────────────────────────────────────────────────┘
```

---

## Component Responsibilities

### 1. API Layer (`api/`)

| File | Responsibility |
|---|---|
| `app.py` | FastAPI wiring, OpenEnv `create_app`, root HTML page |
| `middleware.py` | 9-layer security middleware stack |
| `metrics.py` | Prometheus counter/histogram registration |
| `tracing.py` | OpenTelemetry auto-instrumentation |
| `experiments.py` | A/B experiment CRUD + routing + SQLite store |

### 2. Environment Layer (`environment/`)

| File | Responsibility |
|---|---|
| `workplace_environment.py` | Core `reset()`/`step()` logic, scenario cycling, metrics |
| `gym_wrapper.py` | Gymnasium `Env` wrapper for SB3 training |

### 3. Grading Layer (`core/graders/`)

| File | Responsibility |
|---|---|
| `interfaces.py` | `RewardPolicy` protocol, `EvaluationContext` dataclass |
| `framework.py` | `WeightedParallelGradingEngine` — sequential weighted aggregation |
| `rule_based.py` | `RuleBasedRewardPolicy` — production grading (0.40/0.35/0.25) |

### 4. Configuration (`core/`)

| File | Responsibility |
|---|---|
| `config.py` | Centralized config from env vars + `.env` file |
| `logging_config.py` | JSON/text structured logging setup |
| `models/` | Pydantic models: `Scenario`, `WorkplaceAction`, `WorkplaceObservation` |

### 5. Security (`security/`)

| File | Responsibility |
|---|---|
| `rate_limit_strict.py` | Per-endpoint + global sliding window rate limiter |
| `audit_logging.py` | SIEM-compatible JSON audit trail |

### 6. Data (`data/`)

| File | Responsibility |
|---|---|
| `scenario_repository.py` | 100 validated scenarios (33 easy, 34 medium, 33 hard) |

---

## Key Design Decisions

### 1. Sequential Grading (not Parallel)
The `WeightedParallelGradingEngine` executes graders sequentially despite the name. This avoids GIL overhead for CPU-bound grading and ensures deterministic reward ordering.

### 2. Dense Rewards
Every step returns an immediate reward (not just end-of-episode). This accelerates training convergence, especially for PPO which benefits from per-step feedback.

### 3. Immutable Models
All Pydantic models use `frozen=True` to prevent accidental state mutation. Scenarios, observations, and actions are all immutable after creation.

### 4. Configurable Reward Weights
The A/B experiment framework allows testing alternative reward weightings without changing the core grading logic. All experimental policies share the same graders.

### 5. In-Memory Experiment Routing
Experiment routing uses an in-memory cache (5s TTL) to avoid SQLite queries on the hot path. Consistent hashing ensures deterministic variant assignment.

---

## Data Flow: Episode Lifecycle

```
1. POST /reset
   └→ WorkplaceEnvironment.reset()
      ├→ Select scenario from repository (round-robin)
      ├→ Initialize episode state (step_count=0, rewards=[])
      └→ Return initial observation (email, categories, metadata)

2. POST /step {action_type: "classify", content: "refund"}
   └→ WorkplaceEnvironment.step()
      ├→ Validate action (Pydantic model)
      ├→ Route to experiment variant (if active)
      ├→ Grade action via RewardPolicy.calculate_step_reward()
      │   ├→ ClassificationGrader.evaluate()
      │   ├→ Apply weight (0.40)
      │   └→ Return weighted score + breakdown
      ├→ Record metrics (Prometheus, audit log)
      └→ Return observation (email, reward, done=False)

3. POST /step {action_type: "reply", content: "Thank you..."}
   └→ Same flow, ReplyQualityGrader, weight 0.35

4. POST /step {action_type: "escalate", content: "no"}
   └→ Same flow, EscalationGrader, weight 0.25, done=True
```
