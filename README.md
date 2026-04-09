---
title: Workplace Env Environment Server
emoji: 🏢
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
---

# 🏢 Workplace Customer Support Environment

> An OpenEnv-compliant reinforcement learning environment for training and evaluating AI agents on real-world customer support workflows.

[![CI](https://github.com/akshar-3011/meta-environment/actions/workflows/ci.yml/badge.svg)](https://github.com/akshar-3011/meta-environment/actions/workflows/ci.yml)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-brightgreen)](https://github.com/meta-pytorch/OpenEnv)
[![Tests](https://img.shields.io/badge/tests-68%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-≥3.10-blue)]()

---

## Overview

Customer support is one of the most universally high-stakes human tasks: a single poorly-handled email can cost a business a customer, while a well-triaged and empathetic response builds loyalty. This environment formalises that task as an RL training ground.

An agent receives an inbound customer email and must execute a three-step workflow:

1. **Classify** — identify the ticket type (`refund` / `complaint` / `query`)
2. **Reply** — draft a helpful, empathetic response meeting quality criteria
3. **Escalate** — decide whether to pass the case to a senior team

Each step is graded immediately (dense reward), so the environment never gives only sparse end-of-episode signal. Difficulty is controlled via scenario complexity, sentiment, and multi-intent ambiguity.

---

## ✅ What's Fixed (v1.0.0)

This release includes a comprehensive production audit:

- **10 critical bugs fixed (C1–C10):** Environment fallback, dead escalation timing, thread pool crashes, retry loops, consistency penalty scaling, route conflicts, Docker healthcheck
- **10 high-impact refactors (N1–N10):** Sequential grading (+30% throughput), Pydantic validation at load, frozen models, Literal action types, JSON structured logging, reloadable config, real /infer scores
- **68 tests** across 10 test files with CI/CD pipeline
- **Production middleware:** API key auth, CORS allowlist, per-IP rate limiting, Prometheus metrics, OpenTelemetry tracing
- **Performance:** P50=0.3ms, P99=0.4ms, 3,022 episodes/sec, 0.1MB memory per episode

---

## Action Space

Actions are submitted as a JSON object with two required fields:

| Field | Type | Values | Description |
|---|---|---|---|
| `action_type` | `str` | `"classify"`, `"reply"`, `"escalate"` | Which step to execute |
| `content` | `str` | free text / one-word response | The agent's response |

### Examples

```json
{"action_type": "classify", "content": "refund"}
{"action_type": "reply",    "content": "Thank you for reaching out. We have processed your refund and it will appear within 3–5 business days."}
{"action_type": "escalate", "content": "no"}
```

---

## Observation Space

All fields are returned after every `reset()` and `step()` call:

| Field | Type | Description |
|---|---|---|
| `email` | `str` | The inbound customer email text |
| `category_options` | `List[str]` | Valid classification labels: `["refund", "complaint", "query"]` |
| `history` | `List[str]` | Actions taken so far in the current episode |
| `reward` | `Optional[float]` | Reward from the most recent step (`null` after reset) |
| `done` | `bool` | Whether the episode has ended |
| `scenario_difficulty` | `Optional[str]` | `"easy"` / `"medium"` / `"hard"` |
| `urgency` | `Optional[str]` | `"low"` / `"medium"` / `"high"` |
| `sentiment` | `Optional[str]` | `"positive"` / `"neutral"` / `"negative"` / `"mixed"` |
| `complexity_score` | `Optional[int]` | 1–5 scale (higher = more ambiguous) |
| `scenario_metadata` | `Optional[dict]` | `{"min_reply_length": int}` |

---

## Tasks

Each task is one episode: `classify → reply → escalate`. Tasks are organized by difficulty level with scenarios drawn from all categories.

### `easy-triage` — Easy
**Scenario:** Clear single-intent emails with neutral sentiment (simple refund requests, basic delivery queries).  
**Challenge:** Identify the correct category, write a concise reply, and correctly decide escalation.  
**Grader:** Classification accuracy (40%), reply quality (35%), escalation correctness (25%).

### `medium-triage` — Medium
**Scenario:** Mixed-sentiment emails with some ambiguity — frustrated but polite refund requests, complaints that overlap with queries, formal language variations.  
**Challenge:** Correctly distinguish intent despite emotional signals, write empathetic replies, and make nuanced escalation decisions.  
**Grader:** Same rubric with difficulty multiplier (×1.05) rewarding good performance on harder cases.

### `hard-triage` — Hard
**Scenario:** Adversarial emails with sarcasm, multi-intent, competitor threats, GDPR questions, loyalty signals, retaliation accusations, misdelivery urgency — designed to fool keyword heuristics.  
**Challenge:** Requires nuanced understanding of customer intent behind surface language. 15 scenarios covering edge cases that challenge frontier LLMs.  
**Grader:** Same rubric with difficulty multiplier (×1.12), adjacent-label partial credit for ambiguous cases, and trajectory consistency bonus.

---

## Reward Function

Rewards are **dense** — every step returns a score in `[0.0, 1.0]` with a weight applied before summing.

| Step | Weight | Signal |
|---|---|---|
| `classify` | 0.40 | Exact match → 1.0; related label → 0.2–0.4; wrong → 0.0; hard adjacent partial credit → 0.25 |
| `reply` | 0.35 | Length (0.0–0.40 scaled), keywords (0.05×n, max 0.45), concise (+0.10/−0.05), solution (+0.10), greeting (+0.08), closing (+0.07), empathy (+0.05 for complaints), difficulty multiplier |
| `escalate` | 0.25 | Correct decision → 0.9–1.0; wrong direction → 0.1–0.3; early escalation (×0.7); trajectory bonus (+0.05 when prior steps high-quality); trajectory penalty (−0.03 for incorrect over-escalation) |

**Max episode reward:** ≈ 0.99 (sum of weighted step scores with bonuses, clamped per step).

The `reply` step additionally applies a **consistency penalty** (scaled linearly) when the classification score was below 0.5, creating a trajectory coupling signal that rewards coherent multi-step reasoning.

**Difficulty-adaptive grading:** Hard scenarios receive a ×1.12 multiplier on reply scores, medium ×1.05, rewarding agents that perform well on more challenging cases.

---

## Baseline Scores

Scores below are from the deterministic mock agent (no LLM, `EmailAwareInference` heuristic). They serve as a reproducible lower bound.

| Task | Difficulty | Classify | Reply | Escalate | **Total** |
|---|---|---|---|---|---|
| `easy-triage` | easy | 0.40 | 0.20 | 0.23 | **0.83** |
| `medium-triage` | medium | 0.40 | 0.21 | 0.25 | **0.86** |
| `hard-triage` | hard | 0.00 | 0.06 | 0.23 | **0.29** |

> Scores are reproducible: `python inference.py 2>/dev/null` (no `HF_TOKEN` required).

---

## 🚀 Quick Start

### Docker (recommended)

```bash
docker build -t workplace-env .
docker run -p 8000:8000 workplace-env
curl http://localhost:8000/health
```

### Local Installation

```bash
git clone https://github.com/akshar-3011/meta-environment.git
cd meta-environment

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e ".[dev,observability]"
```

### Run the FastAPI Server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/health
```

### Run Inference (in-process, no server needed)

```bash
# Mock agent (no API key required)
python inference.py

# Real LLM via HuggingFace router
HF_TOKEN=hf_xxx MODEL_NAME=Qwen/Qwen2.5-72B-Instruct python inference.py
```

### Validate OpenEnv Compliance

```bash
openenv validate
# [OK] workplace: Ready for multi-mode deployment
```

---

## 🧪 Training an RL Agent

A Gymnasium wrapper and PPO training script are included:

```bash
# Train on easy scenarios (converges in ~10k steps):
python examples/train_ppo.py --difficulty easy --timesteps 50000

# Train on all difficulties with TensorBoard:
python examples/train_ppo.py --timesteps 100000 --tb-log ./logs/
tensorboard --logdir ./logs/

# Resume from checkpoint:
python examples/train_ppo.py --resume models/ppo_workplace_easy/best_model.zip
```

### Training Results (PPO, 10k steps)

| Difficulty | Mean Reward | Classify Accuracy | Escalation Accuracy |
|---|---|---|---|
| Easy | 0.986 ± 0.009 | 100% | 100% |
| Medium | 0.986 ± 0.009 | 100% | 100% |
| Hard | 0.986 ± 0.008 | 100% | 100% |

---

## 📊 Performance Benchmarks

```bash
python benchmarks/load_test.py --mode direct --episodes 500
```

| Metric | Target | Actual |
|---|---|---|
| P50 latency | < 200ms | **0.3ms** |
| P99 latency | < 500ms | **0.4ms** |
| Throughput | > 100 eps/s | **3,022 eps/s** |
| Memory | < 50MB | **0.1MB** |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/reset` | Start a new episode → returns initial observation |
| `POST` | `/step` | Submit one action → returns observation + reward + done |
| `GET` | `/state` | Current episode state (step count, rewards, history) |
| `GET` | `/metrics` | Prometheus metrics (request count, latency, rewards) |
| `GET` | `/docs` | Interactive Swagger API docs |

### Example Session

```bash
# Reset
curl -s -X POST http://localhost:8000/reset | python3 -m json.tool

# Classify
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "classify", "content": "refund"}}'

# Reply
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "reply", "content": "We have processed your refund — expect it in 3–5 business days."}}'

# Escalate
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "escalate", "content": "no"}}'
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | For LLM | — | HuggingFace API key |
| `OPENAI_API_KEY` | Alias for HF_TOKEN | — | Standard OpenAI key alias |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `APP_ENV` | No | `development` | `production` enables JSON logging |
| `API_KEY` | No | — | API key for auth (empty = disabled) |
| `CORS_ORIGINS` | No | — | Comma-separated allowed origins |
| `RATE_LIMIT_PER_MINUTE` | No | `100` | Per-IP rate limit (0 = disabled) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | No | — | OpenTelemetry trace exporter |

---

## Project Structure

```
workplace_env/
├── inference.py                 # Baseline inference script (OpenEnv spec)
├── openenv.yaml                 # OpenEnv environment manifest
├── Dockerfile                   # Multi-stage container build
├── pyproject.toml               # Package metadata & dependencies
├── environment/
│   ├── workplace_environment.py # Core WorkplaceEnvironment class
│   └── gym_wrapper.py           # Gymnasium wrapper for SB3 training
├── core/
│   ├── config.py                # Centralised configuration
│   ├── logging_config.py        # JSON/text structured logging
│   ├── graders/                 # Modular reward/grader pipeline
│   │   ├── framework.py         # Grading engine
│   │   └── rule_based.py        # Rule-based reward policy
│   └── models/                  # Pydantic data models (frozen)
├── api/
│   ├── app.py                   # OpenEnv create_app wiring
│   ├── middleware.py            # Auth, CORS, rate limiting, metrics
│   ├── metrics.py               # Prometheus counters & histograms
│   ├── tracing.py               # OpenTelemetry distributed tracing
│   └── pipeline_app.py          # /infer, /grade, /pipeline endpoints
├── server/
│   └── app.py                   # FastAPI entry point
├── data/
│   └── scenario_repository.py   # 39 validated scenarios (11E/13M/15H)
├── examples/
│   └── train_ppo.py             # PPO training with SB3
├── benchmarks/
│   └── load_test.py             # Performance benchmark suite
├── tests/                       # 68 tests across 10 files
├── prometheus/
│   └── alerts.yml               # Alerting rules
├── scripts/
│   └── rollback.sh              # Deployment rollback script
└── .github/workflows/
    └── ci.yml                   # CI/CD pipeline
```

---

## License

BSD-style — see [LICENSE](LICENSE) for details.
