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

## Overview

Customer support is one of the most universally high-stakes human tasks: a single poorly-handled email can cost a business a customer, while a well-triage and empathetic response builds loyalty. This environment formalises that task as an RL training ground.

An agent receives an inbound customer email and must execute a three-step workflow:

1. **Classify** — identify the ticket type (`refund` / `complaint` / `query`)
2. **Reply** — draft a helpful, empathetic response meeting quality criteria
3. **Escalate** — decide whether to pass the case to a senior team

Each step is graded immediately (dense reward), so the environment never gives only sparse end-of-episode signal. Difficulty is controlled via scenario complexity, sentiment, and multi-intent ambiguity.

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

Each task is one episode: `classify → reply → escalate`. Task names match the primary email category the episode is drawn from.

### `refund` — Easy
**Scenario:** Clear refund requests with neutral-to-positive sentiment.  
**Challenge:** Identify the correct category, write a concise resolution reply, and correctly decide not to escalate.  
**Grader:** Classification accuracy (40%), reply quality/keywords (35%), escalation correctness (25%).

### `complaint` — Medium
**Scenario:** Negative-sentiment complaints including damaged items, double charges, unresponsive support.  
**Challenge:** Distinguish complaints from refund requests, write empathetic replies, and correctly escalate when the case warrants senior review.  
**Grader:** Same rubric, but correct escalation (`yes`) is required and more heavily penalised if missed.

### `query` — Hard
**Scenario:** Ambiguous information requests — often phrased with frustration, sarcasm, or mixed signals (e.g. query + latent complaint).  
**Challenge:** Resist misclassifying frustrated queries as complaints, write informative replies, and correctly hold escalation.  
**Grader:** Same rubric; SemanticSimilarity component matters more because keyword matching is harder for query replies.

---

## Reward Function

Rewards are **dense** — every step returns a score in `[0.0, 1.0]` with a weight applied before summing.

| Step | Weight | Signal |
|---|---|---|
| `classify` | 0.40 | Exact match → 1.0; related label → 0.2–0.4; wrong → 0.0 |
| `reply` | 0.35 | Length OK (+0.35), concise (+0.15), keywords matched (+0.15×n), solution-oriented (+0.10); penalised for missing keywords (−0.20) or harsh tone (−0.15) |
| `escalate` | 0.25 | Correct decision → 0.9–1.0; wrong direction → 0.1–0.3; early escalation is penalised (×0.7) |

**Max episode reward:** ≈ 1.0 (sum of weighted step scores, clamped per step).

The `reply` step additionally applies a **consistency penalty** (−0.20 × 0.35) when the classification score was below 0.5, creating a trajectory coupling signal that rewards coherent multi-step reasoning.

---

## Baseline Scores

Scores below are from the deterministic mock agent (no LLM, rule-based responses). They serve as a reproducible lower bound. A frontier LLM agent is expected to score significantly higher, especially on `complaint` and `query`.

| Task | Difficulty | Classify | Reply | Escalate | **Total** |
|---|---|---|---|---|---|
| `refund` | easy | 0.40 | 0.08 | 0.23 | **0.71** |
| `complaint` | medium | 0.14 | 0.17 | 0.02 | **0.33** |
| `query` | hard | 0.00 | 0.08 | 0.23 | **0.31** |

> Scores are reproducible: `python inference.py 2>/dev/null` (no `HF_TOKEN` required).

---

## Setup & Usage

### Prerequisites

- Python ≥ 3.10
- `uv` (recommended) or `pip`
- Docker (for containerised deployment)

### Local Installation

```bash
git clone https://github.com/akshar-3011/meta-environment.git
cd meta-environment

# Create virtual environment and install
uv sync
# or:  pip install -e ".[dev]"
```

### Run the FastAPI Server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
# Health check:
curl http://localhost:8000/health
```

### Run Inference (in-process, no server needed)

```bash
# Mock agent (no API key required)
python inference.py

# Real LLM via HuggingFace router
HF_TOKEN=hf_xxx \
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
python inference.py
```

### Validate OpenEnv Compliance

```bash
pip install openenv-core
openenv validate
```

### Docker

```bash
docker build -t workplace-env .
docker run -p 8000:8000 workplace-env
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/reset` | Start a new episode → returns initial observation |
| `POST` | `/step` | Submit one action → returns observation + reward + done |
| `GET` | `/state` | Current episode state (step count, rewards, history) |
| `GET` | `/schema` | Action and observation JSON schemas |

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
| `API_SERVER_PORT` | No | `8000` | HTTP server port |
| `ENV_DEBUG` | No | `false` | Verbose episode logging |

---

## Project Structure

```
workplace_env/
├── inference.py              # Baseline inference script (OpenEnv spec)
├── openenv.yaml              # OpenEnv environment manifest
├── Dockerfile                # Container build file
├── pyproject.toml            # Package metadata & dependencies
├── requirements.txt          # Pinned dependencies
├── environment/
│   └── workplace_environment.py   # Core WorkplaceEnvironment class
├── core/
│   ├── config.py             # Centralised configuration
│   ├── graders/              # Modular reward/grader pipeline
│   └── models/               # Pydantic data models
├── server/
│   └── app.py                # FastAPI app entry point
├── api/
│   └── app.py                # OpenEnv create_app wiring
├── data/
│   └── scenario_repository.py     # Scenario loader
└── data.py                   # 30+ annotated customer support scenarios
```

---

## License

BSD-style — see [LICENSE](LICENSE) for details.
