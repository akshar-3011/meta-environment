---
title: Workplace Env Environment Server
emoji: 🏢
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
---

# 🏢 Workplace Environment — OpenEnv Customer Support RL Environment

A **production-grade reinforcement learning environment** that simulates real-world automated customer support workflows. AI agents learn to triage, respond to, and escalate customer emails across a spectrum of difficulty levels — mirroring tasks performed daily by support operations teams at scale.

---

## 📋 Environment Description & Motivation

Customer support is one of the highest-volume, highest-stakes text tasks humans perform at work. Every support team faces the same challenge: given an incoming email, an agent must:

1. **Classify** the intent (refund request, complaint, general query)
2. **Reply** with an empathetic, accurate, policy-compliant response
3. **Escalate** (or not) to a senior agent based on urgency and complexity

This environment trains and evaluates AI agents on exactly this multi-step workflow. Unlike toy classification benchmarks, the agent must reason across a full episode — its reply quality affects escalation reward, and its classification affects reply coherence scoring. The reward function is shaped to reward **partial progress**, not just end-state correctness.

**Why this matters:** Enterprises spend billions on customer support. Agents that can learn this task end-to-end have immediate economic value and represent a genuine unsolved problem in applied RL.

---

## 🎬 Action Space

Each step, the agent submits an action composed of:

| Field         | Type   | Description                                                                     |
|---------------|--------|---------------------------------------------------------------------------------|
| `action_type` | `str`  | One of `"classify"`, `"reply"`, `"escalate"`                                    |
| `content`     | `str`  | The action payload — category label, reply text, or `"yes"` / `"no"`           |
| `confidence`  | `float`| Optional confidence score (0.0–1.0); not required for scoring                  |
| `explanation` | `str`  | Optional reasoning; not required for scoring                                    |

### Valid Action Sequence

A single episode follows a strict 3-step protocol:

```
Step 1: classify  → content ∈ {"refund", "complaint", "query"}
Step 2: reply     → content = free-form text reply to the customer (≥30 chars recommended)
Step 3: escalate  → content ∈ {"yes", "no"}
```

An invalid `action_type` returns `done=True` with `reward=0.0`.

---

## 👁️ Observation Space

After each `reset()` or `step()`, the environment returns a `WorkplaceObservation`:

| Field                | Type             | Description                                                |
|----------------------|------------------|------------------------------------------------------------|
| `email`              | `str`            | The customer email to process                              |
| `category_options`   | `List[str]`      | Valid classification labels: `["refund","complaint","query"]` |
| `history`            | `List[str]`      | Log of previous actions taken this episode                |
| `reward`             | `Optional[float]`| Reward from the most recent step (`None` after reset)     |
| `done`               | `bool`           | Whether the episode has ended                             |
| `scenario_difficulty`| `Optional[str]`  | `"easy"`, `"medium"`, or `"hard"`                        |
| `urgency`            | `Optional[str]`  | `"low"`, `"medium"`, or `"high"`                         |
| `sentiment`          | `Optional[str]`  | Customer sentiment: `"negative"`, `"neutral"`, `"positive"`, `"mixed"` |
| `complexity_score`   | `Optional[int]`  | 1–5 scale of scenario complexity                          |
| `scenario_metadata`  | `Optional[dict]` | Extra metadata (e.g. `min_reply_length`)                 |

---

## 🎯 Tasks & Difficulty Levels

Three tasks are defined in `openenv.yaml`, each with three difficulty levels:

### Task 1: `refund` (Easy)
- **Objective:** Correctly classify a refund request, write a professional acknowledgment, and decide whether escalation is needed.
- **Scenario:** Clear, unambiguous refund requests (e.g. *"I want a refund for my order"*)
- **Expected baseline score:** ~0.4–0.7 (a competent model should classify correctly and write a decent reply)
- **Grader focus:** Classification accuracy (40%), reply quality (35%), escalation accuracy (25%)

### Task 2: `complaint` (Medium)
- **Objective:** Handle emotionally charged complaints with empathy, resolve ambiguity about intent, and correctly escalate when required.
- **Scenario:** Negative-sentiment complaints, often requiring escalation (e.g. *"Your service is terrible, I'm very unhappy"*, *"I was charged twice!"*)
- **Expected baseline score:** ~0.25–0.5 (escalation decisions are harder; sentiment management matters)
- **Grader focus:** Empathy keywords in reply, correct escalation decision, classification under ambiguity

### Task 3: `query` (Hard)
- **Objective:** Handle edge-case queries that mix frustration, implicit complaints, and genuine questions. Correctly classify despite misleading surface signals.
- **Scenario:** Hard edge cases — sarcasm, passive-aggressive phrasing, multi-issue overload (e.g. *"I was just wondering... is it normal for orders to take three weeks?"*)
- **Expected baseline score:** ~0.1–0.35 (frontier models can fail on classification; reply keywords differ from intent)
- **Grader focus:** Correct classification under adversarial signals, appropriate (non-)escalation, reply coherence

---

## 🏆 Reward Function

Rewards are shaped to provide **dense, partial progress signals** throughout the trajectory — not just a binary end-of-episode score.

### Per-Step Weights

| Step       | Weight | Scoring Components                                                         |
|------------|--------|----------------------------------------------------------------------------|
| `classify` | 40%    | Exact match → 1.0 · 0.4; related label → 0.2–0.4 · 0.4; wrong → 0       |
| `reply`    | 35%    | Length ≥ min → +0.35; concise < 500 chars → +0.15; keyword hits → +0.15 each (max 0.4); solution-oriented words → +0.1; harsh phrases → −0.15 |
| `escalate` | 25%    | Correct decision → 0.9–1.0 · 0.25; wrong direction → 0.1–0.3 · 0.25     |

### Episode Total

```
total_reward = classify_reward + reply_reward + escalate_reward  ∈ [0.0, 1.0]
```

**Partial progress:** Even a wrong classification earns partial credit if it's a semantically related label. A too-short reply earns partial credit. An unnecessary escalation earns 0.3× rather than 0.

**Trajectory dependency:** The reply grader applies a **consistency penalty** if the classify step was wrong (penalizes 0.2 from reply score) — encouraging accurate classification as a prerequisite to good replies.

---

## 🔌 OpenEnv Interface

```python
from environment.workplace_environment import WorkplaceEnvironment
from models import WorkplaceAction

env = WorkplaceEnvironment()

# Start a new episode
obs = env.reset()
print(obs.email)           # customer email
print(obs.category_options)  # ["refund", "complaint", "query"]

# Step 1: classify
obs = env.step(WorkplaceAction(action_type="classify", content="refund"))
print(obs.reward)   # float in [0.0, 0.4]

# Step 2: reply
obs = env.step(WorkplaceAction(action_type="reply", content="Thank you for reaching out..."))
print(obs.reward)   # float in [0.0, 0.35]

# Step 3: escalate
obs = env.step(WorkplaceAction(action_type="escalate", content="no"))
print(obs.reward)   # float in [0.0, 0.25]
print(obs.done)     # True — episode complete

# Current state (property, not method call)
state = env.state   # dict with episode_count, step_count, cumulative_reward, ...
```

---

## ⚙️ Setup & Usage

### Prerequisites

- Python 3.10+
- `pip install openenv-core>=0.2.2 openai>=1.0.0`

### Local Installation

```bash
git clone https://github.com/akshar-3011/meta-environment
cd meta-environment

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
# or
pip install -r requirements.txt
```

### Run the Server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
# Health check:
curl http://localhost:8000/health
```

### Run Inference (In-Process, No Server Required)

```bash
# Offline / mock mode (no API key needed)
python inference.py

# Live mode with a real model
export HF_TOKEN=hf_your_token
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export API_BASE_URL="https://router.huggingface.co/v1"
python inference.py
```

### Validate OpenEnv Compliance

```bash
openenv validate
# Expected: [OK] workplace: Ready for multi-mode deployment
```

### Docker

```bash
docker build -t workplace-env .
docker run -p 8000:8000 workplace-env
```

---

## 📊 Baseline Scores

Scores produced by the **deterministic mock agent** (no LLM, rule-based responses):

| Task      | Difficulty | Classify | Reply | Escalate | Total |
|-----------|------------|----------|-------|----------|-------|
| refund    | easy       | 0.40     | 0.12  | 0.23     | 0.71  |
| complaint | medium     | 0.14     | 0.17  | 0.02     | 0.33  |
| query     | hard       | 0.09     | 0.11  | 0.02     | 0.21  |

> Scores with a real frontier model (Qwen2.5-72B) will be substantially higher, especially on `refund` (easy). The `query` (hard) task is designed to challenge even strong models.

To reproduce:

```bash
python inference.py   # runs with mock by default when HF_TOKEN is unset
```

---

## 📁 Project Structure

```
workplace_env/
├── inference.py              # Baseline inference script (root, required)
├── openenv.yaml              # OpenEnv spec metadata
├── Dockerfile                # Container build definition
├── pyproject.toml            # Package config + dependencies
├── requirements.txt          # Pinned dependency list
├── data.py                   # 30+ annotated customer email scenarios
├── models.py                 # WorkplaceAction, WorkplaceObservation exports
│
├── environment/
│   └── workplace_environment.py   # Core env: reset(), step(), state
│
├── core/
│   ├── config.py             # Configuration management
│   ├── graders/              # Modular reward graders (accuracy, keyword, semantic)
│   │   ├── rule_based.py     # RuleBasedRewardPolicy (main grader)
│   │   ├── framework.py      # WeightedParallelGradingEngine
│   │   └── interfaces.py     # BaseGrader, EvaluationContext
│   └── models/
│       └── workplace.py      # Pydantic models: WorkplaceObservation, WorkplaceAction
│
├── server/
│   └── app.py                # FastAPI server (OpenEnv HTTP endpoints)
│
├── api/
│   └── app.py                # Alternative API wiring with inspect.signature safety
│
├── data/
│   └── scenario_repository.py  # ScenarioRepository abstraction
│
└── tests/                    # pytest test suite
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🌐 Hugging Face Space

Live environment: [https://huggingface.co/spaces/Akshar-3011/meta-environment](https://huggingface.co/spaces/Akshar-3011/meta-environment)

The Space exposes the standard OpenEnv HTTP API:

```
POST /reset        → WorkplaceObservation
POST /step         → WorkplaceObservation (with reward, done)
GET  /state        → current episode state dict
GET  /health       → {"status": "ok"}
```

---

## 📝 Environment Variables

| Variable       | Default                                 | Description                          |
|----------------|-----------------------------------------|--------------------------------------|
| `API_BASE_URL` | `https://router.huggingface.co/v1`     | LLM API endpoint                     |
| `MODEL_NAME`   | `Qwen/Qwen2.5-72B-Instruct`            | Model identifier for inference       |
| `HF_TOKEN`     | *(unset → mock mode)*                   | Hugging Face / OpenAI API key        |

---

## License

BSD-style license. See LICENSE file.
