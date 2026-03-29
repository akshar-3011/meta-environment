---
title: Workplace Env Environment Server
emoji: 🧠
colorFrom: purple
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
---

# Workplace Env — Multi-Step Customer Support RL Environment

A production-grade reinforcement learning environment simulating a 3-step customer support decision pipeline. An RL agent must classify incoming email intent, generate a contextually appropriate reply, and decide whether to escalate — with a shaped reward signal that requires balancing accuracy across all three subtasks.

**This is not a toy environment.** Wrong classification penalises downstream steps. Escalation requires understanding true intent. The 3-step dependency creates a real credit assignment challenge.

---

## Quick Start
```python
from workplace_env import WorkplaceAction, WorkplaceEnv

with WorkplaceEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(result.observation.email)

    result = env.step(WorkplaceAction(action_type="classify", content="refund"))
    result = env.step(WorkplaceAction(action_type="reply", content="We will process your refund within 3 business days."))
    result = env.step(WorkplaceAction(action_type="escalate", content="no"))
    print(result.reward)
```

Or via Docker:
```python
env = WorkplaceEnv.from_docker_image("workplace_env-env:latest")
try:
    result = env.reset()
    ...
finally:
    env.close()
```

---

## Running Locally
```bash
# Terminal 1: Start server
uv run server

# Terminal 2: Run the agent
python inference.py

# Run enhanced agent with rich output
python inference_enhanced.py

# Run validation tests
python test_production_grade.py
```

---

## Deploying to Hugging Face Spaces
```bash
# From environment directory (where openenv.yaml is located)
openenv push

# With options
openenv push --repo-id my-org/my-env --private
```

The `openenv push` command validates the directory, prepares a Docker build, and uploads to Hugging Face.

After deployment, your space exposes:
- **Web Interface** at `/web`
- **API Docs** at `/docs`
- **Health Check** at `/health`
- **WebSocket** at `/ws`

---

## Environment Details

### Actions
**WorkplaceAction** — One per step:

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | str | `classify`, `reply`, or `escalate` |
| `content` | str | Category name, reply text, or escalation decision |
| `confidence` | float (optional) | Agent self-assessment (0.0–1.0) |
| `explanation` | str (optional) | Reasoning for the action |

### Observations
**WorkplaceObservation** — Returned after every reset/step:

| Field | Type | Description |
|-------|------|-------------|
| `email` | str | Customer email to handle |
| `category_options` | list | Valid categories: `refund`, `complaint`, `query` |
| `history` | list | Actions taken so far this episode |
| `scenario_difficulty` | str | `easy` / `medium` / `hard` |
| `urgency` | str | `low` / `medium` / `high` |
| `sentiment` | str | `negative` / `neutral` / `positive` / `mixed` |
| `complexity_score` | int | 1–5 scale |
| `scenario_metadata` | dict | Ground truth label, escalation requirement, min reply length |
| `reward` | float | Step reward (weighted composite, see below) |
| `done` | bool | True after 3 steps |

### Reward Structure

Rewards are weighted by task importance:

| Step | Task | Weight | Max Reward |
|------|------|--------|------------|
| 1 | Classify | 40% | 0.40 |
| 2 | Reply | 35% | 0.35 |
| 3 | Escalate | 25% | 0.25 |

**Total maximum per episode: 1.00**

A perfect agent scores ≥ 0.95 by correctly classifying intent, writing an empathetic keyword-rich reply, and making the right escalation decision.

#### Reward Design Principles

- **Partial credit**: Related-category classification earns 0.2–0.4 instead of 0.0
- **Consistency penalty**: Reply reward is reduced by 0.2 if classification was wrong (error propagation)
- **Escalation policy**: Complaints must be escalated; queries and refunds must not. Over-escalation is penalised
- **Timing penalty**: Escalating before step 2 is penalised (bypassing the workflow)
- **Keyword grading**: Replies are scored on category-specific vocabulary (empathy for complaints, process language for refunds, etc.)

---

## Scenario Dataset

18 scenarios across 3 difficulty levels:

| Difficulty | Count | Description |
|------------|-------|-------------|
| Easy | 7 | Clear single intent, neutral tone |
| Medium | 7 | Mixed signals, negative sentiment, some urgency |
| Hard | 4 | Ambiguous, multi-intent, edge cases |

Each scenario includes: email text, true label, difficulty, sentiment, urgency, complexity score, escalation requirement, and minimum reply length.

Scenarios cycle deterministically for reproducible training.

---

## Why This Is Hard for a Naive Agent

1. **Classification cascades**: A wrong label in step 1 triggers a consistency penalty in step 2, making the total reward lower than the sum of individual mistakes.
2. **Escalation is non-trivial**: The agent cannot default to always-escalate (penalised for over-escalation) or never-escalate (penalised for missed complaints).
3. **Reply quality is rubric-based**: Generic replies fail keyword checks; the agent must learn category-specific language.
4. **Difficulty progression**: Hard scenarios have ambiguous emails where the primary intent is not explicit.

---

## Project Structure
```
workplace_env/
├── __init__.py                    # Module exports
├── README.md                      # This file
├── DESIGN.md                      # RL design rationale
├── IMPROVEMENTS.md                # Upgrade documentation
├── openenv.yaml                   # OpenEnv manifest
├── pyproject.toml                 # Project metadata
├── client.py                      # WorkplaceEnv client (WebSocket)
├── models.py                      # Action, Observation, GradeResult
├── data.py                        # 18 scenario dataset
├── inference.py                   # Basic agent
├── inference_enhanced.py          # Enhanced agent with rich output
├── test_production_grade.py       # Validation tests (9/9 passing)
└── server/
    ├── workplace_env_environment.py  # Core RL logic
    ├── app.py                        # FastAPI app
    └── Dockerfile                    # Container image
```
