# Workplace Env

A production-grade, modular reinforcement learning environment for multi-step customer support workflows.

This project simulates a realistic 3-step support pipeline where an agent must:

1. classify customer intent,
2. generate a response,
3. decide whether to escalate.

The environment provides structured rewards, a strategy-based inference system, a modular grading engine, CLI tooling, and a FastAPI service layer for integration and experimentation.

## Project overview

`workplace_env` is designed for teams building and evaluating decision-making agents in support scenarios. It combines deterministic evaluation rules with extensible architecture patterns, making it suitable for:

- RL environment prototyping,
- policy benchmarking,
- inference strategy comparison,
- API-first orchestration.

The codebase emphasizes maintainability through clear module boundaries (`core`, `environment`, `api`, `data`, `tests`) and production concerns such as logging, config management, and structured errors.

## Architecture diagram (text-based)

```text
                                                            +----------------------+
                                                            |  CLI (Typer)         |
                                                            |  main.py             |
                                                            +----------+-----------+
                                                                                 |
                                                                                 v
 +------------------------+    +---------+----------+    +----------------------+
 |  FastAPI Layer         |    |  Core Domain       |    |  OpenEnv Runtime     |
 |  api/pipeline_app.py   +--->+  - Inference       +--->+  server/app.py       |
 |  api/app.py            |    |  - Graders         |    |  environment loop    |
 +-----------+------------+    |  - Models          |    +----------+-----------+
                         |                 |  - Config/Logging  |               |
                         v                 +---------+----------+               v
 +-----------+------------+              |               +----------+-----------+
 |  Structured Responses  |              v               |  Data Layer          |
 |  success/error payload |    +---------+----------+    |  scenarios + metadata|
 +------------------------+    |  Environment       |    |  data/, data.py      |
                                                             |  workplace_env.py  |    +----------------------+
                                                             +--------------------+
```

## Features

- **Modular architecture** with clear separation of concerns.
- **Multi-strategy inference**:
    - `StandardInference`
    - `EnhancedInference`
    - `AsyncInference`
- **Inference result caching** with TTL + bounded in-memory cache
- **Weighted modular grading framework**:
    - accuracy grader
    - semantic similarity grader
    - rule-based grader
- **Grader plugin system** (`module:attribute` dynamic loading)
- **Benchmark mode** for strategy/model comparison
- **Result visualization** via ASCII charts and JSONL logs
- **End-to-end pipeline API** with validated request models.
- **CLI commands** for quick inference/grading/pipeline runs.
- **Production foundations**:
    - centralized `.env` config (`core/config.py`)
    - unified logging (`core/logging_config.py`)
    - typed custom exceptions (`core/exceptions.py`)
- **Pytest suite** for inference, graders, API endpoints, and regression checks.

## Installation steps

### Prerequisites

- Python `>= 3.10`
- Recommended: `uv` for fast environment and dependency management

### 1) Clone repository

```bash
git clone https://github.com/akshar-3011/meta-environment.git
cd meta-environment/workplace_env
```

### 2) Create environment and install dependencies

Using `uv` (recommended):

```bash
uv sync
```

Or using `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e .[dev]
```

### 3) Configure environment variables

```bash
cp .env.example .env
```

Then update values in `.env` as needed (ports, retry config, log level, debug mode).

Advanced runtime knobs:

- `CACHE_ENABLED`
- `CACHE_INFERENCE_TTL_SECONDS`
- `CACHE_MAX_ENTRIES`
- `BENCHMARK_DEFAULT_RUNS`
- `BENCHMARK_DEFAULT_CONCURRENCY`

## Usage examples

### Python client usage

```python
from workplace_env import WorkplaceAction, WorkplaceEnv

with WorkplaceEnv(base_url="http://localhost:8000") as env:
        obs = env.reset().observation

        env.step(WorkplaceAction(action_type="classify", content="complaint"))
        env.step(WorkplaceAction(action_type="reply", content="We are sorry and will resolve this quickly."))
        result = env.step(WorkplaceAction(action_type="escalate", content="yes"))

        print("reward:", result.reward)
```

### Start services

OpenEnv server:

```bash
uv run server
```

Pipeline API server:

```bash
uv run pipeline-api
```

### CLI usage

Run inference:

```bash
uv run run-inference --email "I have not received a response for days" --strategy enhanced
```

Run grader:

```bash
uv run run-grader --action-type reply --content "We are sorry and will resolve this." --actual-category complaint --step-count 2
```

Run full pipeline:

```bash
uv run run-pipeline --email "My issue is unresolved" --actual-category complaint --strategy standard
```

Run full pipeline with cache disabled and plugin grader:

```bash
uv run run-pipeline \
    --email "My issue is unresolved" \
    --actual-category complaint \
    --no-cache \
    --plugin my_plugins.graders:PolitenessGrader \
    --plugin-weight 0.05
```

Run benchmark mode (sync):

```bash
uv run run-benchmark --strategies standard,enhanced,async --iterations 3
```

Run benchmark mode (async with visualization + logs):

```bash
uv run run-benchmark \
    --strategies standard,enhanced,async \
    --iterations 5 \
    --async-run \
    --concurrency 8 \
    --show-chart \
    --log-file ./artifacts/benchmark.jsonl
```

### Run tests

```bash
python -m pytest -q
```

Targeted test modules:

```bash
python -m pytest -q tests/test_inference.py tests/test_graders.py tests/test_api.py
```

## API documentation

Base URL (default): `http://localhost:8010`

Interactive docs (when server is running):

- Swagger UI: `/docs`

### Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/infer` | Generate action plan |
| `POST` | `/grade` | Grade one action |
| `POST` | `/pipeline` | Execute inference + grading flow |

### `POST /infer`

Request:

```json
{
    "email": "Support has not replied to my issue",
    "strategy": "enhanced",
    "category_options": ["refund", "complaint", "query"],
    "scenario_difficulty": "medium",
    "urgency": "high",
    "sentiment": "negative",
    "complexity_score": 3
}
```

Response (shape):

```json
{
    "success": true,
    "score": 1.0,
    "breakdown": {
        "strategy": "enhanced",
        "action_count": 3,
        "actions": []
    }
}
```

### `POST /grade`

Request:

```json
{
    "action_type": "reply",
    "content": "We are sorry and will resolve this quickly.",
    "actual_category": "complaint",
    "step_count": 2,
    "scenario_difficulty": "medium",
    "min_reply_length": 30,
    "previous_actions": {
        "classify": 0.9
    }
}
```

Response:

```json
{
    "success": true,
    "score": 0.21,
    "breakdown": {
        "action_type": "reply",
        "final_reward": 0.21
    }
}
```

### `POST /pipeline`

Request:

```json
{
    "email": "My issue is unresolved and frustrating",
    "actual_category": "complaint",
    "strategy": "standard",
    "scenario_difficulty": "easy",
    "min_reply_length": 30
}
```

Response:

```json
{
    "success": true,
    "score": 0.88,
    "breakdown": {
        "total_steps": 3,
        "steps": []
    }
}
```

### Error response format

All errors are returned in a structured shape:

```json
{
    "success": false,
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "Invalid request payload",
        "details": {}
    }
}
```

## Project structure

```text
workplace_env/
├── api/                # FastAPI apps and endpoint orchestration
├── core/               # Config, logging, exceptions, inference, graders, models
├── data/               # Scenario repository abstractions
├── environment/        # OpenEnv-compatible environment implementation
├── tests/              # Pytest suite (inference, graders, API, regression)
├── main.py             # CLI entrypoint
├── client.py           # Python client integration
└── pyproject.toml      # Packaging, dependencies, scripts
```

## Future improvements

- Add benchmark tooling for strategy-vs-strategy evaluation over full scenario sets.
- Add richer semantic grading models (embeddings / LLM-judge adapters).
- Add per-endpoint auth and rate limiting for production deployments.
- Add CI pipeline with coverage thresholds and static analysis gates.
- Add container-first local developer workflow (`docker-compose` for API + tests).
- Expose metrics (`Prometheus`/OpenTelemetry) for reward distribution and latency.

## Contributing

Contributions are welcome. For substantial changes, please open an issue first to discuss scope and design.

When contributing:

- include tests for behavioral changes,
- keep modules cohesive and typed,
- preserve structured response contracts (`success`, `score`, `breakdown` / `error`).

## License

This project follows the repository license terms.

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
