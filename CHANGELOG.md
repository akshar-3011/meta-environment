# Changelog

All notable changes to the Workplace Customer Support Environment.

## [1.0.0] — 2026-04-10

### 🚀 Production Release

First production-grade release after comprehensive audit and refactoring.

### Critical Bug Fixes (C1–C10)
- **C1:** Fixed `Environment` base class fallback — stub provided when `openenv-core` not installed
- **C2:** Fixed dead escalation timing logic — bonus now correctly fires at step 3 (was `< 2` / `>= 2`, now `< 3` / `>= 3`)
- **C3:** Removed per-call `ThreadPoolExecutor` — sequential grading eliminates thread churn
- **C4:** Removed global `DEBUG` mutation — debug flag is now instance-owned
- **C5:** Fixed retry loop — `raise_for_status()` replaced with `continue` for proper backoff
- **C6:** Scaled consistency penalty — linear scaling replaces binary cliff (was 0.2 flat, now `0.4 × (0.5 − score)`)
- **C7:** Removed duplicate `/health` endpoint in `server/app.py` — prevents route shadowing
- **C8:** Added top-level `import json` — was imported inside the step hot path
- **C9:** Updated `/tasks` endpoint to return correct task names (`easy-triage`, `medium-triage`, `hard-triage`)
- **C10:** Added `EXPOSE 8000` to Dockerfile for orchestrator compatibility

### High-Impact Refactors (N1–N10)
- **N1:** Removed `ThreadPoolExecutor` from grading engine → +30–50% throughput
- **N2:** Added Pydantic validation at scenario load — all 39 scenarios validated at import time
- **N3:** Made `get_config()` reloadable — `reload_config()` + `override_config()` for tests
- **N4:** Fixed `/infer` to return real graded score (was hardcoded `1.0`)
- **N5–N6:** Repository hygiene — `.venv/`, `__pycache__/`, duplicate trees excluded
- **N7:** Added `connect_timeout` to HTTP requests — prevents infinite TCP hangs
- **N8:** JSON structured logging — auto-activates when `APP_ENV=production`
- **N9:** Frozen `GradeResult` dataclass — immutable scores after creation
- **N10:** `Literal["classify","reply","escalate"]` for `action_type` — invalid types rejected at API boundary

### Testing & CI/CD
- 68 tests across 10 test files (was ~30)
- New test files: `test_grader_crash_handling.py`, `test_escalation_metadata.py`, `test_scenario_validation.py`, `test_escalation_timing_bonus.py`, `test_docker_healthcheck.py`
- GitHub Actions CI pipeline: lint → type check → tests → OpenEnv validate → Docker build
- pytest config with coverage enforcement

### Security & Operations
- API key authentication (`API_KEY` env var, `X-API-Key` header)
- CORS allowlist (`CORS_ORIGINS` env var, comma-separated)
- Per-IP rate limiting (`RATE_LIMIT_PER_MINUTE`, sliding window, 429 on exceed)
- Rollback script (`scripts/rollback.sh --dry-run`)
- `.dockerignore` for clean build context

### Observability
- `/metrics` endpoint — Prometheus exposition format with:
  - `env_requests_total` (counter by endpoint/method/status)
  - `env_request_duration_seconds` (histogram with P50/P95/P99 buckets)
  - `env_reward_distribution` (histogram by step/difficulty)
  - `env_episodes_total`, `env_steps_total`, `env_errors_total`
- OpenTelemetry distributed tracing (auto-instruments FastAPI)
- Prometheus alerting rules (`prometheus/alerts.yml`)

### Agent Training
- Gymnasium wrapper (`environment/gym_wrapper.py`): MultiDiscrete action space, 13-feature Box observations
- PPO training script (`examples/train_ppo.py`): converges in 10k steps, 100% classification/escalation accuracy
- Registered gym environments: `WorkplaceTriage-v1`, `WorkplaceTriageEasy-v1`, `WorkplaceTriageMedium-v1`, `WorkplaceTriageHard-v1`

### Performance
- P50 latency: 0.3ms (target: < 200ms)
- P99 latency: 0.4ms (target: < 500ms)
- Throughput: 3,022 episodes/sec (target: > 100)
- Memory: 0.1MB per episode (target: < 50MB)
- Benchmark suite (`benchmarks/load_test.py`): direct, HTTP, and load test modes

---

## [0.1.0] — 2026-04-08

### Initial Release
- Basic OpenEnv environment with 39 scenarios
- Rule-based reward function (classify 40%, reply 35%, escalate 25%)
- FastAPI server with `/reset`, `/step`, `/state` endpoints
- Mock inference script
- Docker multi-stage build
