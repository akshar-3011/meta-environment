# Changelog

All notable changes to meta-environment are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2026-04-10

### 🎉 First Production Release

Production-grade OpenEnv-compliant RL environment for customer support triage
with 100 scenarios, dense rewards, sub-millisecond latency, and full
observability stack.

### Added

#### Environment
- 100 validated scenarios across 3 difficulty levels (33 easy, 34 medium, 33 hard)
- 3-step episodes: classify → reply → escalate with dense per-step rewards
- Gymnasium wrapper for Stable-Baselines3 training compatibility
- Configurable difficulty multipliers (easy ×1.0, medium ×1.05, hard ×1.12)
- Scenario cycling with round-robin selection

#### Reward System
- Weighted grading: 40% classify, 35% reply, 25% escalate
- Classification: exact match, adjacent-label partial credit, hard-scenario partial credit
- Reply quality: length, keywords, empathy, solution specificity, greeting/closing
- Escalation: correctness, trajectory consistency bonus/penalty
- A/B experiment framework with 4 reward policies (equal, escalation_first, reply_quality)
- Statistical analysis tool (Welch's t-test, automated deploy/continue/abort)

#### API
- POST `/reset` — start new episode
- POST `/step` — submit action
- GET `/state` — episode state
- GET `/health` — liveness probe
- GET `/metrics` — Prometheus metrics
- POST `/experiments` — create A/B experiment
- GET `/experiments/{id}` — experiment status

#### Security
- API key authentication with audit logging
- Per-endpoint rate limiting (/reset: 10/min, /step: 100/min, etc.)
- Global rate limiting (1000/min per API key)
- Security headers: CSP, HSTS, X-Frame-Options, nosniff, Referrer-Policy
- Error sanitization (no stack traces in production)
- 1MB request body limit
- SIEM-compatible audit logging (JSON → Splunk/Datadog)

#### Observability
- Prometheus counters and histograms (request count, latency, reward distribution)
- OpenTelemetry distributed tracing with auto-instrumentation
- Request ID correlation (X-Request-ID header)
- Structured JSON logging

#### Deployment
- Docker multi-stage build (OpenEnv base image)
- Kubernetes manifests (Kustomize base + production overlay)
- Helm chart with dev/staging/prod values files
- HPA (10-50 replicas, CPU 70%, memory 80%)
- PodDisruptionBudget (minAvailable: 7)
- NetworkPolicy (ingress from nginx + observability only)
- GitHub Actions CI/CD (build → deploy → smoke test → auto-rollback)

#### Training
- Multi-agent training pipeline (conservative, aggressive, balanced archetypes)
- PPO training script with checkpointing and TensorBoard logging
- Convergence detection (reward variance < 0.01 over 5k steps)
- 5 runnable example scripts (quickstart → evaluation)

#### Documentation
- Architecture guide with system diagrams
- Complete API reference with request/response schemas
- Security documentation with STRIDE threat model
- Kubernetes deployment guide
- A/B experimentation guide
- Troubleshooting guide
- FAQ (10 questions)
- Contributing guide

### Fixed (Critical Bugs C1–C10)
- **C1:** Docker container wouldn't start (missing `__main__.py`)
- **C2:** Deterministic rewards broken by race condition in grading
- **C3:** Reset crashed on invalid scenario ID
- **C4:** Step function didn't validate action order
- **C5:** Escalation timing window was dead (always false)
- **C6:** Thread pool caused GIL contention in grading
- **C7:** Retry loops in inference client had no backoff
- **C8:** Consistency penalty scaled incorrectly above 1.0
- **C9:** Route conflicts between `/state` and `/step`
- **C10:** Docker healthcheck path was wrong

### Changed (High-Impact Refactors N1–N10)
- **N1:** Sequential grading engine (+30% throughput)
- **N2:** Pydantic validation at scenario load time (not runtime)
- **N3:** Frozen dataclass models (immutable state)
- **N4:** Literal action types (type-safe)
- **N5:** JSON structured logging (machine-parseable)
- **N6:** Reloadable config singleton
- **N7:** Real `/infer` scores (not hardcoded)
- **N8:** Prometheus lazy registration (no duplicate errors)
- **N9:** Scenario count expanded from 39 to 100
- **N10:** Test coverage expanded from 68 to 232 tests

### Performance
| Metric | v0.x | v1.0.0 |
|---|---|---|
| P50 latency | ~150ms | **0.3ms** |
| P99 latency | ~400ms | **0.4ms** |
| Throughput | ~80 eps/s | **3,022 eps/s** |
| Memory/episode | ~30MB | **0.1MB** |
| Test count | 68 | **232** |
| Scenarios | 39 | **100** |
