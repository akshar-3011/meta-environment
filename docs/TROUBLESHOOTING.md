# Troubleshooting

> Common errors and fixes for meta-environment.

---

## Installation Issues

### `ModuleNotFoundError: No module named 'openenv'`

**Cause:**The OpenEnv core library is not installed.

**Fix:**
```bash
pip install openenv-core>=0.2.2
# Or install from the repo:
pip install -e ".[dev]"
```

### `ImportError: cannot import name 'create_app'`

**Cause:**Incompatible version of openenv-core.

**Fix:**
```bash
pip install --upgrade openenv-core>=0.2.2
```

---

## Runtime Errors

### `ValueError: Duplicated timeseries in CollectorRegistry`

**Cause:**Prometheus metrics are being registered twice (common during test runs or hot-reload).

**Fix:**
```python
# The metrics module already handles this with lazy initialization.
# If running tests, ensure you use the test fixtures that reload config.
# For uvicorn: don't use --reload in production.
```

### `RuntimeError: Environment not reset`

**Cause:**Calling `/step` before `/reset`.

**Fix:**
```bash
# Always reset first:
curl -X POST http://localhost:8000/reset -d '{}'
# Then step:
curl -X POST http://localhost:8000/step -d '{"action": {"action_type": "classify", "content": "refund"}}'
```

### `422 Unprocessable Entity: Invalid action type`

**Cause:**Action type not one of `classify`, `reply`, `escalate`.

**Fix:**Ensure `action_type` is exactly one of:
- `"classify"` (step 1)
- `"reply"` (step 2)
- `"escalate"` (step 3)

Steps must be performed in order.

---

## Docker Issues

### `docker build` fails with `openenv-base not found`

**Cause:**Base image not available locally.

**Fix:**
```bash
# Pull the base image first:
docker pull ghcr.io/meta-pytorch/openenv-base:latest

# Or build without base image:
docker build --build-arg BASE_IMAGE=python:3.11-slim -t meta-env .
```

### Container exits immediately

**Cause:**Port conflict or missing dependencies.

**Fix:**
```bash
# Check logs:
docker logs <container_id>

# Run with interactive mode:
docker run -it --rm -p 8000:8000 meta-env /bin/sh

# Check if port is in use:
lsof -i :8000
```

### Health check fails in container

**Cause:**Application hasn't started yet (startup time).

**Fix:**The `startupProbe` in K8s allows 50 seconds (10 failures × 5s period). For Docker:
```bash
docker run -p 8000:8000 meta-env
# Wait 5 seconds, then:
curl http://localhost:8000/health
```

---

## Training Issues

### `ImportError: stable_baselines3`

**Cause:**SB3 not installed.

**Fix:**
```bash
pip install stable-baselines3[extra]
```

### Agent reward doesn't improve

**Possible causes:**
1. **Wrong difficulty:**Start with `easy` difficulty first
2. **Learning rate too high:**Try `3e-4` (default)
3. **Too few timesteps:**Easy scenarios need ~10k steps to converge
4. **Observation space mismatch:**Ensure your custom wrapper matches the expected observation format

**Fix:**
```bash
python examples/03_training_ppo.py --difficulty easy --timesteps 50000
# Monitor with TensorBoard:
tensorboard --logdir ./logs/
```

### `KeyError: 'email'` in Gym wrapper

**Cause:**Observation dict structure changed.

**Fix:**Check that your environment version matches the wrapper version:
```python
env = WorkplaceEnvironment()
obs = env.reset()
print(obs.keys())  # Should include: email, category_options, history, reward, done
```

---

## Performance Issues

### High latency (>10ms per episode)

**Possible causes:**
1. Debug logging enabled
2. Too many concurrent environments
3. OpenTelemetry tracing with high sample rate

**Fix:**
```bash
# Set production mode:
export APP_ENV=production
export APP_LOG_LEVEL=WARNING
export OTEL_TRACES_SAMPLE_RATE=0.1
```

### Memory growth over time

**Cause:**Rate limiter or experiment store not garbage collecting.

**Fix:**The rate limiter has automatic GC (every 2× window). For long-running deployments:
```bash
# Check memory:
kubectl top pods -l app.kubernetes.io/name=meta-environment

# Restart pods (rolling):
kubectl rollout restart deployment/meta-env
```

---

## Testing Issues

### Tests fail with `CollectorRegistry` error

**Cause:**Prometheus metrics registered multiple times across test modules.

**Fix:**Run tests in isolation or with the `--forked` flag:
```bash
# Recommended:
python -m pytest tests/ -v

# If still failing:
python -m pytest tests/test_security.py -v  # Run specific file
```

### Rate limiter tests are flaky

**Cause:**Time-sensitive sliding window calculations.

**Fix:**The tests use generous thresholds. If still flaky:
```bash
python -m pytest tests/test_security.py::TestStrictRateLimiter -v --count=3
```

---

## Deployment Issues

### Helm install fails

**Cause:**Missing namespace or RBAC permissions.

**Fix:**
```bash
# Create namespace:
kubectl create namespace meta-environment

# Check permissions:
kubectl auth can-i create deployments --namespace meta-environment

# Debug Helm:
helm install meta-env ./helm/meta-environment --dry-run --debug
```

### Pods in CrashLoopBackOff

**Fix:**
```bash
# Check logs:
kubectl logs -l app.kubernetes.io/name=meta-environment --previous

# Common causes:
# 1. Missing secrets → kubectl get secrets -n meta-environment
# 2. OOM killed → kubectl describe pod <name> | grep OOMKilled
# 3. Probe failure → check /health endpoint works
```

---

## Getting Help

1. Check this document first
2. Search [existing issues](https://github.com/akshar-3011/meta-environment/issues)
3. Open a new issue with:
   - Python version (`python --version`)
   - Package versions (`pip freeze | grep -E "openenv|fastapi|pydantic"`)
   - Full error traceback
   - Steps to reproduce
