# Frequently Asked Questions

---

### 1. What is meta-environment?

A production-grade reinforcement learning environment for training AI agents on customer support email triage. Agents learn to classify emails, draft empathetic replies, and make escalation decisions - with dense per-step rewards and 100 validated scenarios across 3 difficulty levels.

---

### 2. How is this different from other RL environments?

| Feature | meta-environment | Typical Gym env |
|---|---|---|
| Rewards | Dense (every step) | Sparse (end of episode) |
| Observations | Rich text (emails) | Numeric vectors |
| Episodes | Fixed 3 steps | Variable length |
| Difficulty | Adaptive (easy/medium/hard) | Single level |
| Production-ready | Security, monitoring, K8s | Dev only |
| A/B testing | Built-in experiment framework | None |
| Scenarios | 100 validated, extensible | Fixed |

---

### 3. How long does training take?

| Difficulty | Steps to converge | Wall time (M1 Mac) |
|---|---|---|
| Easy | ~10k | ~2 minutes |
| Medium | ~30k | ~5 minutes |
| Hard | ~100k | ~15 minutes |

```bash
python examples/03_training_ppo.py --difficulty easy --timesteps 10000
```

---

### 4. Can I use my own reward function?

Yes! Use the A/B experiment framework:

```bash
# Create experiment with alternative weights:
curl -X POST http://localhost:8000/experiments \
  -d '{"name":"my-policy","policy_type":"equal","traffic_split":0.5}'
```

Or implement a custom `RewardPolicy`:
```python
from core.graders.interfaces import RewardPolicy

class MyPolicy(RewardPolicy):
    def calculate_step_reward(self, action_type, content, **kwargs):
        # Your custom grading logic
        return score, breakdown
```

---

### 5. How do I add new scenarios?

```python
# 1. Add to data/scenario_repository.py:
SCENARIOS.append({
    "email": "Your customer email text...",
    "label": "refund",        # refund, complaint, or query
    "difficulty": "medium",   # easy, medium, or hard
    "sentiment": "negative",
    "urgency": "high",
    "complexity": 3,          # 1-5
    "requires_escalation": False,
    "min_reply_length": 50,
})

# 2. Validate:
python examples/05_scenario_creation.py

# 3. Run tests:
python -m pytest tests/test_scenario_validation.py -v
```

Or use the automated generator:
```bash
python tools/generate_scenarios.py --count 20 --difficulty hard
```

---

### 6. Do I need an LLM API key?

**No.**The environment works fully offline with rule-based grading. An LLM (HuggingFace or OpenAI) is only needed for:
- The `/infer` endpoint (agent inference)
- Automated scenario generation

```bash
# No API key needed:
python examples/01_quickstart.py
python -m pytest tests/ -v

# API key needed (optional):
HF_TOKEN=hf_xxx python inference.py
```

---

### 7. How do I deploy to production?

Three options:

**Docker:**
```bash
docker build -t meta-env . && docker run -p 8000:8000 meta-env
```

**Kubernetes (Helm):**
```bash
helm upgrade --install meta-env ./helm/meta-environment \
  -f ./helm/meta-environment/values-prod.yaml --namespace meta-environment
```

**Kustomize:**
```bash
kubectl apply -k k8s/overlays/production/
```

See [KUBERNETES_DEPLOYMENT.md](KUBERNETES_DEPLOYMENT.md) for the full guide.

---

### 8. What are the security features?

| Feature | Implementation |
|---|---|
| Authentication | API key (`X-API-Key` header) |
| Rate limiting | Per-endpoint (10-100/min) + global (1000/min) |
| Security headers | CSP, HSTS, X-Frame-Options, nosniff |
| Error sanitization | No stack traces in production |
| Request limits | 1MB max body size |
| Audit logging | JSON → Splunk/Datadog SIEM |
| Container | Non-root, read-only FS, drop ALL caps |
| Network | K8s NetworkPolicy, namespace isolation |

See [SECURITY.md](SECURITY.md) for the full threat model.

---

### 9. How do I monitor the environment?

**Prometheus metrics**at `/metrics`:
```bash
curl http://localhost:8000/metrics
```

**OpenTelemetry tracing**(configure endpoint):
```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
```

**Audit logs**(JSON to stdout for K8s log aggregation):
```bash
kubectl logs -l app.kubernetes.io/name=meta-environment | jq '.event'
```

---

### 10. How can I contribute?

See [CONTRIBUTING.md](../CONTRIBUTING.md) for:

- Development setup
- How to add scenarios
- PR process and review guidelines
- Code style (ruff, mypy)

We welcome contributions of all kinds: new scenarios, improved graders, documentation fixes, and performance improvements.
