# Security Policy

> Security architecture, threat model, and incident response for meta-environment.

---

## Security Contact

**security@akshar-3011.dev**

For responsible disclosure, please email with:
- Description of the vulnerability
- Steps to reproduce
- Impact assessment
- Your name/handle for credit (optional)

We will acknowledge receipt within **24 hours**and provide a resolution timeline within **72 hours**.

---

## Security Architecture

```
Internet
    │
    ▼
┌──────────────────────────────────────────────┐
│              Ingress (nginx + TLS)           │  ← TLS 1.3 termination
│              Rate limit: 100 req/min         │
└──────────────────┬───────────────────────────┘
                   │
┌──────────────────▼───────────────────────────┐
│           NetworkPolicy                      │  ← Namespace isolation
│     Allow: ingress-nginx, observability      │
└──────────────────┬───────────────────────────┘
                   │
┌──────────────────▼───────────────────────────┐
│  FastAPI Middleware Stack (outermost first)   │
│                                              │
│  1. API Key Authentication                   │  ← Reject unauthenticated
│  2. CORS (allowlist)                         │  ← Cross-origin restriction
│  3. Per-Endpoint Rate Limiting               │  ← /reset: 10/min, etc.
│  4. Global Rate Limiting                     │  ← 1000/min per API key
│  5. Request Size Limit (1MB)                 │  ← Prevent payload abuse
│  6. Error Sanitization                       │  ← Strip stack traces
│  7. Request ID + Trace Context               │  ← Correlation
│  8. Security Headers (CSP, HSTS, etc.)       │  ← Browser hardening
│  9. Prometheus Metrics                       │  ← Observability
└──────────────────┬───────────────────────────┘
                   │
┌──────────────────▼───────────────────────────┐
│           Application Layer                  │
│                                              │
│  ┌────────────┐  ┌────────┐  ┌───────────┐  │
│  │ Environment │  │ Grader │  │ Experiments│  │
│  │  (no eval)  │  │(no I/O)│  │ (SQLite)  │  │
│  └────────────┘  └────────┘  └───────────┘  │
└──────────────────────────────────────────────┘
```

### Defense in Depth

| Layer | Control | Implementation |
|---|---|---|
| Network | Namespace isolation | K8s NetworkPolicy |
| Transport | TLS 1.3 | Ingress + cert-manager |
| Authentication | API key | Middleware (X-API-Key header) |
| Authorization | Endpoint ACLs | Rate limit per endpoint |
| Input validation | Pydantic models | Request/response validation |
| Output sanitization | Error middleware | No stack traces in production |
| Container | Hardened | Non-root, read-only FS, drop caps |
| Logging | Audit trail | JSON → SIEM (Splunk/Datadog) |

---

## Threat Model (STRIDE)

### Spoofing

| Threat | Risk | Mitigation |
|---|---|---|
| API key theft | Medium | Keys rotated quarterly; never logged in plaintext (SHA-256 hash only) |
| IP spoofing | Low | Rate limiting uses `X-Forwarded-For` from trusted ingress only |
| Client impersonation | Low | API keys are per-client; no session tokens to steal |

### Tampering

| Threat | Risk | Mitigation |
|---|---|---|
| Request body modification | Low | TLS in transit; Pydantic validation on all inputs |
| Scenario data corruption | Low | Read-only filesystem; scenarios loaded from immutable Python module |
| Reward manipulation | Medium | Grading is deterministic + server-side; client cannot influence scoring |

### Repudiation

| Threat | Risk | Mitigation |
|---|---|---|
| Deny API usage | Low | Audit log records all auth attempts, resets, and rate limit events |
| Deny experiment creation | Low | Experiment store has immutable created_at + creator tracking |

### Information Disclosure

| Threat | Risk | Mitigation |
|---|---|---|
| Stack trace leaks | Medium | ErrorSanitizationMiddleware strips traces in production |
| API key in logs | Medium | Keys are SHA-256 hashed before logging |
| Scenario data leakage | Low | Scenarios are part of the public environment spec |
| Internal IP exposure | Low | X-Forwarded-For not leaked; response headers controlled |

### Denial of Service

| Threat | Risk | Mitigation |
|---|---|---|
| Request flooding | High | 3-tier rate limiting (per-endpoint → global → K8s ingress) |
| Large payload attacks | Medium | 1MB request body limit |
| Slowloris attacks | Low | Uvicorn timeout + K8s readiness probe |
| Resource exhaustion | Medium | Container CPU/memory limits; HPA for auto-scaling |

### Elevation of Privilege

| Threat | Risk | Mitigation |
|---|---|---|
| Container breakout | Low | Non-root, read-only FS, drop ALL caps, seccomp |
| Lateral movement | Low | NetworkPolicy restricts egress; SA auto-mount disabled |
| Code injection | Low | No `eval()`, no `pickle`, no user-controlled code execution |

---

## Incident Response

### Severity Levels

| Level | Description | Response Time | Example |
|---|---|---|---|
| P0 | Data breach, RCE, auth bypass | **15 min**| API key compromise |
| P1 | Service-wide DOS, data corruption | **1 hour**| Rate limiter failure |
| P2 | Single-user impact, info disclosure | **4 hours**| Stack trace leak |
| P3 | Low-impact, hardening gaps | **1 week**| Missing header |

### Response Procedure

```
1. DETECT
   ├── Automated: Prometheus alerts, audit log monitoring
   ├── External: Bug bounty / responsible disclosure
   └── Internal: Code review, dependency scan

2. TRIAGE (within SLA)
   ├── Assign severity (P0-P3)
   ├── Identify blast radius
   └── Notify stakeholders

3. CONTAIN
   ├── P0: Immediately rotate compromised credentials
   ├── P1: Apply rate limit override / circuit breaker
   └── P2/P3: Document and schedule fix

4. REMEDIATE
   ├── Develop fix on security/* branch
   ├── Peer review by security-trained engineer
   ├── Deploy to staging → verify → production
   └── Update SECURITY_AUDIT.md

5. POST-MORTEM
   ├── Root cause analysis
   ├── Timeline documentation
   ├── Prevention measures
   └── Update threat model
```

### Emergency Commands

```bash
# Rotate API key immediately
kubectl set env deployment/meta-env API_KEY="$(openssl rand -hex 32)" \
  --namespace meta-environment

# Block specific IP at ingress
kubectl annotate ingress meta-env \
  nginx.ingress.kubernetes.io/configuration-snippet='deny 10.0.0.1;' \
  --namespace meta-environment

# Scale down to minimum (circuit breaker)
kubectl scale deployment/meta-env --replicas=1 --namespace meta-environment

# View audit logs
kubectl logs -l app.kubernetes.io/name=meta-environment \
  --namespace meta-environment | grep '"event":"auth.failure"'

# Rollback to last known good
helm rollback meta-env --namespace meta-environment
```

---

## Responsible Disclosure Policy

We follow a **90-day disclosure timeline**:

1. **Report**the vulnerability to security@akshar-3011.dev
2. We **acknowledge**within 24 hours
3. We provide a **fix timeline**within 72 hours
4. We release a **patch**within 90 days
5. We **credit**the reporter (unless anonymity requested)
6. Reporter may **publicly disclose**after patch release

### In Scope

- API authentication bypass
- Rate limiting bypass
- Reward manipulation / grading exploits
- Information disclosure (stack traces, internal IPs)
- Container escape / privilege escalation
- Dependency vulnerabilities (CVEs)

### Out of Scope

- Social engineering attacks
- Physical access attacks
- Issues in upstream dependencies with existing CVEs (report upstream)
- Denial of service via high request volume (we have rate limiting)

---

## Security Checklist (Pre-Release)

- [x] API key authentication on all state-mutating endpoints
- [x] Per-endpoint rate limiting with Retry-After headers
- [x] Security response headers (CSP, HSTS, X-Frame-Options, etc.)
- [x] Error sanitization (no stack traces in production)
- [x] Request body size limit (1MB)
- [x] Structured audit logging (JSON → SIEM)
- [x] API keys hashed in logs (SHA-256)
- [x] Container hardened (non-root, read-only FS, drop caps)
- [x] Network policies (namespace isolation)
- [x] Dependency vulnerability scan (pip-audit)
- [x] Static security analysis (bandit)
- [x] CORS allowlist (no wildcard in production)
- [x] Input validation (Pydantic models)
- [x] No `eval()`, `exec()`, `pickle` usage
- [x] Secrets management (SOPS / External Secrets Operator)
