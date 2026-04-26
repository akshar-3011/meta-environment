# Kubernetes Deployment Guide

> Production-ready deployment for the meta-environment RL platform on Kubernetes.

---

## Architecture

```
                    ┌─────────────┐
                    │   Ingress   │  (nginx + TLS)
                    │  Controller │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Service   │  ClusterIP :8000
                    │   meta-env  │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
         ┌────▼───┐   ┌───▼────┐   ┌───▼────┐
         │ Pod 1  │   │ Pod 2  │   │ Pod N  │  (10-50 via HPA)
         │  :8000 │   │  :8000 │   │  :8000 │
         └────────┘   └────────┘   └────────┘
              │
         ┌────▼────────────────────────────┐
         │  NetworkPolicy: ingress-nginx   │
         │  + observability namespaces only │
         └─────────────────────────────────┘
```

## Prerequisites

| Tool | Version | Install |
|---|---|---|
| `kubectl` | >= 1.28 | [Instructions](https://kubernetes.io/docs/tasks/tools/) |
| `helm` | >= 3.14 | `brew install helm` |
| `sops` | >= 3.8 | `brew install sops` (for secrets encryption) |
| Cluster access | - | `KUBECONFIG` env var or `~/.kube/config` |

### Verify Access

```bash
kubectl cluster-info
kubectl auth can-i create deployments --namespace meta-environment
helm version
```

---

## Quick Start

### 1. Deploy to Staging

```bash
helm upgrade --install meta-env ./helm/meta-environment \
  -f ./helm/meta-environment/values-staging.yaml \
  --namespace meta-environment --create-namespace \
  --set image.tag=sha-abc1234 \
  --set secrets.API_KEY="your-api-key" \
  --wait --timeout 5m
```

### 2. Deploy to Production

```bash
helm upgrade --install meta-env ./helm/meta-environment \
  -f ./helm/meta-environment/values-prod.yaml \
  --namespace meta-environment --create-namespace \
  --set image.tag=v1.2.3 \
  --set secrets.API_KEY="$API_KEY" \
  --set secrets.OPENAI_API_KEY="$OPENAI_API_KEY" \
  --wait --atomic --timeout 5m
```

> **Note:**`--atomic` automatically rolls back if the deploy fails.

### 3. Use Kustomize (alternative)

```bash
# Base (3 replicas)
kubectl apply -k k8s/base/

# Production overlay (10 replicas + HPA + PDB + NetworkPolicy)
kubectl apply -k k8s/overlays/production/
```

---

## Secrets Management

### Option A: SOPS (recommended for GitOps)

```bash
# Encrypt secret file
sops --encrypt --in-place k8s/base/secret.yaml

# Decrypt for editing
sops k8s/base/secret.yaml

# Set up .sops.yaml in repo root:
cat > .sops.yaml << 'EOF'
creation_rules:
  - path_regex: k8s/.*secret.*\.yaml$
    kms: arn:aws:kms:us-east-1:123456789:key/your-key-id
EOF
```

### Option B: Helm `--set` flags

```bash
helm upgrade --install meta-env ./helm/meta-environment \
  --set secrets.API_KEY="$(vault kv get -field=api_key secret/meta-env)" \
  --set secrets.OPENAI_API_KEY="$(vault kv get -field=openai_key secret/meta-env)"
```

### Option C: External Secrets Operator

```bash
# Use existingSecret to reference a secret managed by ESO
helm upgrade --install meta-env ./helm/meta-environment \
  --set existingSecret=meta-env-external-secrets
```

---

## Scaling

### Manual Scaling

```bash
# Scale to 20 replicas
kubectl scale deployment meta-env --replicas=20 --namespace meta-environment

# Check current replicas
kubectl get deployment meta-env --namespace meta-environment
```

### Auto-Scaling (HPA)

Production enables HPA automatically via `values-prod.yaml`:

```bash
# Check HPA status
kubectl get hpa meta-env --namespace meta-environment

# Watch scaling events
kubectl describe hpa meta-env --namespace meta-environment

# Adjust targets at runtime
kubectl patch hpa meta-env --namespace meta-environment \
  -p '{"spec":{"maxReplicas":100}}'
```

HPA configuration:
| Parameter | Value |
|---|---|
| Min replicas | 10 |
| Max replicas | 50 |
| CPU target | 70% |
| Memory target | 80% |
| Scale-up window | 60s |
| Scale-down window | 300s |

---

## Monitoring & Debugging

### View Logs

```bash
# All pods
kubectl logs -l app.kubernetes.io/name=meta-environment \
  --namespace meta-environment --tail=100 -f

# Specific pod
kubectl logs meta-env-abc123 --namespace meta-environment -f

# Previous container (after crash)
kubectl logs meta-env-abc123 --namespace meta-environment --previous
```

### Interactive Shell

```bash
# Exec into a running pod (read-only FS - use /tmp for writes)
kubectl exec -it deploy/meta-env --namespace meta-environment -- /bin/sh

# Run a one-off command
kubectl exec deploy/meta-env --namespace meta-environment -- python -c "
from data.scenario_repository import SCENARIOS
print(f'Loaded {len(SCENARIOS)} scenarios')
"
```

### Port Forwarding

```bash
# Forward to local port 8000
kubectl port-forward svc/meta-env 8000:8000 --namespace meta-environment

# Test locally
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

### Resource Usage

```bash
# Pod resource consumption
kubectl top pods -l app.kubernetes.io/name=meta-environment \
  --namespace meta-environment

# Node resource pressure
kubectl top nodes
```

### Events & Diagnostics

```bash
# Recent events
kubectl get events --namespace meta-environment \
  --sort-by='.lastTimestamp' --field-selector type=Warning

# Describe pod (for scheduling/probe issues)
kubectl describe pod -l app.kubernetes.io/name=meta-environment \
  --namespace meta-environment

# Check network policies
kubectl get networkpolicy --namespace meta-environment -o yaml
```

---

## Rollback

### Helm Rollback

```bash
# View release history
helm history meta-env --namespace meta-environment

# Rollback to previous release
helm rollback meta-env --namespace meta-environment --wait

# Rollback to specific revision
helm rollback meta-env 3 --namespace meta-environment --wait

# Check rollback succeeded
kubectl rollout status deployment/meta-env --namespace meta-environment
```

### Kubernetes Rollback

```bash
# View rollout history
kubectl rollout history deployment/meta-env --namespace meta-environment

# Undo last rollout
kubectl rollout undo deployment/meta-env --namespace meta-environment

# Undo to specific revision
kubectl rollout undo deployment/meta-env --to-revision=5 --namespace meta-environment
```

---

## CI/CD Pipeline

The `.github/workflows/deploy-k8s.yml` workflow automates the full lifecycle:

```
Tag Push (v*)  ──→  Build Image  ──→  Deploy (Helm)  ──→  Smoke Tests  ──→ 
                         │                  │                   │
                         │            --atomic flag        On failure
                         │            (auto-rollback)           │
                         │                                      ▼
                         └────────────────────────────── Rollback Job
```

### Trigger a Deploy

```bash
# Via tag (deploys to production)
git tag v1.2.3
git push origin v1.2.3

# Via workflow dispatch (any environment)
gh workflow run deploy-k8s.yml \
  -f environment=staging \
  -f image_tag=sha-abc1234
```

### Required Secrets

Set these in GitHub repo settings → Secrets:

| Secret | Description |
|---|---|
| `KUBECONFIG` | Base64-encoded kubeconfig for cluster access |
| `API_KEY` | Environment API key (optional, if not using ESO) |

---

## Environment Comparison

| Setting | Dev | Staging | Production |
|---|---|---|---|
| Replicas | 1 | 2-5 | 10-50 |
| CPU request | 100m | 250m | 1000m |
| Memory request | 128Mi | 256Mi | 1Gi |
| Log level | DEBUG | INFO | WARNING |
| Trace sampling | 100% | 100% | 10% |
| HPA |  |  |  |
| PDB |  | min 1 | min 7 |
| Network Policy |  |  |  |
| Ingress |  |  |  + TLS |

---

## Security Checklist

- [x] Non-root container (`runAsUser: 1000`)
- [x] Read-only root filesystem
- [x] Drop all capabilities
- [x] No privilege escalation
- [x] Seccomp profile: RuntimeDefault
- [x] ServiceAccount auto-mount disabled
- [x] NetworkPolicy restricts ingress/egress
- [x] Secrets via SOPS / External Secrets Operator
- [x] Resource limits prevent noisy-neighbor
- [x] PDB ensures availability during upgrades
