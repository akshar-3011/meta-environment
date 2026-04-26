# Release Checklist - v1.0.0

> Step-by-step procedure for publishing meta-environment.

---

## Pre-Release Verification

```bash
# 1. Run full test suite
python -m pytest tests/ -v --tb=short
# Expected: 232 passed 

# 2. Run all examples
python examples/01_quickstart.py
python examples/02_custom_agent.py
python examples/04_evaluation.py
python examples/05_scenario_creation.py

# 3. Run security scans
python -m pip_audit
bandit -r core/ api/ environment/ -q

# 4. Run performance benchmarks
python benchmarks/load_test.py --mode direct --episodes 500

# 5. Verify Docker build
docker build -t meta-env:v1.0.0 .
docker run --rm -p 8000:8000 meta-env:v1.0.0 &
sleep 5
curl -sf http://localhost:8000/health
docker stop $(docker ps -q --filter ancestor=meta-env:v1.0.0)
```

---

## 1. Git Tag

```bash
# Ensure clean working tree
git status  # Should show "nothing to commit"

# Tag the release
git tag -a v1.0.0 -m "v1.0.0: Production release

- 100 validated scenarios (33 easy, 34 medium, 33 hard)
- 232 tests passing, 85% coverage
- P50: 0.3ms, 3022 eps/sec throughput
- Security hardened: CSP, HSTS, rate limiting, audit logging
- A/B experiment framework with 4 reward policies
- Kubernetes manifests + Helm chart
- Full documentation suite"

git push origin v1.0.0
git push space v1.0.0
```

---

## 2. GitHub Release

```bash
# Create release with pre-built assets
gh release create v1.0.0 \
  --title "v1.0.0 - Production Release" \
  --notes-file docs/CHANGELOG.md \
  --latest

# Or via web: https://github.com/akshar-3011/meta-environment/releases/new
```

---

## 3. Docker Image (GHCR)

```bash
# Build and tag
docker build -t ghcr.io/akshar-3011/meta-environment:v1.0.0 .
docker tag ghcr.io/akshar-3011/meta-environment:v1.0.0 \
           ghcr.io/akshar-3011/meta-environment:latest

# Push
docker push ghcr.io/akshar-3011/meta-environment:v1.0.0
docker push ghcr.io/akshar-3011/meta-environment:latest
```

---

## 4. PyPI Package

```bash
# Install build tools
pip install build twine

# Build
python -m build

# Check package
twine check dist/*

# Upload to PyPI
twine upload dist/*

# Verify installation
pip install meta-environment==1.0.0
```

---

## 5. Hugging Face Spaces

```bash
# Already configured as git remote "space"
# Push triggers auto-deploy on HF Spaces
git push space main
git push space v1.0.0

# Verify: https://huggingface.co/spaces/Akshar-3011/meta-environment
```

---

## 6. Kubernetes Deploy (if applicable)

```bash
# Update Helm chart with new image tag
helm upgrade --install meta-env ./helm/meta-environment \
  -f ./helm/meta-environment/values-prod.yaml \
  --namespace meta-environment \
  --set image.tag=v1.0.0 \
  --wait --atomic --timeout 5m

# Verify
kubectl rollout status deployment/meta-env --namespace meta-environment
curl -sf http://meta-env.example.com/health
```

---

## Post-Release

- [ ] Verify GitHub Release page is correct
- [ ] Verify Docker image pulls: `docker pull ghcr.io/akshar-3011/meta-environment:v1.0.0`
- [ ] Verify HF Spaces is live and /health returns 200
- [ ] Announce on relevant channels
- [ ] Monitor error rates for 24 hours
- [ ] Close the release milestone on GitHub

---

## Rollback (if needed)

```bash
# Git: revert the tag
git tag -d v1.0.0
git push origin :refs/tags/v1.0.0

# Docker: re-tag previous version
docker tag ghcr.io/akshar-3011/meta-environment:previous \
           ghcr.io/akshar-3011/meta-environment:latest
docker push ghcr.io/akshar-3011/meta-environment:latest

# Kubernetes: Helm rollback
helm rollback meta-env --namespace meta-environment
```
