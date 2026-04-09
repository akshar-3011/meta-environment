# Contributing to meta-environment

Thank you for your interest in contributing! This guide will help you get started.

---

## Table of Contents

- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Adding Scenarios](#adding-scenarios)
- [Code Style](#code-style)
- [Pull Request Process](#pull-request-process)
- [PR Templates](#pr-templates)

---

## Development Setup

### Prerequisites

- Python ≥ 3.10
- Docker (optional, for container testing)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/akshar-3011/meta-environment.git
cd meta-environment

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install in development mode
pip install -e ".[dev,observability]"

# Or with uv (faster):
uv sync --dev

# Verify installation
python -m pytest tests/ -v --tb=short
# Expected: 232 passed ✅
```

### Pre-commit Hooks (recommended)

```bash
pip install pre-commit
pre-commit install

# Hooks run automatically on git commit:
# - ruff (linting + formatting)
# - mypy (type checking)
# - pytest (quick smoke test)
```

### IDE Setup

**VS Code** (recommended settings):
```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "none",
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true
  }
}
```

---

## Running Tests

```bash
# Full test suite (232 tests, ~0.3s)
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_security.py -v

# With coverage
python -m pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html

# Performance benchmarks
python -m pytest benchmarks/test_performance.py -v --benchmark-enable

# Run only fast tests (skip benchmarks)
python -m pytest tests/ -v -m "not benchmark"
```

### Test Structure

```
tests/
├── test_workplace_environment.py    # Core environment tests
├── test_production_grade.py         # Production hardening tests
├── test_experiments.py              # A/B experiment framework tests
├── test_security.py                 # Security hardening tests
├── test_scenario_validation.py      # Scenario data validation
└── ...                              # Additional test files
```

### Writing Tests

```python
"""Tests for my_new_feature."""
import pytest
from environment.workplace_environment import WorkplaceEnvironment

class TestMyFeature:
    @pytest.fixture
    def env(self):
        return WorkplaceEnvironment()

    def test_basic_behavior(self, env):
        obs = env.reset()
        assert "email" in obs
        assert not obs["done"]

    def test_edge_case(self, env):
        """Document the edge case being tested."""
        # Arrange
        obs = env.reset()
        # Act
        result = env.step({"action_type": "classify", "content": "refund"})
        # Assert
        assert 0.0 <= result["reward"] <= 1.0
```

---

## Adding Scenarios

### Manual Addition

1. Add scenario dict to `data/scenario_repository.py`:

```python
{
    "email": "Dear Support Team, ...",
    "label": "refund",             # refund, complaint, or query
    "difficulty": "medium",         # easy, medium, or hard
    "sentiment": "negative",        # positive, neutral, negative, mixed
    "urgency": "high",             # low, medium, high
    "complexity": 3,               # 1-5
    "requires_escalation": False,   # True or False
    "min_reply_length": 50,        # Minimum reply length for full score
}
```

2. Validate:
```bash
python examples/05_scenario_creation.py
python -m pytest tests/test_scenario_validation.py -v
```

3. Check balance:
```python
from data.scenario_repository import SCENARIOS
from collections import Counter
print(Counter(s["difficulty"] for s in SCENARIOS))
# Target: ~33% easy, ~34% medium, ~33% hard
```

### Automated Generation

```bash
# Generate 20 new hard scenarios:
python tools/generate_scenarios.py --count 20 --difficulty hard

# Validate and merge:
python tools/validate_scenarios.py
python data/merge_scenarios.py
```

### Scenario Guidelines

- **Easy:** Single clear intent, neutral/positive sentiment, no ambiguity
- **Medium:** Mixed signals, emotional language, some ambiguity
- **Hard:** Sarcasm, multi-intent, adversarial, edge cases
- **Avoid:** Duplicate themes, unrealistic scenarios, offensive content
- **Length:** 3-8 sentences per email
- **Labeling:** Each scenario must have exactly one correct label

---

## Code Style

### Tools

| Tool | Purpose | Config |
|---|---|---|
| `ruff` | Linting + formatting | `pyproject.toml` |
| `mypy` | Type checking | `pyproject.toml` |
| `pytest` | Testing | `pyproject.toml` |

### Quick Check

```bash
# Lint
ruff check .

# Format
ruff format .

# Type check
mypy core/ api/ environment/ --ignore-missing-imports
```

### Conventions

1. **Type hints** on all public functions
2. **Docstrings** on all classes and public methods (Google style)
3. **No `eval()`, `exec()`, `pickle`** in application code
4. **Frozen dataclasses** for data models
5. **Relative imports** within packages, absolute for cross-package
6. **Constants** in UPPER_SNAKE_CASE at module level
7. **Test classes** prefixed with `Test`, test methods with `test_`

### Example

```python
"""Module docstring explaining purpose."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MyModel:
    """One-line description.

    Attributes:
        name: Human-readable name.
        value: Numeric value in range [0, 1].
    """
    name: str
    value: float

    def validate(self) -> bool:
        """Check invariants."""
        return 0.0 <= self.value <= 1.0
```

---

## Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/my-feature
# or: bugfix/fix-description
# or: docs/update-readme
```

### 2. Make Changes

- Write code with tests
- Ensure all tests pass: `python -m pytest tests/ -v`
- Run linting: `ruff check . && ruff format --check .`

### 3. Commit

```bash
git add -A
git commit -m "feat: add new grading metric for reply sentiment

- Added SentimentGrader that checks reply tone matches customer emotion
- Added 5 test cases covering positive, negative, and mixed scenarios
- Updated reward breakdown to include sentiment_score field"
```

**Commit message format:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Tests
- `perf:` Performance
- `refactor:` Code restructuring
- `security:` Security improvement
- `infra:` Infrastructure/deployment

### 4. Open PR

Push and open a PR against `main`:
```bash
git push origin feature/my-feature
```

### 5. Review

- **1 approval required** from a maintainer
- **CI must pass** (all tests, linting)
- **Performance tests** must not regress >10%

---

## PR Templates

### Feature PR

```markdown
## What
Brief description of the feature.

## Why
Link to issue or motivation.

## How
Technical approach.

## Testing
- [ ] Unit tests added
- [ ] Integration tests added
- [ ] Performance tested (no regression)
- [ ] Documentation updated

## Screenshots
(if UI changes)
```

### Bugfix PR

```markdown
## Bug
Description of the bug (link to issue).

## Root Cause
What caused the bug.

## Fix
How it was fixed.

## Testing
- [ ] Regression test added
- [ ] Existing tests still pass
- [ ] Manually verified fix
```

### Documentation PR

```markdown
## What Changed
Which docs were updated.

## Why
What was missing or incorrect.

## Checklist
- [ ] Links verified
- [ ] Code examples tested
- [ ] Spelling/grammar checked
```

---

## Questions?

- Open a [GitHub Discussion](https://github.com/akshar-3011/meta-environment/discussions)
- Check [FAQ.md](docs/FAQ.md)
- Check [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

Thank you for contributing! 🎉
