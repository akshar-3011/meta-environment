# Performance SLOs & Troubleshooting

> Service Level Objectives for the workplace environment runtime.

---

## Latency Targets

| Percentile | Target | Measured (v1.0.0) | What It Means |
|---|---|---|---|
| **P50** | < 200ms | **0.32ms** | Half of all episodes finish faster than this |
| **P95** | < 400ms | **0.35ms** | Only 5% of episodes exceed this |
| **P99** | < 500ms | **0.42ms** | One-in-100 worst case |
| **Mean** | < 200ms | **0.32ms** | Arithmetic average across all scenarios |

All latencies are measured per-episode (3 steps: classify + reply + escalate).

---

## Throughput Minimums

| Metric | Target | Measured (v1.0.0) |
|---|---|---|
| Sequential (1 env) | **> 100 eps/s** | **3,022 eps/s** |
| Concurrent (4 envs) | **> 80 eps/s** | **~2,500 eps/s** |

Throughput is measured as complete 3-step episodes per second.

---

## Memory Budget

| Metric | Target | Measured (v1.0.0) |
|---|---|---|
| Peak per 100 episodes | **< 50 MB** | **0.1 MB** |
| Leak ratio (500ep / 100ep) | **< 2.0x** | **~1.0x** |

---

## How To Run Performance Tests

```bash
# Quick run (no benchmark comparison):
pytest benchmarks/test_performance.py -v

# Full run with benchmark data:
pytest benchmarks/test_performance.py -v \
  --benchmark-json=benchmarks/results.json

# Compare against saved baseline:
pytest benchmarks/test_performance.py -v \
  --benchmark-compare=benchmarks/baseline.json

# Update baseline (after merge to main):
python benchmarks/update_baseline.py
python benchmarks/update_baseline.py --confirm   # CI mode
python benchmarks/update_baseline.py --dry-run    # Preview only
```

---

## Regression Detection

Tests automatically detect regressions by comparing against `benchmarks/baseline.json`:

- **Threshold:** Any metric that degrades by **>10%** from baseline fails the test
- **Direction-aware:** Latency/memory use "lower is better", throughput uses "higher is better"
- **CI enforcement:** The `perf-test.yml` workflow blocks PRs on regression

### What Triggers a Regression Alert

| Metric | Alert When |
|---|---|
| `p50_ms` | Increases >10% from baseline |
| `p99_ms` | Increases >10% from baseline |
| `throughput_sequential` | Decreases >10% from baseline |
| `throughput_concurrent` | Decreases >10% from baseline |
| `peak_memory_mb` | Increases >10% from baseline |

---

## Troubleshooting Regressions

### 1. Latency Spike

**Symptom:** P50 or P99 increased  
**Common causes:**
- New code in `_grade_step()` or reward policy added O(n²) logic
- Additional logging/metrics instrumentation in the hot path
- Import added at module level that triggers heavy initialization

**Debug:**
```bash
# Profile a single episode
python -c "
import cProfile
from environment.workplace_environment import WorkplaceEnvironment
from models import WorkplaceAction
e = WorkplaceEnvironment()
cProfile.run('''
for _ in range(100):
    e.reset()
    e.step(WorkplaceAction(action_type=\"classify\", content=\"refund\"))
    e.step(WorkplaceAction(action_type=\"reply\", content=\"Thanks\"))
    e.step(WorkplaceAction(action_type=\"escalate\", content=\"no\"))
''', sort='cumulative')
"
```

### 2. Throughput Drop

**Symptom:** eps/s decreased  
**Common causes:**
- Lock contention from shared state (was fixed in N1)
- New I/O in the step path (file reads, network calls)
- Thread pool reintroduced for CPU-bound work

**Debug:**
```bash
python benchmarks/load_test.py --mode direct --episodes 1000
```

### 3. Memory Leak

**Symptom:** `test_no_memory_leak` failed with ratio > 2.0x  
**Common causes:**
- Unbounded list growth in `EpisodeState` (history not cleared on reset)
- Cached references to scenario data that grow per-episode
- Prometheus metric label explosion (high-cardinality labels)

**Debug:**
```bash
python -c "
import tracemalloc, linecache
tracemalloc.start(25)
from environment.workplace_environment import WorkplaceEnvironment
from models import WorkplaceAction
e = WorkplaceEnvironment()
for _ in range(1000):
    e.reset()
    e.step(WorkplaceAction(action_type='classify', content='refund'))
    e.step(WorkplaceAction(action_type='reply', content='thanks'))
    e.step(WorkplaceAction(action_type='escalate', content='no'))
snapshot = tracemalloc.take_snapshot()
for stat in snapshot.statistics('lineno')[:20]:
    print(stat)
"
```

### 4. CI Flakiness

**Symptom:** Tests pass locally but fail in CI  
**Common causes:**
- CI runners have different CPU (slower); adjust SLO targets in test file
- Resource contention from parallel jobs
- Cold imports (first run includes import overhead)

**Fix:** The warmup fixture (`warmed_env`) runs one episode before measuring.
If CI is consistently slower, update baseline with `workflow_dispatch`.

---

## Updating Baselines

After significant changes that intentionally affect performance:

1. Merge the PR
2. Go to **Actions → Performance Tests → Run workflow**
3. Check **"Update baseline after merge"**
4. Approve the `production` environment deployment
5. The bot commits the new `baseline.json`

---

## Architecture Decisions

| Decision | Rationale |
|---|---|
| Sequential grading (no threads) | GIL makes threads slower for CPU-bound work; N1 fix gave +30% throughput |
| Instance-owned state | Eliminates lock contention for concurrent sessions |
| Lazy metrics import | Prevents duplicate prometheus registry errors in tests |
| `time.perf_counter()` | Highest resolution timer available; monotonic |
| `tracemalloc` for memory | More accurate than RSS for Python allocations |
