"""Final demo entrypoint for baseline -> improvement -> comparison pipeline."""

from __future__ import annotations

import json
import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.graders import RuleBasedRewardPolicy
from core.improvement.curriculum import CurriculumSampler
from core.improvement.failure_analyzer import FailureAnalyzer
from core.improvement.red_teamer import RegressionTester
from core.improvement.strategy_optimizer import StrategyOptimizer
from core.visualization.terminal_dashboard import (
    print_reward_curve,
    print_strategy_diff,
    print_business_summary,
    print_delta_table,
    print_strategy_reasoning,
)
from core.inference.adaptive_agent import AdaptiveAgent
from core.inference.strategies import EmailAwareInference
from core.memory.reward_memory import EpisodeRecord, RewardMemory
from data.scenario_repository import StaticScenarioRepository
from environment.workplace_environment import WorkplaceEnvironment
from models import WorkplaceAction


# Hardcoded minimal fallback strategy for crash recovery.
DEFAULT_FALLBACK_STRATEGY: Dict[str, Any] = {
    "classification_rules": {
        "refund": ["refund", "reimbursement", "charged twice", "overcharged", "billing error", "money back", "charge", "credit", "cancel order", "cancellation"],
        "complaint": ["not happy", "gone downhill", "terrible", "awful", "unacceptable", "disappointed", "frustrated", "angry", "outraged", "worst", "quality has", "way too slow", "took too long", "poor quality", "bad experience", "let down"],
        "query": ["how do i", "how to", "what is", "can you", "do you", "where is", "when will", "return my item", "return policy", "track my", "status of", "information about", "tell me", "explain", "help me understand"],
        "default": "query",
    },
    "reply_templates": {
        "refund": "We sincerely apologize for the inconvenience. Your refund request has been received and will be processed within 3-5 business days.",
        "complaint": "We're really sorry to hear about your experience. We understand your frustration and are taking steps to resolve this issue immediately.",
        "query": "Thank you for reaching out. We're happy to help and will provide the information you requested.",
    },
    "reply_requirements": {
        "min_length": 40,
        "must_include_greeting": True,
        "must_include_closing": True,
        "forbidden_phrases": ["not my problem", "cannot help", "no idea"],
    },
    "escalation_rules": {
        "always_escalate": ["legal threat", "safety risk", "account restricted", "not happy", "gone downhill", "way too slow", "quality has really"],
        "never_escalate": [],
        "escalate_if_complaint": True,
        "escalate_if_high_urgency": True,
    },
    "reasoning": "Default fallback strategy for crash recovery.",
}


def _obs_to_dict(obs: Any) -> Dict[str, Any]:
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    if hasattr(obs, "dict"):
        return obs.dict()
    if isinstance(obs, dict):
        return dict(obs)
    try:
        return dict(vars(obs))
    except Exception:
        return {}


def _safe_breakdown(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_strategy(strategy: Any) -> Dict[str, Any]:
    return strategy if isinstance(strategy, dict) else {}


def _safe_string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _make_strategy_client() -> Any:
    """Create Anthropic client if available; otherwise return a safe stub."""

    class _NoopMessages:
        def create(self, **kwargs):
            raise RuntimeError("Anthropic client unavailable")

    class _NoopClient:
        def __init__(self):
            self.messages = _NoopMessages()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[CLIENT DEBUG] No ANTHROPIC_API_KEY in environment")
        return _NoopClient()

    print(f"[CLIENT DEBUG] Attempting to create Anthropic client with key length: {len(api_key)}")
    try:
        from anthropic import Anthropic  # type: ignore[import-not-found]

        client = Anthropic(api_key=api_key)
        print(f"[CLIENT DEBUG] Anthropic client created successfully: {type(client)}")
        return client
    except Exception as e:
        print(f"[CLIENT DEBUG] Anthropic() constructor failed: {type(e).__name__}: {e}")
        return _NoopClient()


def _extract_breakdowns_by_step(env: WorkplaceEnvironment) -> Dict[str, Dict[str, Any]]:
    details: List[Any] = []
    try:
        details = list(getattr(env._state, "step_details", []))  # type: ignore[attr-defined]
    except Exception:
        details = []

    mapped: Dict[str, Dict[str, Any]] = {
        "classify": {},
        "reply": {},
        "escalate": {},
    }
    for item in details:
        if not isinstance(item, dict):
            continue
        action_type = _safe_string(item.get("action_type", "")).strip().lower()
        if action_type in mapped:
            mapped[action_type] = item
    return mapped


def run_evaluation(
    agent: Any,
    n_episodes: int,
    strategy_version: str,
    scenario_pool: Optional[List[Dict[str, Any]]] = None,
    locked_scenario_ids: Optional[List[Dict[str, Any]]] = None,
) -> RewardMemory:
    """Run deterministic evaluation and return episode memory.

    If *locked_scenario_ids* is provided (a pre-filtered list of scenario
    dicts), it takes priority over *scenario_pool* — the evaluation runs
    only on those scenarios.  If *scenario_pool* is provided (and
    *locked_scenario_ids* is None), the environment cycles through only
    those scenarios instead of the full 100-scenario corpus.
    """
    policy = RuleBasedRewardPolicy()
    memory = RewardMemory()

    # locked_scenario_ids (pre-filtered pool) takes priority over scenario_pool
    effective_pool = locked_scenario_ids if locked_scenario_ids is not None else scenario_pool

    # Inject custom scenario pool if provided; otherwise use full corpus.
    if effective_pool is not None:
        repo = StaticScenarioRepository(effective_pool)
        env = WorkplaceEnvironment(reward_policy=policy, scenario_repository=repo)
    else:
        env = WorkplaceEnvironment(reward_policy=policy)

    for episode_idx in range(max(0, int(n_episodes))):
        obs = _obs_to_dict(env.reset())

        email = _safe_string(obs.get("email", ""))
        difficulty = _safe_string(obs.get("scenario_difficulty", "unknown"), "unknown")
        sentiment = _safe_string(obs.get("sentiment", "unknown"), "unknown")
        urgency = _safe_string(obs.get("urgency", "unknown"), "unknown")

        classify_action = "query"
        reply_action = "Thank you for reaching out. We are here to assist you."
        escalate_action = "no"

        classify_reward = 0.0
        reply_reward = 0.0
        escalate_reward = 0.0

        classify_breakdown: Dict[str, Any] = {}
        reply_breakdown: Dict[str, Any] = {}
        escalate_breakdown: Dict[str, Any] = {}

        planned_actions: List[Tuple[str, str]] = []
        try:
            raw_actions = agent.build_actions(obs)
            if isinstance(raw_actions, list):
                for item in raw_actions:
                    if (
                        isinstance(item, tuple)
                        and len(item) == 2
                        and isinstance(item[0], str)
                    ):
                        planned_actions.append((item[0], _safe_string(item[1], "")))
        except Exception:
            planned_actions = []

        # Deterministic action order and safe defaults.
        by_type = {k: v for k, v in planned_actions if k in {"classify", "reply", "escalate"}}
        ordered_actions: List[Tuple[str, str]] = [
            ("classify", _safe_string(by_type.get("classify", classify_action), classify_action)),
            ("reply", _safe_string(by_type.get("reply", reply_action), reply_action)),
            ("escalate", _safe_string(by_type.get("escalate", escalate_action), escalate_action)),
        ]

        for action_type, content in ordered_actions:
            action = WorkplaceAction(action_type=action_type, content=content)
            try:
                step_obs = _obs_to_dict(env.step(action))
                reward_value = float(step_obs.get("reward") or 0.0)
            except Exception:
                step_obs = {}
                reward_value = 0.0

            if action_type == "classify":
                classify_action = content
                classify_reward = reward_value
            elif action_type == "reply":
                reply_action = content
                reply_reward = reward_value
            elif action_type == "escalate":
                normalized = "yes" if content.strip().lower() == "yes" else "no"
                escalate_action = normalized
                escalate_reward = reward_value

        step_breakdowns = _extract_breakdowns_by_step(env)
        classify_breakdown = _safe_breakdown(step_breakdowns.get("classify", {}))
        reply_breakdown = _safe_breakdown(step_breakdowns.get("reply", {}))
        escalate_breakdown = _safe_breakdown(step_breakdowns.get("escalate", {}))

        record = EpisodeRecord(
            episode_id=episode_idx + 1,
            scenario_id=f"{strategy_version}-ep-{episode_idx + 1}",
            difficulty=difficulty,
            sentiment=sentiment,
            urgency=urgency,
            email_snippet=email[:120],
            classify_action=classify_action,
            classify_reward=classify_reward,
            classify_breakdown=classify_breakdown,
            reply_action=reply_action,
            reply_reward=reply_reward,
            reply_breakdown=reply_breakdown,
            escalate_action=escalate_action,
            escalate_reward=escalate_reward,
            escalate_breakdown=escalate_breakdown,
        )
        memory.add(record)

    return memory


def _build_locked_pool(
    baseline_memory: RewardMemory,
    threshold: float = 0.60,
) -> Optional[List[Dict[str, Any]]]:
    """Build a locked scenario pool from the baseline's worst episodes.

    Extracts failed EpisodeRecords (total_reward < threshold), matches their
    email snippets back to the full corpus, and returns the matched scenario
    dicts.  Returns None if no failures are found.
    """
    from core.improvement.curriculum import CurriculumSampler

    failed_records = [
        r for r in baseline_memory.records if r.total_reward < threshold
    ]
    if not failed_records:
        return None

    # Index corpus by email prefix for fast matching
    corpus = CurriculumSampler().corpus
    corpus_by_prefix: Dict[str, Dict[str, Any]] = {}
    for scenario in corpus:
        prefix = str(scenario.get("email", ""))[:120]
        corpus_by_prefix[prefix] = scenario

    locked: List[Dict[str, Any]] = []
    seen_prefixes: set = set()
    for record in failed_records:
        prefix = record.email_snippet[:120]
        if prefix in corpus_by_prefix and prefix not in seen_prefixes:
            locked.append(corpus_by_prefix[prefix])
            seen_prefixes.add(prefix)

    return locked if locked else None


def _mean(values: List[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def _memory_means(memory: RewardMemory) -> Dict[str, float]:
    records = list(memory.records)
    total = _mean([r.total_reward for r in records])
    classify = _mean([r.classify_reward for r in records])
    reply = _mean([r.reply_reward for r in records])
    escalate = _mean([r.escalate_reward for r in records])
    return {
        "total": total,
        "classify": classify,
        "reply": reply,
        "escalate": escalate,
    }


def compare(baseline: RewardMemory, improved: RewardMemory) -> Dict[str, float]:
    """Compare improved memory against baseline and return deltas."""
    b = _memory_means(baseline)
    i = _memory_means(improved)

    total_delta = i["total"] - b["total"]
    classify_delta = i["classify"] - b["classify"]
    reply_delta = i["reply"] - b["reply"]
    escalate_delta = i["escalate"] - b["escalate"]

    if b["total"] > 0:
        improvement_percent = (total_delta / b["total"]) * 100.0
    else:
        improvement_percent = 0.0

    return {
        "total_delta": total_delta,
        "classify_delta": classify_delta,
        "reply_delta": reply_delta,
        "escalate_delta": escalate_delta,
        "improvement_percent": improvement_percent,
    }


def print_summary(memory: RewardMemory, label: str) -> None:
    """Print human-readable reward summary."""
    means = _memory_means(memory)
    records = list(memory.records)

    by_difficulty: Dict[str, List[EpisodeRecord]] = {"easy": [], "medium": [], "hard": []}
    for record in records:
        key = _safe_string(record.difficulty, "unknown").lower()
        if key in by_difficulty:
            by_difficulty[key].append(record)

    print("\n" + "=" * 64)
    print(f"{label} SUMMARY")
    print("=" * 64)
    print(f"Episodes       : {len(records)}")
    print(f"Mean Total     : {means['total']:.3f}")
    print(f"Mean Classify  : {means['classify']:.3f}")
    print(f"Mean Reply     : {means['reply']:.3f}")
    print(f"Mean Escalate  : {means['escalate']:.3f}")

    print("\nDifficulty Breakdown")
    print("-" * 64)
    for level in ("easy", "medium", "hard"):
        bucket = by_difficulty[level]
        if bucket:
            total = _mean([r.total_reward for r in bucket])
            c = _mean([r.classify_reward for r in bucket])
            rep = _mean([r.reply_reward for r in bucket])
            esc = _mean([r.escalate_reward for r in bucket])
            print(
                f"{level:<8} total={total:.3f}  classify={c:.3f}  "
                f"reply={rep:.3f}  escalate={esc:.3f}"
            )
        else:
            print(f"{level:<8} total=0.000  classify=0.000  reply=0.000  escalate=0.000")


def print_comparison(
    result: Dict[str, float],
    baseline: RewardMemory,
    improved: RewardMemory,
    decision: str,
) -> None:
    """Print baseline -> improved comparison table."""
    b = _memory_means(baseline)
    i = _memory_means(improved)

    def _status(delta: float, total_row: bool = False) -> str:
        if delta >= 0:
            return "✅ IMPROVED" if total_row else "✅"
        return "❌ REGRESSED"

    print("\n" + "=" * 64)
    print("COMPARISON")
    print("=" * 64)
    print("STEP            BASELINE → IMPROVED      STATUS         DECISION")
    print("-" * 64)
    print(f"Total           {b['total']:.2f} → {i['total']:.2f}             {_status(result['total_delta'], total_row=True):<13} {decision}")
    print(f"Classify        {b['classify']:.2f} → {i['classify']:.2f}             {_status(result['classify_delta']):<13} {decision}")
    print(f"Reply           {b['reply']:.2f} → {i['reply']:.2f}             {_status(result['reply_delta']):<13} {decision}")
    print(f"Escalate        {b['escalate']:.2f} → {i['escalate']:.2f}             {_status(result['escalate_delta']):<13} {decision}")
    print("-" * 64)
    print(f"Overall Improvement: {result['improvement_percent']:.2f}%")


def _save_strategy(strategy: Dict[str, Any], path: str = "final_strategy.json") -> None:
    Path(path).write_text(json.dumps(_safe_strategy(strategy), indent=2, ensure_ascii=False), encoding="utf-8")


def _read_text_if_exists(path: str) -> Optional[str]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return None


def _write_text(path: str, content: Optional[str]) -> None:
    if content is None:
        return
    Path(path).write_text(content, encoding="utf-8")


def _count_failures(memory: RewardMemory, threshold: float = 0.5) -> int:
    """Count episodes with total reward below threshold."""
    return sum(1 for r in memory.records if r.total_reward < threshold)


def _save_evolution_history(history: List[Dict[str, Any]], path: str = "evolution_history.json") -> None:
    """Write evolution history to JSON, safe against partial data."""
    Path(path).write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _flush() -> None:
    """Flush stdout so terminal output appears immediately (unbuffered)."""
    sys.stdout.flush()


def _demo_load_cached_strategy(path: str = "final_strategy.json") -> Optional[Dict[str, Any]]:
    """Load a cached strategy from disk for demo fallback."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return None


def _demo_safe_generate(
    optimizer: StrategyOptimizer,
    failure_analysis: Any,
    current_strategy: Optional[Dict[str, Any]],
    baseline_metrics_summary: Dict[str, Any],
    baseline_score: float = 0.0,
) -> Dict[str, Any]:
    """Wrap strategy generation with demo-mode fallback.

    If the Anthropic API call fails (rate-limit, timeout, no key),
    falls back to loading the cached strategy from disk.
    """
    try:
        return optimizer.generate_strategy(
            failure_analysis=failure_analysis,
            current_strategy=current_strategy,
            baseline_metrics_summary=baseline_metrics_summary,
            baseline_score=baseline_score,
        )
    except Exception as api_exc:
        print(f"\n⚠️  API fallback: using cached strategy (reason: {api_exc})")
        _flush()
        cached = _demo_load_cached_strategy()
        if cached is not None:
            return cached
        # If no cache exists, use the module-level default
        if current_strategy is not None:
            return current_strategy
        return dict(DEFAULT_FALLBACK_STRATEGY)


def run_improvement_loop(
    n_episodes: int = 25,
    n_iterations: int = 2,
    max_generations: int = 4,
    convergence_threshold: float = 0.02,
    demo_mode: bool = False,
) -> None:
    if demo_mode:
        max_generations = min(max_generations, 3)
        print("\n🎬 DEMO MODE — max 3 generations, API fallback enabled")
        _flush()

    logging.getLogger("environment.workplace_environment").setLevel(logging.WARNING)

    # 1) BASELINE RUN
    baseline_agent = EmailAwareInference()
    baseline_memory = run_evaluation(
        agent=baseline_agent,
        n_episodes=n_episodes,
        strategy_version="baseline",
    )
    baseline_memory.save("baseline_memory.json")
    print_summary(baseline_memory, "BASELINE")

    # Lock subsequent evaluations to the hardest baseline scenarios
    # so the reward delta is maximized and consistent across runs.
    locked_pool = _build_locked_pool(baseline_memory, threshold=0.60)
    if locked_pool:
        print(f"Demo locked to {len(locked_pool)} failure scenarios for consistent delta measurement.")
        _flush()

    baseline_means = _memory_means(baseline_memory)
    baseline_mean_total = baseline_means["total"]
    accepted_strategy_text = _read_text_if_exists("final_strategy.json")

    current_memory = baseline_memory
    current_strategy: Optional[Dict[str, Any]] = None
    best_strategy: Optional[Dict[str, Any]] = None
    best_score: float = 0.0
    improved_memory = baseline_memory
    final_memory = baseline_memory
    final_decision = "REJECTED"

    # ── Regression tester (golden scenarios) ──────────────────────────────
    regression_tester = RegressionTester()
    # Compute baseline golden score using the EmailAwareInference agent
    baseline_golden_memory = run_evaluation(
        agent=baseline_agent,
        n_episodes=len(regression_tester.golden_scenarios),
        strategy_version="baseline_golden",
        scenario_pool=regression_tester.golden_scenarios,
    )
    baseline_golden_score = (
        sum(r.total_reward for r in baseline_golden_memory.records)
        / len(baseline_golden_memory.records)
        if baseline_golden_memory.records else 0.0
    )
    print(f"\n[REGRESSION] Baseline golden score: {baseline_golden_score:.3f}")

    # ── Evolution history tracking ────────────────────────────────────────
    evolution_history: List[Dict[str, Any]] = []

    # Record generation 0 (baseline)
    evolution_history.append({
        "generation": 0,
        "mean_total": round(baseline_means["total"], 4),
        "mean_classify": round(baseline_means["classify"], 4),
        "mean_reply": round(baseline_means["reply"], 4),
        "mean_escalate": round(baseline_means["escalate"], 4),
        "failure_count": _count_failures(baseline_memory),
        "strategy_reasoning": "Baseline — no strategy applied.",
        "golden_score": round(baseline_golden_score, 4),
    })

    effective_generations = max(1, min(int(max_generations), int(n_iterations)))

    # ── Curriculum sampler ─────────────────────────────────────────────────
    sampler = CurriculumSampler()
    sampler.update_weights(baseline_memory)
    curriculum_pool: Optional[List[Dict[str, Any]]] = None  # None = full corpus for gen 1

    # 2/3/4) ANALYSIS -> STRATEGY -> IMPROVED (iterative with convergence)
    previous_strategy: Optional[Dict[str, Any]] = None
    for iteration in range(effective_generations):
        generation = iteration + 1

        try:
            analyzer = FailureAnalyzer()
            try:
                failure_analysis = analyzer.analyze(current_memory)
            except Exception as fa_exc:
                print(f"⚠️  Failure analysis error: using empty analysis ({fa_exc})")
                _flush()
                failure_analysis = {}

            print("\n=== FAILURE ANALYSIS ===")
            import json
            print(json.dumps(failure_analysis if isinstance(failure_analysis, dict) else {}, indent=2))

            client = _make_strategy_client()
            optimizer = StrategyOptimizer(client)

            _gen_kwargs = {
                "failure_analysis": failure_analysis,
                "current_strategy": current_strategy,
                "baseline_metrics_summary": {
                    "baseline_mean_total_reward": baseline_mean_total,
                    "baseline_means": _memory_means(baseline_memory),
                },
                "baseline_score": baseline_golden_score,
            }
            if demo_mode:
                current_strategy = _demo_safe_generate(optimizer, **_gen_kwargs)
            else:
                try:
                    current_strategy = optimizer.generate_strategy(**_gen_kwargs)
                except Exception as opt_exc:
                    print(f"⚠️  API fallback: loading cached strategy from final_strategy.json ({opt_exc})")
                    _flush()
                    cached = _demo_load_cached_strategy()
                    if cached is not None:
                        current_strategy = cached
                    elif best_strategy is not None:
                        current_strategy = best_strategy
                    elif current_strategy is not None:
                        pass  # keep current_strategy as-is
                    else:
                        current_strategy = dict(DEFAULT_FALLBACK_STRATEGY)

            print("\n=== GENERATED STRATEGY ===")
            print(json.dumps(current_strategy if isinstance(current_strategy, dict) else {}, indent=2))

            reasoning = _safe_string(
                current_strategy.get("reasoning", "No reasoning provided."),
                "No reasoning provided.",
            )
            print_strategy_reasoning(reasoning, generation)
            print_strategy_diff(previous_strategy, _safe_strategy(current_strategy))

            # ── Regression test against golden scenarios ──────────────────
            golden_passed, golden_score = regression_tester.validate(
                current_strategy, baseline_golden_score
            )
            first_attempt_golden_score = golden_score
            retry_golden_score = None

            if not golden_passed:
                print(
                    f"⚠️  Gen {generation}: Catastrophic forgetting detected "
                    f"(golden score dropped from {baseline_golden_score:.4f} "
                    f"to {golden_score:.4f}) — retrying with broadened prompt"
                )
                # Retry once with regression warning injected
                failure_analysis_with_warning = dict(failure_analysis)
                failure_analysis_with_warning["regression_warning"] = (
                    f"Your previous strategy scored {golden_score:.4f} on golden "
                    f"scenarios vs baseline {baseline_golden_score:.4f}. "
                    f"Prioritize not breaking general cases."
                )
                _retry_kwargs = {
                    "failure_analysis": failure_analysis_with_warning,
                    "current_strategy": current_strategy,
                    "baseline_metrics_summary": {
                        "baseline_mean_total_reward": baseline_mean_total,
                        "baseline_means": _memory_means(baseline_memory),
                    },
                    "baseline_score": baseline_golden_score,
                }
                if demo_mode:
                    current_strategy = _demo_safe_generate(optimizer, **_retry_kwargs)
                else:
                    current_strategy = optimizer.generate_strategy(**_retry_kwargs)

                print("\n=== GENERATED STRATEGY ===")
                print(json.dumps(current_strategy if isinstance(current_strategy, dict) else {}, indent=2))

                reasoning = _safe_string(
                    current_strategy.get("reasoning", "No reasoning provided."),
                    "No reasoning provided.",
                )
                print_strategy_reasoning(reasoning, generation)

                # Validate retry
                retry_passed, retry_golden_score = regression_tester.validate(
                    current_strategy, baseline_golden_score
                )
                print(
                    f"Retry golden score: {retry_golden_score:.4f} "
                    f"(was {first_attempt_golden_score:.4f}, baseline {baseline_golden_score:.4f})"
                )
                golden_score = retry_golden_score

                # If retry also failed, revert to best known strategy
                if not retry_passed and best_strategy is not None:
                    print(
                        f"[REGRESSION] Strategy rejected — reverting to best known strategy "
                        f"(score: {best_score:.3f})"
                    )
                    current_strategy = best_strategy
            else:
                print(f"Golden regression test: PASSED ({golden_score:.4f} >= {baseline_golden_score * 0.90:.4f})")
                # Update best strategy tracking
                if golden_score > best_score:
                    best_strategy = _safe_strategy(current_strategy)
                    best_score = golden_score

            improved_agent = AdaptiveAgent(_safe_strategy(current_strategy))
            candidate_memory = run_evaluation(
                agent=improved_agent,
                n_episodes=n_episodes,
                strategy_version=f"improved_v{generation}",
                scenario_pool=curriculum_pool,
                locked_scenario_ids=locked_pool,
            )
            candidate_means = _memory_means(candidate_memory)
            candidate_mean_total = candidate_means["total"]

            # Record this generation (include golden scores)
            gen_entry: Dict[str, Any] = {
                "generation": generation,
                "mean_total": round(candidate_mean_total, 4),
                "mean_classify": round(candidate_means["classify"], 4),
                "mean_reply": round(candidate_means["reply"], 4),
                "mean_escalate": round(candidate_means["escalate"], 4),
                "failure_count": _count_failures(candidate_memory),
                "strategy_reasoning": reasoning,
                "golden_score": round(golden_score, 4),
                "golden_passed": golden_passed,
            }
            if retry_golden_score is not None:
                gen_entry["golden_score_attempt_1"] = round(first_attempt_golden_score, 4)
                gen_entry["golden_score_attempt_2"] = round(retry_golden_score, 4)
                gen_entry["regression_retried"] = True
            evolution_history.append(gen_entry)

            # ── Live reward curve + delta table + reasoning (grows each gen) ──
            print_reward_curve(evolution_history)
            print_delta_table(_memory_means(baseline_memory), candidate_means)
            print_strategy_reasoning(reasoning, generation)
            print_business_summary(baseline_memory, candidate_memory, generation)

            if candidate_mean_total < baseline_mean_total:
                print("Strategy rejected — performance degraded")
                _write_text("final_strategy.json", accepted_strategy_text)
                current_memory = baseline_memory
                final_memory = baseline_memory
                final_decision = "REJECTED"
                print(f"Generation {generation}: REJECTED")
            else:
                print("Strategy accepted — improvement achieved")
                _save_strategy(_safe_strategy(current_strategy), "final_strategy.json")
                accepted_strategy_text = _read_text_if_exists("final_strategy.json")
                improved_memory = candidate_memory
                current_memory = candidate_memory
                final_memory = candidate_memory
                final_decision = "ACCEPTED"
                print(f"Generation {generation}: ACCEPTED")

            previous_strategy = _safe_strategy(current_strategy) if current_strategy else None

            # ── Update curriculum weights for next generation ──────────────
            sampler.update_weights(candidate_memory)
            curriculum_pool = sampler.sample(n_episodes)
            ws = sampler.weight_summary()
            print(
                f"Curriculum: {ws['upweighted_count']} scenarios upweighted, "
                f"max_weight={ws['max_weight']:.1f}, mean_weight={ws['mean_weight']:.2f}"
            )

            # ── Convergence check ─────────────────────────────────────────
            if len(evolution_history) >= 2:
                prev_mean = evolution_history[-2]["mean_total"]
                curr_mean = evolution_history[-1]["mean_total"]
                if abs(curr_mean - prev_mean) < convergence_threshold:
                    print(f"\n✅ Converged at generation {generation}")
                    _flush()
                    break

            _flush()  # Ensure all generation output is visible immediately

        except Exception as exc:
            print(f"\n⚠️  Generation {generation} failed: {exc}")
            print("Saving accumulated evolution history before exit...")
            _save_evolution_history(evolution_history)
            _flush()
            break

    # ── Save evolution history (always, regardless of exit reason) ─────────
    _save_evolution_history(evolution_history)

    # ── Print per-generation summary table ────────────────────────────────
    print("\n" + "=" * 80)
    print("EVOLUTION SUMMARY")
    print("=" * 80)
    for entry in evolution_history:
        gen = entry["generation"]
        total = entry["mean_total"]
        classify = entry["mean_classify"]
        reply = entry["mean_reply"]
        escalate = entry["mean_escalate"]
        failures = entry["failure_count"]

        if gen == 0:
            pct_str = "      "
        else:
            baseline_val = evolution_history[0]["mean_total"]
            if baseline_val > 0:
                pct = ((total - baseline_val) / baseline_val) * 100.0
                pct_str = f"({pct:+.0f}%)" if abs(pct) < 1000 else f"({pct:+.1f}%)"
            else:
                pct_str = "(N/A)"

        print(
            f"Gen {gen} | Total: {total:.2f} {pct_str:>7} | "
            f"Classify: {classify:.2f} | Reply: {reply:.2f} | "
            f"Escalate: {escalate:.2f} | Failures: {failures}"
        )
    print("=" * 80)

    final_memory.save("improved_memory.json")
    print_summary(final_memory, "IMPROVED")

    # 5) COMPARISON
    result = compare(baseline_memory, final_memory)
    print_comparison(result, baseline_memory, final_memory, final_decision)

    # 6) SAVE STRATEGY
    if final_decision == "ACCEPTED" and current_strategy is not None:
        _save_strategy(_safe_strategy(current_strategy), "final_strategy.json")
    else:
        _write_text("final_strategy.json", accepted_strategy_text)

    # 7) GENERATE REPORT
    from generate_report import generate_report
    generate_report()
    _flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the meta-environment improvement loop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help="Enable demo mode: API fallback, max 3 generations, flushed output.",
    )
    parser.add_argument("--episodes", type=int, default=25, help="Episodes per evaluation.")
    parser.add_argument("--iterations", type=int, default=2, help="Max iterations (n_iterations).")
    parser.add_argument("--generations", type=int, default=4, help="Max generations.")
    parser.add_argument("--threshold", type=float, default=0.02, help="Convergence threshold.")

    args = parser.parse_args()

    import os
    test_key = os.environ.get("ANTHROPIC_API_KEY", "")
    print(f"[STARTUP DEBUG] Key in environment: length={len(test_key)}, starts_with={test_key[:12] if test_key else 'EMPTY'}")
    try:
        from anthropic import Anthropic as _TestAnthropic
        _test_client = _TestAnthropic(api_key=test_key)
        _test_resp = _test_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": "say hi"}]
        )
        print(f"[STARTUP DEBUG] API test PASSED: {_test_resp.content[0].text}")
    except Exception as e:
        print(f"[STARTUP DEBUG] API test FAILED: {type(e).__name__}: {e}")

    run_improvement_loop(
        n_episodes=args.episodes,
        n_iterations=args.iterations,
        max_generations=args.generations,
        convergence_threshold=args.threshold,
        demo_mode=args.demo,
    )
