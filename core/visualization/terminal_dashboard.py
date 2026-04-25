"""ASCII terminal dashboard for the improvement loop.

Pure-Python visualization using ANSI escape codes — no external plotting
libraries required.  Two entry points:

* ``print_reward_curve(evolution_history)`` — growing ASCII bar chart showing
  total, classify, reply, and escalate rewards per generation.
* ``print_strategy_diff(old_strategy, new_strategy)`` — compact diff of changed
  fields between two strategy dicts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ─── ANSI color codes ────────────────────────────────────────────────────────

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"

_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_WHITE = "\033[37m"
_MAGENTA = "\033[35m"

_BG_GREEN = "\033[42m"
_BG_YELLOW = "\033[43m"
_BG_RED = "\033[41m"

_BLOCK_FULL = "█"
_BLOCK_EMPTY = "░"

BAR_WIDTH = 10


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _color_for_value(value: float) -> str:
    """Return ANSI color code based on reward value thresholds."""
    if value >= 0.75:
        return _GREEN
    elif value >= 0.55:
        return _YELLOW
    return _RED


def _make_bar(value: float, width: int = BAR_WIDTH) -> str:
    """Build a colored ASCII bar of *width* characters for a 0.0–1.0 value."""
    clamped = max(0.0, min(1.0, value))
    filled = int(round(clamped * width))
    empty = width - filled
    color = _color_for_value(value)
    return f"{color}{_BLOCK_FULL * filled}{_DIM}{_BLOCK_EMPTY * empty}{_RESET}"


def _format_val(value: float) -> str:
    """Format a reward value with color."""
    color = _color_for_value(value)
    return f"{color}{value:.2f}{_RESET}"


# ─── print_reward_curve ──────────────────────────────────────────────────────

def print_reward_curve(evolution_history: List[Dict[str, Any]]) -> None:
    """Print an ASCII bar chart of reward metrics per generation.

    Each generation is one row with four bars: Total, Classify, Reply, Escalate.
    Bars are colored green (≥0.75), yellow (0.55–0.74), red (<0.55).

    Parameters
    ----------
    evolution_history : list[dict]
        List of generation dicts with keys: generation, mean_total,
        mean_classify, mean_reply, mean_escalate.
    """
    if not evolution_history:
        return

    # Header
    print()
    print(f"{_BOLD}{_CYAN}{'─' * 90}{_RESET}")
    print(f"{_BOLD}{_CYAN}  REWARD CURVE{_RESET}")
    print(f"{_BOLD}{_CYAN}{'─' * 90}{_RESET}")
    print(
        f"  {_DIM}{'Gen':<5} {'Total':^18}  {'Classify':^18}  "
        f"{'Reply':^18}  {'Escalate':^18}{_RESET}"
    )
    print(f"  {_DIM}{'─' * 85}{_RESET}")

    # One row per generation
    best_gen = 0
    best_total = 0.0

    for entry in evolution_history:
        gen = entry.get("generation", 0)
        total = float(entry.get("mean_total", 0))
        classify = float(entry.get("mean_classify", 0))
        reply = float(entry.get("mean_reply", 0))
        escalate = float(entry.get("mean_escalate", 0))

        if total >= best_total:
            best_total = total
            best_gen = gen

        gen_label = f"Gen {gen}"
        total_bar = f"[{_make_bar(total)}] {_format_val(total)}"
        classify_bar = f"[{_make_bar(classify)}] {_format_val(classify)}"
        reply_bar = f"[{_make_bar(reply)}] {_format_val(reply)}"
        escalate_bar = f"[{_make_bar(escalate)}] {_format_val(escalate)}"

        print(f"  {gen_label:<5} {total_bar}  {classify_bar}  {reply_bar}  {escalate_bar}")

    # Footer
    baseline_total = float(evolution_history[0].get("mean_total", 0))
    if baseline_total > 0 and len(evolution_history) > 1:
        improvement_pct = ((best_total - baseline_total) / baseline_total) * 100.0
        pct_str = f"{improvement_pct:+.1f}%"
    else:
        pct_str = "+0.0%"

    print(f"  {_DIM}{'─' * 85}{_RESET}")
    print(
        f"  {_BOLD}Best generation: {_CYAN}{best_gen}{_RESET}"
        f"{_BOLD} | Total improvement: {_GREEN}{pct_str}{_RESET}"
        f"{_BOLD} from baseline{_RESET}"
    )
    print(f"{_BOLD}{_CYAN}{'─' * 90}{_RESET}")
    print()


# ─── print_strategy_diff ─────────────────────────────────────────────────────

def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested dict into dot-separated key paths."""
    items: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, key))
        else:
            items[key] = v
    return items


def _describe_change(key: str, old_val: Any, new_val: Any) -> str:
    """Generate a human-readable description of a single field change."""
    if old_val is None:
        # New field added
        if isinstance(new_val, list):
            return f"{_GREEN}[ADDED]{_RESET}   {_BOLD}{key}{_RESET}: {len(new_val)} items"
        elif isinstance(new_val, str) and len(new_val) > 60:
            return f"{_GREEN}[ADDED]{_RESET}   {_BOLD}{key}{_RESET}: \"{new_val[:50]}...\""
        return f"{_GREEN}[ADDED]{_RESET}   {_BOLD}{key}{_RESET}: {new_val}"

    if new_val is None:
        return f"{_RED}[REMOVED]{_RESET} {_BOLD}{key}{_RESET}"

    # Changed value
    if isinstance(old_val, list) and isinstance(new_val, list):
        added = len(new_val) - len(old_val)
        if added > 0:
            return f"{_YELLOW}[CHANGED]{_RESET} {_BOLD}{key}{_RESET}: added {added} items ({len(old_val)} → {len(new_val)})"
        elif added < 0:
            return f"{_YELLOW}[CHANGED]{_RESET} {_BOLD}{key}{_RESET}: removed {abs(added)} items ({len(old_val)} → {len(new_val)})"
        return f"{_YELLOW}[CHANGED]{_RESET} {_BOLD}{key}{_RESET}: {len(new_val)} items (contents changed)"

    if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
        delta = new_val - old_val
        direction = "↑" if delta > 0 else "↓"
        return f"{_YELLOW}[CHANGED]{_RESET} {_BOLD}{key}{_RESET}: {old_val} → {new_val} ({direction}{abs(delta):.4g})"

    if isinstance(old_val, str) and isinstance(new_val, str):
        if len(old_val) > 40 or len(new_val) > 40:
            return f"{_YELLOW}[CHANGED]{_RESET} {_BOLD}{key}{_RESET}: text updated ({len(old_val)} → {len(new_val)} chars)"
        return f"{_YELLOW}[CHANGED]{_RESET} {_BOLD}{key}{_RESET}: \"{old_val}\" → \"{new_val}\""

    return f"{_YELLOW}[CHANGED]{_RESET} {_BOLD}{key}{_RESET}: {old_val!r} → {new_val!r}"


def print_strategy_diff(
    old_strategy: Optional[Dict[str, Any]],
    new_strategy: Dict[str, Any],
) -> None:
    """Print fields that changed between two strategy dicts.

    Parameters
    ----------
    old_strategy : dict | None
        Previous strategy (None for first generation).
    new_strategy : dict
        New strategy from the optimizer.
    """
    if old_strategy is None:
        old_flat: Dict[str, Any] = {}
    else:
        old_flat = _flatten_dict(old_strategy)

    new_flat = _flatten_dict(new_strategy)

    all_keys = sorted(set(list(old_flat.keys()) + list(new_flat.keys())))

    changes: List[str] = []
    for key in all_keys:
        # Skip internal/reasoning fields
        if key in ("reasoning",):
            continue
        old_val = old_flat.get(key)
        new_val = new_flat.get(key)
        if old_val != new_val:
            changes.append(_describe_change(key, old_val, new_val))

    if not changes:
        print(f"  {_DIM}(no strategy changes){_RESET}")
        return

    print(f"\n  {_BOLD}{_MAGENTA}Strategy Diff ({len(changes)} change{'s' if len(changes) != 1 else ''}){_RESET}")
    print(f"  {_DIM}{'─' * 70}{_RESET}")
    for change in changes:
        print(f"  {change}")
    print(f"  {_DIM}{'─' * 70}{_RESET}")


# ─── print_business_summary ──────────────────────────────────────────────────

def _delta_arrow(delta: float) -> str:
    """Return a colored arrow string for a delta value."""
    if delta > 0.001:
        return f"{_GREEN}▲{_RESET}"
    elif delta < -0.001:
        return f"{_RED}▼{_RESET}"
    return f"{_DIM}─{_RESET}"


def _delta_color(delta: float) -> str:
    """Return ANSI color code based on delta direction."""
    if delta > 0.001:
        return _GREEN
    elif delta < -0.001:
        return _RED
    return _YELLOW


def print_business_summary(
    baseline_memory: Any,
    current_memory: Any,
    generation: int,
) -> None:
    """Print a business-impact summary that non-technical judges remember.

    Translates raw reward deltas into language that resonates:
    categorization accuracy, customer satisfaction proxy, escalation
    cost savings.

    Parameters
    ----------
    baseline_memory : RewardMemory
        The baseline evaluation memory.
    current_memory : RewardMemory
        The current generation's evaluation memory.
    generation : int
        Current generation number.
    """
    # Compute means safely
    def _mean(records: list, attr: str) -> float:
        vals = [getattr(r, attr, 0.0) for r in records]
        return (sum(vals) / len(vals)) if vals else 0.0

    b_classify = _mean(baseline_memory.records, "classify_reward")
    c_classify = _mean(current_memory.records, "classify_reward")
    b_reply = _mean(baseline_memory.records, "reply_reward")
    c_reply = _mean(current_memory.records, "reply_reward")
    b_escalate = _mean(baseline_memory.records, "escalate_reward")
    c_escalate = _mean(current_memory.records, "escalate_reward")

    classify_pct_b = b_classify * 100
    classify_pct_c = c_classify * 100
    classify_delta = classify_pct_c - classify_pct_b

    reply_delta = c_reply - b_reply

    escalate_pct_b = b_escalate * 100
    escalate_pct_c = c_escalate * 100
    escalate_delta = c_escalate - b_escalate

    # Cost model: each incorrect escalation costs $150 in real systems
    cost_reduction = round(escalate_delta * 150 * 1000)

    # ── Render ────────────────────────────────────────────────────────────
    print(f"\n  {_BOLD}{_CYAN}╔{'═' * 72}╗{_RESET}")
    print(f"  {_BOLD}{_CYAN}║{_RESET}  {_BOLD}📊 BUSINESS IMPACT — Generation {generation}{' ' * (39 - len(str(generation)))}{_CYAN}║{_RESET}")
    print(f"  {_BOLD}{_CYAN}╠{'═' * 72}╣{_RESET}")

    # 1) Classification accuracy
    cls_arrow = _delta_arrow(classify_delta)
    cls_color = _delta_color(classify_delta)
    print(
        f"  {_BOLD}{_CYAN}║{_RESET}  {_BOLD}📧 Email categorization accuracy:{_RESET}"
        f"  {classify_pct_b:.1f}% → {cls_color}{classify_pct_c:.1f}%{_RESET}"
        f"  ({cls_arrow} {cls_color}{classify_delta:+.1f}%{_RESET})"
        f"{' ' * max(1, 16 - len(f'{classify_delta:+.1f}'))}{_CYAN}║{_RESET}"
    )

    # 2) Reply quality
    rpl_arrow = _delta_arrow(reply_delta)
    rpl_color = _delta_color(reply_delta)
    print(
        f"  {_BOLD}{_CYAN}║{_RESET}  {_BOLD}💬 Reply quality score:{_RESET}"
        f"  {b_reply:.3f} → {rpl_color}{c_reply:.3f}{_RESET}"
        f"  {rpl_arrow}  — customer satisfaction proxy"
        f"{' ' * 7}{_CYAN}║{_RESET}"
    )

    # 3) Escalation accuracy
    esc_arrow = _delta_arrow(escalate_delta)
    esc_color = _delta_color(escalate_delta)
    print(
        f"  {_BOLD}{_CYAN}║{_RESET}  {_BOLD}🚨 Escalation accuracy:{_RESET}"
        f"  {escalate_pct_b:.1f}% → {esc_color}{escalate_pct_c:.1f}%{_RESET}"
        f"  {esc_arrow}  — incorrect escalations cost $150 each"
        f"  {_CYAN}║{_RESET}"
    )

    # Divider
    print(f"  {_BOLD}{_CYAN}╠{'═' * 72}╣{_RESET}")

    # 4) Bottom line: cost reduction
    if cost_reduction >= 0:
        cost_color = _GREEN
        cost_label = "savings"
    else:
        cost_color = _RED
        cost_label = "added cost"

    print(
        f"  {_BOLD}{_CYAN}║{_RESET}  {_BOLD}💰 Estimated {cost_label} per 1,000 emails:"
        f"  {cost_color}${abs(cost_reduction):,}{_RESET}"
        f"{' ' * max(1, 30 - len(f'${abs(cost_reduction):,}'))}{_CYAN}║{_RESET}"
    )

    print(f"  {_BOLD}{_CYAN}╚{'═' * 72}╝{_RESET}")
    print()


__all__ = ["print_reward_curve", "print_strategy_diff", "print_business_summary"]

