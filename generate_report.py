"""Generate RESULTS.md from improvement loop artifacts.

Reads evolution_history.json, baseline_memory.json, improved_memory.json,
and final_strategy.json from disk, then writes a clean GitHub-rendered
markdown report summarizing the entire improvement run.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: str) -> Any:
    """Load JSON from disk, returning None on failure."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _mean_reward(records: List[Dict[str, Any]], key: str) -> float:
    """Compute mean of a reward field across episode records."""
    vals = [float(r.get(key, 0)) for r in records]
    return (sum(vals) / len(vals)) if vals else 0.0


def _total_reward(record: Dict[str, Any]) -> float:
    """Sum classify + reply + escalate for one record."""
    return (
        float(record.get("classify_reward", 0))
        + float(record.get("reply_reward", 0))
        + float(record.get("escalate_reward", 0))
    )


def generate_report(output_path: str = "RESULTS.md") -> str:
    """Generate RESULTS.md and return the markdown content.

    Reads from:
      - evolution_history.json
      - baseline_memory.json
      - improved_memory.json
      - final_strategy.json

    Returns
    -------
    str
        The generated markdown content.
    """
    # ── Load artifacts ────────────────────────────────────────────────────
    evolution = _load_json("evolution_history.json") or []
    baseline_data = _load_json("baseline_memory.json") or {}
    improved_data = _load_json("improved_memory.json") or {}
    strategy = _load_json("final_strategy.json") or {}

    baseline_records = baseline_data.get("records", [])
    improved_records = improved_data.get("records", [])

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: List[str] = []

    # ── Header ────────────────────────────────────────────────────────────
    lines.append("# 📊 Meta-Environment: Improvement Run Results")
    lines.append("")
    lines.append(f"> Generated: {timestamp}")
    lines.append("")

    # ── 1. System Overview ────────────────────────────────────────────────
    lines.append("## 1. System Overview")
    lines.append("")
    lines.append(
        "- **Environment**: A 3-step reinforcement learning environment simulating "
        "customer support email triage — the agent must classify the email, compose "
        "a reply, and decide whether to escalate to a human."
    )
    lines.append(
        "- **Improvement Loop**: An iterative self-improvement pipeline that runs "
        "evaluation → failure analysis → strategy optimization → re-evaluation, "
        "with curriculum-based sampling that upweights failure scenarios and "
        "regression testing against 10 golden scenarios to prevent catastrophic forgetting."
    )
    lines.append(
        "- **Mechanism**: Each generation's failures are analyzed, a strategy "
        "optimizer proposes updated classification rules, reply templates, and "
        "escalation policies. The new strategy is validated against golden scenarios "
        "before being evaluated on the full curriculum-weighted corpus."
    )
    lines.append("")

    # ── 2. Reward Progression ─────────────────────────────────────────────
    lines.append("## 2. Reward Progression")
    lines.append("")

    if evolution:
        lines.append(
            "| Generation | Total | Classify | Reply | Escalate | Failures | Strategy Change |"
        )
        lines.append(
            "|:---:|:---:|:---:|:---:|:---:|:---:|:---|"
        )

        for entry in evolution:
            gen = entry.get("generation", "?")
            total = entry.get("mean_total", 0)
            classify = entry.get("mean_classify", 0)
            reply = entry.get("mean_reply", 0)
            escalate = entry.get("mean_escalate", 0)
            failures = entry.get("failure_count", 0)
            reasoning = entry.get("strategy_reasoning", "—")

            # Truncate reasoning for table readability
            if len(reasoning) > 80:
                reasoning = reasoning[:77] + "..."

            lines.append(
                f"| {gen} | {total:.4f} | {classify:.4f} | {reply:.4f} | "
                f"{escalate:.4f} | {failures} | {reasoning} |"
            )

        # Compute best generation
        best = max(evolution, key=lambda e: e.get("mean_total", 0))
        baseline_total = evolution[0].get("mean_total", 0)
        best_total = best.get("mean_total", 0)
        if baseline_total > 0:
            pct = ((best_total - baseline_total) / baseline_total) * 100
            lines.append("")
            lines.append(
                f"**Best generation**: {best.get('generation', '?')} "
                f"(Total: {best_total:.4f}, {pct:+.1f}% from baseline)"
            )
    else:
        lines.append("*No evolution history found.*")
    lines.append("")

    # ── 3. What the System Learned ────────────────────────────────────────
    lines.append("## 3. What the System Learned")
    lines.append("")

    gen_entries = [e for e in evolution if e.get("generation", 0) > 0]
    if gen_entries:
        for entry in gen_entries:
            gen = entry.get("generation", "?")
            reasoning = entry.get("strategy_reasoning", "No reasoning recorded.")
            retried = entry.get("regression_retried", False)
            golden = entry.get("golden_score", None)

            line = f"{gen}. **Gen {gen}**: {reasoning}"
            annotations = []
            if retried:
                annotations.append("🔄 regression retry triggered")
            if golden is not None:
                annotations.append(f"golden score: {golden:.4f}")
            if annotations:
                line += f" *({', '.join(annotations)})*"
            lines.append(line)
    else:
        lines.append("*No strategy generations recorded.*")
    lines.append("")

    # ── 4. Business Impact ────────────────────────────────────────────────
    lines.append("## 4. Business Impact")
    lines.append("")

    if baseline_records and improved_records:
        b_classify = _mean_reward(baseline_records, "classify_reward")
        c_classify = _mean_reward(improved_records, "classify_reward")
        b_reply = _mean_reward(baseline_records, "reply_reward")
        c_reply = _mean_reward(improved_records, "reply_reward")
        b_escalate = _mean_reward(baseline_records, "escalate_reward")
        c_escalate = _mean_reward(improved_records, "escalate_reward")

        classify_delta = (c_classify - b_classify) * 100
        escalate_delta = c_escalate - b_escalate
        cost_impact = round(escalate_delta * 150 * 1000)

        cls_arrow = "📈" if classify_delta >= 0 else "📉"
        rpl_arrow = "📈" if c_reply >= b_reply else "📉"
        esc_arrow = "📈" if escalate_delta >= 0 else "📉"
        cost_emoji = "💰" if cost_impact >= 0 else "⚠️"

        lines.append(
            f"| Metric | Baseline | Final | Delta |"
        )
        lines.append(
            f"|:---|:---:|:---:|:---:|"
        )
        lines.append(
            f"| {cls_arrow} Email categorization accuracy | "
            f"{b_classify * 100:.1f}% | {c_classify * 100:.1f}% | "
            f"{classify_delta:+.1f}% |"
        )
        lines.append(
            f"| {rpl_arrow} Reply quality score | "
            f"{b_reply:.3f} | {c_reply:.3f} | "
            f"{c_reply - b_reply:+.3f} |"
        )
        lines.append(
            f"| {esc_arrow} Escalation accuracy | "
            f"{b_escalate * 100:.1f}% | {c_escalate * 100:.1f}% | "
            f"{escalate_delta * 100:+.1f}% |"
        )
        lines.append("")

        if cost_impact >= 0:
            lines.append(
                f"> {cost_emoji} **Estimated savings per 1,000 emails: ${cost_impact:,}**"
            )
        else:
            lines.append(
                f"> {cost_emoji} **Estimated added cost per 1,000 emails: ${abs(cost_impact):,}**"
            )
        lines.append(">")
        lines.append(
            "> _Cost model: each incorrect escalation costs $150 in agent time "
            "and customer dissatisfaction._"
        )
    else:
        lines.append("*Insufficient data to compute business impact.*")
    lines.append("")

    # ── 5. Artifacts ──────────────────────────────────────────────────────
    lines.append("## 5. Artifacts Produced")
    lines.append("")

    artifact_files = [
        ("evolution_history.json", "Per-generation metrics, golden scores, and strategy reasoning"),
        ("baseline_memory.json", "Episode-level reward traces for the baseline agent"),
        ("improved_memory.json", "Episode-level reward traces for the best agent"),
        ("final_strategy.json", "The accepted (or baseline-fallback) strategy dict"),
        ("RESULTS.md", "This report"),
    ]
    lines.append("| File | Description |")
    lines.append("|:---|:---|")
    for fname, desc in artifact_files:
        if fname == output_path:
            status = "✅"  # This file — being written now
        else:
            status = "✅" if Path(fname).exists() else "❌"
        lines.append(f"| {status} `{fname}` | {desc} |")
    lines.append("")

    # ── Footer ────────────────────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append(
        "*Generated by `generate_report.py` — "
        "[meta-environment](https://github.com/akshar-3011/meta-environment)*"
    )
    lines.append("")

    # ── Write ─────────────────────────────────────────────────────────────
    content = "\n".join(lines)
    Path(output_path).write_text(content, encoding="utf-8")
    print(f"\n📄 Report written to {output_path} ({len(content)} bytes)")

    return content



def save_reward_curve_png(
    history_path: str = "evolution_history.json",
    output_path: str = "results/reward_curve.png",
    dpi: int = 150,
) -> None:
    """Plot mean total reward per generation and save as PNG.

    Parameters
    ----------
    history_path : str
        Path to the evolution_history.json file.
    output_path : str
        Destination PNG file path.
    dpi : int
        Resolution of the saved image.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    evolution = _load_json(history_path) or []
    if not evolution:
        print("⚠️  No evolution history — skipping reward curve PNG.")
        return

    generations = [e.get("generation", i) for i, e in enumerate(evolution)]
    totals = [e.get("mean_total", 0.0) for e in evolution]
    baseline = totals[0] if totals else 0.0

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Main curve
    ax.plot(
        generations, totals,
        marker="o", linewidth=2.5, markersize=8,
        color="#4F46E5", label="Mean Total Reward",
        zorder=3,
    )

    # Baseline dashed line
    ax.axhline(
        y=baseline, linestyle="--", linewidth=1.5,
        color="#EF4444", alpha=0.7, label=f"Baseline ({baseline:.3f})",
    )

    # Styling
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Mean Total Reward", fontsize=12)
    ax.set_title("Reward Progression Across Improvement Generations", fontsize=14, fontweight="bold")
    ax.set_xticks(generations)
    ax.set_ylim(bottom=0, top=max(max(totals) * 1.15, baseline * 1.15, 0.1))
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"📈 Reward curve saved to {output_path}")


if __name__ == "__main__":
    generate_report()
    save_reward_curve_png()
