"""Visualization utilities for the workplace environment.

Provides:
- ``ascii_bar_chart``: Simple ASCII bar chart from a dict of label → value.
- ``benchmark_report``: Format a benchmark summary dict as a text report.
- Terminal dashboard functions in the ``terminal_dashboard`` submodule.
"""

from __future__ import annotations

from typing import Any, Dict, List


def ascii_bar_chart(
    data: Dict[str, float],
    width: int = 20,
    max_value: float = 1.0,
) -> str:
    """Render a simple ASCII bar chart.

    Parameters
    ----------
    data : dict[str, float]
        Label → value mapping.
    width : int
        Maximum bar width in characters.
    max_value : float
        Scale factor (values are divided by this).

    Returns
    -------
    str
        Multi-line ASCII chart.
    """
    if not data:
        return "(no data)"

    max_label_len = max(len(k) for k in data)
    lines: List[str] = []

    for label, value in data.items():
        bar_len = int(round((value / max_value) * width)) if max_value > 0 else 0
        bar_len = max(0, min(width, bar_len))
        bar = "█" * bar_len + "░" * (width - bar_len)
        lines.append(f"{label:>{max_label_len}} [{bar}] {value:.3f}")

    return "\n".join(lines)


def benchmark_report(summary: Dict[str, Any]) -> str:
    """Format a benchmark summary dictionary as a readable text report.

    Parameters
    ----------
    summary : dict
        Expected keys: ``total_runs``, ``elapsed_seconds``,
        ``per_strategy`` (dict of strategy → stats), ``ranking`` (list).

    Returns
    -------
    str
        Multi-line text report.
    """
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("BENCHMARK SUMMARY")
    lines.append("=" * 60)

    total = summary.get("total_runs", 0)
    elapsed = summary.get("elapsed_seconds", 0)
    lines.append(f"Total runs    : {total}")
    lines.append(f"Elapsed (s)   : {elapsed:.3f}")

    per_strategy = summary.get("per_strategy", {})
    if per_strategy:
        lines.append("")
        lines.append("Per-Strategy Results")
        lines.append("-" * 60)
        for name, stats in per_strategy.items():
            if isinstance(stats, dict):
                avg = stats.get("avg", 0.0)
                lines.append(f"  {name:<20} avg={avg:.3f}")
            else:
                lines.append(f"  {name:<20} {stats}")

    ranking = summary.get("ranking", [])
    if ranking:
        lines.append("")
        lines.append("Ranking")
        lines.append("-" * 60)
        for i, entry in enumerate(ranking, 1):
            name = entry.get("strategy", "unknown")
            avg = entry.get("avg", 0.0)
            lines.append(f"  #{i} {name:<18} avg={avg:.3f}")

    lines.append("=" * 60)
    return "\n".join(lines)


__all__ = ["ascii_bar_chart", "benchmark_report"]
