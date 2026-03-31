"""Simple text visualization helpers for benchmark and scoring output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def ascii_bar_chart(values: Dict[str, float], *, width: int = 30, char: str = "#") -> str:
    if not values:
        return "(no values)"

    max_value = max(values.values()) if values else 1.0
    max_value = max(1e-9, max_value)
    lines: List[str] = []
    for key, value in sorted(values.items(), key=lambda x: x[1], reverse=True):
        normalized = max(0.0, min(1.0, value / max_value))
        bar_count = int(round(normalized * width))
        lines.append(f"{key:>16} | {char * bar_count:<{width}} | {value:.4f}")
    return "\n".join(lines)


def benchmark_report(summary: Dict[str, Any]) -> str:
    per_strategy = summary.get("per_strategy", {})
    averages = {name: float(stats.get("avg", 0.0)) for name, stats in per_strategy.items()}
    chart = ascii_bar_chart(averages)

    header = [
        "BENCHMARK SUMMARY",
        "=" * 60,
        f"Total runs      : {summary.get('total_runs', 0)}",
        f"Elapsed seconds : {summary.get('elapsed_seconds', 0.0):.4f}",
        "",
        "Average score by strategy",
        chart,
        "",
        "Ranking",
    ]

    ranking = summary.get("ranking", [])
    for idx, row in enumerate(ranking, start=1):
        header.append(f"{idx:>2}. {row['strategy']} ({row['avg']:.4f})")

    return "\n".join(header)


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out
