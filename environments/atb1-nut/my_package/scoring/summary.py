"""Reduce recorded episode scores. No Isaac Sim imports."""

from __future__ import annotations


def summarize_recorded_scores(recorded_metric_data: list) -> float:
    values: list[float] = []
    for item in recorded_metric_data:
        if isinstance(item, (int, float)):
            values.append(float(item))
            continue
        values.extend(float(value) for value in item)
    if not values:
        return 0.0
    return sum(values) / len(values)
