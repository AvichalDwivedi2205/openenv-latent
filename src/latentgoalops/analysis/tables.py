"""Paper-ready summary tables."""

from __future__ import annotations

import json
from pathlib import Path

from latentgoalops.analysis.stats import bootstrap_mean_ci


def write_summary_tables(episodes, output_dir: str | Path) -> dict:
    """Write JSON tables with mean score, CI, cost, and runtime."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    if episodes.empty:
        result = {"scores": {}, "costs": {}}
        (output_root / "summary_tables.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    scores: dict[str, dict] = {}
    costs: dict[str, dict] = {}
    for (model_name, task_id), group in episodes.groupby(["model_name", "task_id"]):
        score_values = [float(value) for value in group["score"].tolist()]
        elapsed_values = [float(value) for value in group["elapsed_seconds"].tolist()]
        cost_values = [float(value) for value in group["cost_usd"].tolist()]
        if "strict_episode" in group:
            strict_values = [float(value) for value in group[group["strict_episode"] == 1]["score"].tolist()]
        else:
            strict_values = []
        ci_low, ci_high = bootstrap_mean_ci(score_values, samples=2000)
        scores.setdefault(model_name, {})[task_id] = {
            "mean_score": sum(score_values) / len(score_values),
            "ci95": [ci_low, ci_high],
            "n": len(score_values),
            "strict_mean_score": (sum(strict_values) / len(strict_values)) if strict_values else 0.0,
            "rescued_fraction": float(group["rescued_episode"].mean()) if "rescued_episode" in group else 0.0,
        }
        costs.setdefault(model_name, {})[task_id] = {
            "mean_cost_usd": sum(cost_values) / len(cost_values),
            "mean_elapsed_seconds": sum(elapsed_values) / len(elapsed_values),
            "parse_repaired_rate": float(group["parse_repaired"].mean()) if "parse_repaired" in group else 0.0,
            "parse_fallback_rate": float(group["parse_fallback"].mean()) if "parse_fallback" in group else 0.0,
        }
    result = {"scores": scores, "costs": costs}
    (output_root / "summary_tables.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
