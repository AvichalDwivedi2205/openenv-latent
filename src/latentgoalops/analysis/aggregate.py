"""Aggregate JSONL experiment logs into tabular summaries."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _infer_policy(payload: dict) -> str:
    policy = payload.get("policy") or payload.get("metadata", {}).get("policy")
    if policy:
        return str(policy)
    model_name = str(payload.get("model_name", ""))
    if model_name in {"random", "heuristic", "oracle"}:
        return model_name
    return "model"


def _experiment_metadata(payload: dict) -> dict:
    metadata = payload.get("metadata", {}) or {}
    experiment_config = metadata.get("experiment_config", {}) or {}
    return {
        "ablation": (
            "full"
            if not experiment_config
            else "+".join(
                part
                for part, enabled in [
                    ("no_shift", not experiment_config.get("enable_hidden_shift", True)),
                    ("no_delay", not experiment_config.get("enable_delayed_effects", True)),
                    ("no_ledger", not experiment_config.get("expose_decision_ledger", True)),
                    ("sparse", experiment_config.get("reward_mode", "shaped") == "sparse"),
                ]
                if enabled
            )
            or "full"
        ),
        "reward_mode": experiment_config.get("reward_mode", "shaped"),
        "enable_hidden_shift": bool(experiment_config.get("enable_hidden_shift", True)),
        "enable_delayed_effects": bool(experiment_config.get("enable_delayed_effects", True)),
        "expose_decision_ledger": bool(experiment_config.get("expose_decision_ledger", True)),
    }


def _jsonl_paths(path: str | Path) -> list[Path]:
    candidate = Path(path)
    if candidate.is_file():
        return [candidate]
    if candidate.is_dir():
        return sorted(candidate.rglob("runs.jsonl"))
    return []


def load_run_records(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load step and episode records from one file or a directory tree of logs."""
    step_records = []
    episode_records = []
    for jsonl_path in _jsonl_paths(path):
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                if "total_steps" in payload and "score" in payload:
                    flat = {
                        **payload,
                        "policy": _infer_policy(payload),
                        "cost_usd": payload.get("provider_usage", {}).get("cost_usd", 0.0),
                        "input_tokens": payload.get("provider_usage", {}).get("input_tokens", 0),
                        "output_tokens": payload.get("provider_usage", {}).get("output_tokens", 0),
                        "parse_fallback": int(bool(payload.get("provider_usage", {}).get("parse_fallback", False))),
                        "parse_repaired": int(bool(payload.get("provider_usage", {}).get("parse_repaired", False))),
                        "strict_episode": int(bool(payload.get("metadata", {}).get("strict_episode", False))),
                        "rescued_episode": int(bool(payload.get("metadata", {}).get("rescued_episode", False))),
                        **_experiment_metadata(payload),
                    }
                    for key, value in payload.get("grade", {}).get("sub_scores", {}).items():
                        flat[f"subscore_{key}"] = value
                    episode_records.append(flat)
                elif "step_index" in payload and "reward" in payload:
                    flat = {
                        **payload,
                        "policy": _infer_policy(payload),
                        "cost_usd": payload.get("provider_usage", {}).get("cost_usd", 0.0),
                        "input_tokens": payload.get("provider_usage", {}).get("input_tokens", 0),
                        "output_tokens": payload.get("provider_usage", {}).get("output_tokens", 0),
                        "parse_fallback": int(bool(payload.get("provider_usage", {}).get("parse_fallback", False))),
                        "parse_repaired": int(bool(payload.get("provider_usage", {}).get("parse_repaired", False))),
                        "strict_step": int(bool(payload.get("metadata", {}).get("strict_step", False))),
                        **_experiment_metadata(payload),
                    }
                    step_records.append(flat)
    return pd.DataFrame(step_records), pd.DataFrame(episode_records)


def load_episode_summaries(path: str | Path) -> pd.DataFrame:
    """Load terminal episode rows from run logs."""
    _, episodes = load_run_records(path)
    return episodes


def load_step_logs(path: str | Path) -> pd.DataFrame:
    """Load step rows from run logs."""
    steps, _ = load_run_records(path)
    return steps


def summarize_scores(path: str | Path) -> pd.DataFrame:
    """Compute mean/std scores by model, policy, and task."""
    df = load_episode_summaries(path)
    if df.empty:
        return df
    return (
        df.groupby(["policy", "model_name", "task_id"], as_index=False)["score"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def summarize_costs(path: str | Path) -> pd.DataFrame:
    """Compute cost and token usage by model, policy, and task."""
    df = load_episode_summaries(path)
    if df.empty:
        return df
    return (
        df.groupby(["policy", "model_name", "task_id"], as_index=False)[["cost_usd", "input_tokens", "output_tokens"]]
        .mean()
        .reset_index()
    )


def summarize_parse_fallbacks(path: str | Path) -> pd.DataFrame:
    """Compute parse fallback rates by policy/model/task."""
    df = load_episode_summaries(path)
    if df.empty:
        return df
    return (
        df.groupby(["policy", "model_name", "task_id"], as_index=False)[["parse_fallback", "parse_repaired", "rescued_episode"]]
        .mean()
        .reset_index()
    )
