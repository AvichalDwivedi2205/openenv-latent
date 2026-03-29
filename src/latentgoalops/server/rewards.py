"""Reward shaping and action coherence helpers."""

from __future__ import annotations

import math
from typing import Iterable

from latentgoalops.models import (
    FeedbackLabel,
    LatentGoalOpsAction,
    MessagingAction,
    SupportPolicy,
)
from latentgoalops.server.config import load_config
from latentgoalops.server.hidden_goals import FEEDBACK_TO_CHANNEL


CHANNELS = ("growth", "retention", "revenue", "efficiency")


def _normalize_vector(vector: dict[str, float]) -> dict[str, float]:
    magnitude = math.sqrt(sum(value * value for value in vector.values()))
    if magnitude <= 0:
        return {channel: 0.0 for channel in CHANNELS}
    return {channel: value / magnitude for channel, value in vector.items()}


def strategy_embedding(action: LatentGoalOpsAction) -> dict[str, float]:
    """Embed an action into a four-channel strategic signature."""
    vector = {channel: 0.0 for channel in CHANNELS}

    if action.task_id.value == "task1_feedback_triage":
        for label_assignment in action.labels:
            channel = FEEDBACK_TO_CHANNEL.get(label_assignment.label, "growth")
            vector[channel] += 1.0
    elif action.task_id.value == "task2_roadmap_priority":
        for item_id in action.selected_item_ids:
            lowered = item_id.lower()
            for channel in CHANNELS:
                if channel in lowered:
                    vector[channel] += 1.0
                    break
            else:
                vector["growth"] += 0.5
    else:
        for initiative_id in action.chosen_initiatives:
            lowered = initiative_id.lower()
            for channel in CHANNELS:
                if channel in lowered:
                    vector[channel] += 1.0
                    break
        message_map = {
            MessagingAction.GROWTH_PUSH: "growth",
            MessagingAction.RETENTION_CAMPAIGN: "retention",
            MessagingAction.REVENUE_UPSELL: "revenue",
            MessagingAction.COST_COMMS: "efficiency",
        }
        if action.messaging_action in message_map:
            vector[message_map[action.messaging_action]] += 1.0
        if action.support_policy == SupportPolicy.PREMIUM_SLA:
            vector["retention"] += 0.75
        elif action.support_policy == SupportPolicy.AUTOMATION_FIRST:
            vector["efficiency"] += 0.75
        elif action.pricing_change_pct and action.pricing_change_pct > 0:
            vector["revenue"] += abs(action.pricing_change_pct) * 5
        elif action.pricing_change_pct and action.pricing_change_pct < 0:
            vector["growth"] += abs(action.pricing_change_pct) * 3

    return _normalize_vector(vector)


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    """Cosine similarity for already aligned channel vectors."""
    denom = math.sqrt(sum(v * v for v in left.values())) * math.sqrt(sum(v * v for v in right.values()))
    if denom <= 0:
        return 0.0
    return sum(left[channel] * right[channel] for channel in CHANNELS) / denom


def compute_kpi_progress(previous_metrics: dict[str, float], new_metrics: dict[str, float]) -> tuple[float, float]:
    """Return observable improvement and observable damage."""
    improvements = []
    damage = []
    for channel in CHANNELS:
        delta = float(new_metrics.get(channel, 0.0)) - float(previous_metrics.get(channel, 0.0))
        improvements.append(max(0.0, delta))
        damage.append(max(0.0, -delta))
    return sum(improvements) / len(CHANNELS), sum(damage) / len(CHANNELS)


def compute_proxy_reward(
    previous_metrics: dict[str, float],
    new_metrics: dict[str, float],
    previous_action: LatentGoalOpsAction | None,
    current_action: LatentGoalOpsAction,
    invalid: bool,
    unused_budget_ratio: float,
) -> float:
    """Compute the leakage-safe shaped reward."""
    lambdas = load_config("reward.yaml")["lambdas"]
    kpi_improvement, kpi_damage = compute_kpi_progress(previous_metrics, new_metrics)
    coherence = 0.0
    if previous_action is not None:
        coherence = max(0.0, cosine_similarity(strategy_embedding(previous_action), strategy_embedding(current_action)))
    invalid_penalty = 1.0 if invalid else 0.0
    waste_penalty = unused_budget_ratio * unused_budget_ratio

    reward = (
        float(lambdas["kpi_improvement"]) * kpi_improvement
        + float(lambdas["coherence"]) * coherence
        - float(lambdas["invalid"]) * invalid_penalty
        - float(lambdas["waste"]) * waste_penalty
        - float(lambdas["damage"]) * (kpi_damage * 2.0)
    )
    return float(max(-1.0, min(1.0, reward)))

