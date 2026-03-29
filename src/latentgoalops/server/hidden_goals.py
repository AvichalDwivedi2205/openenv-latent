"""Latent goal sampling and utility helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from latentgoalops.models import FeedbackLabel, GoalArchetype
from latentgoalops.server.config import load_config


CHANNELS = ("growth", "retention", "revenue", "efficiency")

FEEDBACK_TO_CHANNEL = {
    FeedbackLabel.BUG: "efficiency",
    FeedbackLabel.FEATURE_REQUEST: "growth",
    FeedbackLabel.CHURN_RISK: "retention",
    FeedbackLabel.PRAISE: "growth",
    FeedbackLabel.BILLING_ISSUE: "revenue",
    FeedbackLabel.LATENCY_COMPLAINT: "efficiency",
}


@dataclass(slots=True)
class HiddenGoal:
    """Sampled hidden objective for an episode."""

    archetype: GoalArchetype
    weights: dict[str, float]
    alpha: float
    shift_goal: GoalArchetype | None = None
    shift_step: int | None = None
    shift_weights: dict[str, float] | None = None


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values()) or 1.0
    return {key: value / total for key, value in weights.items()}


def sample_hidden_goal(rng: random.Random, allow_shift: bool = False) -> HiddenGoal:
    """Sample a goal archetype and lightly perturbed weight vector."""
    config = load_config("hidden_goals.yaml")
    archetype = GoalArchetype(rng.choice(list(config["archetypes"].keys())))
    alpha = rng.choice(config["alpha_choices"])
    weights = sample_weights_for_archetype(archetype, float(alpha), rng)

    shift_goal: GoalArchetype | None = None
    shift_step: int | None = None
    shift_weights: dict[str, float] | None = None
    if allow_shift and rng.random() < float(config["shift_probability"]):
        choices = [GoalArchetype(value) for value in config["archetypes"] if value != archetype.value]
        shift_goal = rng.choice(choices)
        shift_step = int(rng.choice(config["shift_steps"]))
        shift_weights = sample_weights_for_archetype(shift_goal, float(alpha), rng)

    return HiddenGoal(
        archetype=archetype,
        weights=weights,
        alpha=float(alpha),
        shift_goal=shift_goal,
        shift_step=shift_step,
        shift_weights=shift_weights,
    )


def sample_weights_for_archetype(archetype: GoalArchetype, alpha: float, rng: random.Random) -> dict[str, float]:
    """Sample a perturbed weight vector around a goal archetype."""
    config = load_config("hidden_goals.yaml")
    base = config["archetypes"][archetype.value]["weights"]
    dirichlet = np.random.default_rng(rng.randint(0, 10_000_000)).dirichlet([alpha] * 4)
    blended = {
        channel: 0.75 * float(base[channel]) + 0.25 * float(dirichlet[index])
        for index, channel in enumerate(CHANNELS)
    }
    return _normalize(blended)


def channel_weight(hidden_goal: HiddenGoal, channel: str) -> float:
    """Return channel weight for a hidden goal."""
    return float(hidden_goal.weights.get(channel, 0.0))


def compute_utility(metric_vector: dict[str, float], hidden_goal: HiddenGoal) -> float:
    """Compute latent utility for a normalized four-channel metric vector."""
    return max(
        0.0,
        min(
            1.0,
            sum(float(metric_vector.get(channel, 0.0)) * hidden_goal.weights[channel] for channel in CHANNELS),
        ),
    )


def active_weights(hidden_goal: HiddenGoal, step_index: int) -> dict[str, float]:
    """Return the active weight vector at a given step index."""
    if hidden_goal.shift_step is not None and hidden_goal.shift_weights is not None and step_index >= hidden_goal.shift_step:
        return hidden_goal.shift_weights
    return hidden_goal.weights


def active_goal_name(hidden_goal: HiddenGoal, step_index: int) -> str:
    """Return the active goal archetype name at a step."""
    if hidden_goal.shift_step is not None and hidden_goal.shift_goal is not None and step_index >= hidden_goal.shift_step:
        return hidden_goal.shift_goal.value
    return hidden_goal.archetype.value


def feedback_category_weight(hidden_goal: HiddenGoal, label: FeedbackLabel) -> float:
    """Map a feedback category to its hidden utility weight."""
    return channel_weight(hidden_goal, FEEDBACK_TO_CHANNEL[label])


def initiative_alignment_score(hidden_goal: HiddenGoal, initiative_kind: str) -> float:
    """How aligned an initiative kind is with the latent goal."""
    return channel_weight(hidden_goal, initiative_kind if initiative_kind in CHANNELS else "growth")
