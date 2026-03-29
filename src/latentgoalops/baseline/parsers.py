"""Robust JSON parsing for model actions."""

from __future__ import annotations

import json

from latentgoalops.models import LatentGoalOpsAction


def extract_json_object(text: str) -> dict:
    """Extract the most plausible top-level JSON object from text."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model response did not contain a JSON object.")
    return json.loads(text[start : end + 1])


def parse_action(text: str) -> LatentGoalOpsAction:
    """Parse a model response into a validated action."""
    return LatentGoalOpsAction.model_validate(extract_json_object(text))

