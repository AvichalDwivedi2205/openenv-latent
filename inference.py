"""Submission inference runner with strict START/STEP/END stdout logging."""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

try:
    from openai import OpenAI as _OpenAIClient
except Exception:
    _OpenAIClient = None


BENCHMARK = "latentgoalops"
DEFAULT_API_BASE_URL = "https://inference.do-ai.run/v1"
DEFAULT_MODEL_NAME = "openai-gpt-oss-20b"
DEFAULT_ENV_BASE_URL = "https://avichal2205-latentgoalops-openenv.hf.space"
DEFAULT_TASKS = ",".join(
    [
        "task1_feedback_triage",
        "task2_roadmap_priority",
        "task3_startup_week",
    ]
)
SUCCESS_SCORE_THRESHOLD = 0.10
MAX_NOTE_TOKENS = 32

API_BASE_URL = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL") or os.getenv("PING_URL") or DEFAULT_ENV_BASE_URL


class _CompatMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _CompatChoice:
    def __init__(self, content: str) -> None:
        self.message = _CompatMessage(content)


class _CompatResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_CompatChoice(content)]


class _CompatChatCompletions:
    def __init__(self, base_url: str, api_key: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    def create(self, **kwargs) -> _CompatResponse:
        payload = {
            "model": kwargs["model"],
            "messages": kwargs["messages"],
            "temperature": kwargs.get("temperature", 0.0),
            "stream": False,
        }
        if "max_completion_tokens" in kwargs:
            payload["max_completion_tokens"] = kwargs["max_completion_tokens"]
        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]

        request = urllib.request.Request(
            self._base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8")
        data = json.loads(body)
        choices = data.get("choices") or []
        message = (choices[0].get("message") or {}) if choices else {}
        content = message.get("content") or ""
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return _CompatResponse(str(content))


class _CompatChat:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.completions = _CompatChatCompletions(base_url=base_url, api_key=api_key)


class OpenAI:
    def __init__(self, base_url: str, api_key: str) -> None:
        if _OpenAIClient is not None:
            self._client = _OpenAIClient(base_url=base_url, api_key=api_key)
            self.chat = self._client.chat
        else:
            self._client = None
            self.chat = _CompatChat(base_url=base_url, api_key=api_key)


def _parse_tasks(task_arg: str) -> list[str]:
    return [task.strip() for task in task_arg.split(",") if task.strip()]


def _parse_seeds(seed_arg: str) -> list[int]:
    if ":" in seed_arg:
        start, count = [int(part) for part in seed_arg.split(":", 1)]
        return list(range(start, start + count))
    return [int(part) for part in seed_arg.split(",") if part.strip()]


def _sanitize_single_line(value: str | None) -> str:
    if not value:
        return "null"
    compact = re.sub(r"\s+", " ", str(value)).strip()
    return compact if compact else "null"


def _clamp_score(score: float | None) -> float:
    if score is None:
        return 0.0
    return max(0.0, min(1.0, float(score)))


def _resolve_token(api_base_url: str, cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token
    base_url = api_base_url.lower()
    if "do-ai.run" in base_url or "digitalocean" in base_url:
        return (
            os.getenv("DIGITALOCEAN_API_TOKEN")
            or os.getenv("HF_TOKEN")
            or os.getenv("API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("DIGITALOCEAN_API_TOKEN")
        or os.getenv("API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )


def _resolve_submission_env(
    cli_model: str | None,
    cli_api_base_url: str | None,
    cli_token: str | None,
) -> tuple[str, str, str]:
    api_base_url = cli_api_base_url or os.getenv("API_BASE_URL") or API_BASE_URL
    model_name = cli_model or os.getenv("MODEL_NAME") or MODEL_NAME
    token = _resolve_token(api_base_url=api_base_url, cli_token=cli_token)
    if not token:
        raise ValueError(
            "HF_TOKEN is required for submission inference. "
            "For Gradient runs you may also set DIGITALOCEAN_API_TOKEN."
        )
    return api_base_url, model_name, token


def _http_json(method: str, path: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        ENV_BASE_URL.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read().decode("utf-8")
        return json.loads(body) if body else {}


def _positive_value(item: dict) -> float:
    deltas = item.get("kpi_deltas") or {}
    return sum(max(0.0, float(value)) for value in deltas.values())


def _infer_goal_hint(observation: dict) -> str:
    evidence_parts = [observation.get("narrative", "")]
    evidence_parts.extend(observation.get("alerts", []))
    evidence_parts.extend(observation.get("stakeholder_notes", []))
    for item in observation.get("inbox", []):
        evidence_parts.append(item.get("text", ""))
    lowered = " ".join(evidence_parts).lower()
    if any(keyword in lowered for keyword in ("renewal", "account health", "trust", "support backlog", "fragile")):
        return "retention"
    if any(keyword in lowered for keyword in ("pricing", "monetization", "board", "commercial", "pipeline")):
        return "revenue"
    if any(keyword in lowered for keyword in ("margin", "latency", "cost-to-serve", "ops", "runway discipline")):
        return "efficiency"
    return "growth"


def _rank_backlog(observation: dict, preferred_kind: str | None = None) -> list[dict]:
    backlog = list(observation.get("backlog", []))
    return sorted(
        backlog,
        key=lambda item: (
            1 if preferred_kind and item.get("kind") == preferred_kind else 0,
            _positive_value(item) / max(float(item.get("cost") or 1.0), 1.0),
            -float(item.get("implementation_risk") or 0.0),
        ),
        reverse=True,
    )


def _heuristic_task1(observation: dict) -> dict:
    keyword_map = {
        "charged": "billing_issue",
        "invoice": "billing_issue",
        "cancel": "churn_risk",
        "renew": "churn_risk",
        "slow": "latency_complaint",
        "latency": "latency_complaint",
        "crash": "bug",
        "blank screen": "bug",
        "please add": "feature_request",
        "want": "feature_request",
    }
    labels = []
    priorities = []
    scored = []
    for item in observation.get("inbox", []):
        lowered = item.get("text", "").lower()
        predicted = "praise"
        for keyword, label in keyword_map.items():
            if keyword in lowered:
                predicted = label
                break
        metadata = item.get("metadata") or {}
        severity = int(metadata.get("severity", 3) or 3)
        support_tier = str(metadata.get("support_tier", "standard"))
        priority = min(5, max(1, severity + (1 if support_tier in {"priority", "enterprise"} else 0)))
        labels.append({"item_id": item["item_id"], "label": predicted})
        priorities.append({"item_id": item["item_id"], "priority": priority})
        scored.append((item["item_id"], priority))
    scored.sort(key=lambda row: row[1], reverse=True)
    return {
        "task_id": observation["task_id"],
        "labels": labels,
        "priorities": priorities,
        "escalate_ids": [item_id for item_id, _ in scored[:3]],
    }


def _heuristic_task2(observation: dict, note: str) -> dict:
    ranked = _rank_backlog(observation)
    remaining = float(observation.get("sprint_budget") or 0.0)
    selected = []
    for item in ranked:
        cost = float(item.get("cost") or 0.0)
        if cost <= remaining:
            selected.append(item["item_id"])
            remaining -= cost
    return {
        "task_id": observation["task_id"],
        "selected_item_ids": selected,
        "rationale_summary": note,
    }


def _heuristic_task3(observation: dict, note: str) -> dict:
    goal_hint = _infer_goal_hint(observation)
    ranked = _rank_backlog(observation, preferred_kind=goal_hint)
    chosen = [item["item_id"] for item in ranked[:2]]
    messaging = {
        "growth": "growth_push",
        "retention": "retention_campaign",
        "revenue": "revenue_upsell",
        "efficiency": "cost_comms",
    }[goal_hint]
    support = {
        "growth": "balanced_triage",
        "retention": "premium_sla",
        "revenue": "balanced_triage",
        "efficiency": "automation_first",
    }[goal_hint]
    pricing = 0.04 if goal_hint == "revenue" else (-0.03 if goal_hint == "growth" else 0.0)
    return {
        "task_id": observation["task_id"],
        "chosen_initiatives": chosen,
        "messaging_action": messaging,
        "pricing_change_pct": pricing,
        "support_policy": support,
        "rationale": note,
    }


def _heuristic_task4(observation: dict, note: str) -> dict:
    ranked = _rank_backlog(observation)
    remaining = int(round(float(observation.get("sprint_budget") or 0.0)))
    allocations: dict[str, float] = {}
    for item in ranked:
        if remaining <= 0:
            break
        saturation = item.get("saturation_point")
        allocation_max = item.get("allocation_max")
        preferred = min(
            remaining,
            int(round(float(saturation if saturation is not None else allocation_max or 0.0))),
        )
        if preferred <= 0:
            continue
        allocations[item["item_id"]] = float(preferred)
        remaining -= preferred
    return {
        "task_id": observation["task_id"],
        "budget_allocations": allocations,
        "rationale_summary": note,
    }


def _heuristic_task5(observation: dict, note: str) -> dict:
    goal_hint = _infer_goal_hint(observation)
    ranked = _rank_backlog(observation, preferred_kind=goal_hint)
    remaining_budget = float(observation.get("budget_remaining") or 0.0)
    remaining_capacity = int(float(observation.get("capacity_remaining") or 0.0))
    chosen: list[str] = []
    for item in ranked:
        if len(chosen) >= remaining_capacity:
            break
        cost = float(item.get("cost") or 0.0)
        if cost > remaining_budget:
            continue
        chosen.append(item["item_id"])
        remaining_budget -= cost
    return {
        "task_id": observation["task_id"],
        "chosen_initiatives": chosen,
        "messaging_action": {
            "growth": "growth_push",
            "retention": "retention_campaign",
            "revenue": "revenue_upsell",
            "efficiency": "cost_comms",
        }[goal_hint],
        "pricing_change_pct": 0.0,
        "support_policy": {
            "growth": "balanced_triage",
            "retention": "premium_sla",
            "revenue": "balanced_triage",
            "efficiency": "automation_first",
        }[goal_hint],
        "rationale": note,
    }


def _build_action(observation: dict, note: str) -> dict:
    task_id = observation["task_id"]
    if task_id == "task1_feedback_triage":
        return _heuristic_task1(observation)
    if task_id == "task2_roadmap_priority":
        return _heuristic_task2(observation, note)
    if task_id == "task3_startup_week":
        return _heuristic_task3(observation, note)
    if task_id == "task4_capital_allocation":
        return _heuristic_task4(observation, note)
    return _heuristic_task5(observation, note)


def _extract_score(result: dict) -> float:
    observation = result.get("observation") or {}
    metadata = observation.get("metadata") or {}
    grader = metadata.get("grader") or {}
    if "score" in grader:
        return _clamp_score(grader["score"])
    return _clamp_score(result.get("reward"))


def _compact_observation_for_model(observation: dict) -> dict:
    payload = {
        "task_id": observation.get("task_id"),
        "task_summary": observation.get("task_summary"),
        "sim_day_label": observation.get("sim_day_label"),
        "dashboard": observation.get("dashboard"),
        "alerts": observation.get("alerts", [])[:4],
        "stakeholder_notes": observation.get("stakeholder_notes", [])[:4],
        "budget_remaining": observation.get("budget_remaining"),
        "capacity_remaining": observation.get("capacity_remaining"),
    }
    if observation.get("narrative"):
        payload["narrative"] = observation["narrative"]
    if observation.get("inbox"):
        payload["inbox"] = observation["inbox"][:4]
    if observation.get("backlog"):
        payload["backlog"] = observation["backlog"][:6]
    return payload


def _model_note(client: OpenAI, model_name: str, observation: dict) -> str:
    prompt = {
        "instruction": (
            "Infer the likely operating goal from the visible context. "
            "Return one short plain phrase only."
        ),
        "observation": _compact_observation_for_model(observation),
    }
    kwargs = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Reply with one short phrase only. "
                    "No markdown, no JSON, no explanation."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, separators=(",", ":"), sort_keys=True)},
        ],
        "temperature": 0.0,
        "max_completion_tokens": MAX_NOTE_TOKENS,
    }
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception:
        return "visible-context heuristic"

    content = response.choices[0].message.content
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    note = _sanitize_single_line(content)
    if note == "null":
        return "visible-context heuristic"
    return note[:96]


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={_sanitize_single_line(error)}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def _run_episode(client: OpenAI, model_name: str, task_id: str, seed: int) -> dict:
    log_start(task=task_id, env=BENCHMARK, model=model_name)

    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        result = _http_json("POST", "/reset", {"task_id": task_id, "seed": seed})
        observation = result.get("observation") or {}
        note = _model_note(client, model_name, observation)
        max_steps = max(1, int(observation.get("horizon") or 1))

        for step in range(1, max_steps + 1):
            action = _build_action(observation, note)
            action_str = json.dumps(action, separators=(",", ":"), sort_keys=True)
            error_message = None
            try:
                result = _http_json("POST", "/step", {"action": action})
                reward = float(result.get("reward") or 0.0)
                done = bool(result.get("done"))
                observation = result.get("observation") or {}
            except Exception as exc:
                reward = 0.0
                done = True
                error_message = str(exc)

            rewards.append(reward)
            steps_taken = step
            log_step(step, action_str, reward, done, error_message)

            if done:
                score = _extract_score(result if "result" in locals() else {})
                success = score >= SUCCESS_SCORE_THRESHOLD
                break
        else:
            score = _extract_score(result)
            success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception as exc:
        log_end(success=False, steps=steps_taken, score=0.0, rewards=rewards)
        return {
            "task_id": task_id,
            "seed": seed,
            "score": 0.0,
            "steps": steps_taken,
            "rewards": rewards,
            "success": False,
            "error": _sanitize_single_line(str(exc)),
            "local_image_name": LOCAL_IMAGE_NAME,
        }

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {
        "task_id": task_id,
        "seed": seed,
        "score": score,
        "steps": steps_taken,
        "rewards": rewards,
        "success": success,
        "local_image_name": LOCAL_IMAGE_NAME,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the LatentGoalOps submission baseline with strict START/STEP/END stdout logs."
    )
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--seeds", default="100:1")
    parser.add_argument("--output-dir", default="outputs/submission-baseline")
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-base-url", default=None)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    api_base_url, model_name, token = _resolve_submission_env(
        cli_model=args.model,
        cli_api_base_url=args.api_base_url,
        cli_token=args.token,
    )
    client = OpenAI(base_url=api_base_url, api_key=token)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = []
    for task_id in _parse_tasks(args.tasks):
        for seed in _parse_seeds(args.seeds):
            episodes.append(_run_episode(client, model_name, task_id, seed))

    summary = {
        "env_base_url": ENV_BASE_URL,
        "api_base_url": api_base_url,
        "model_name": model_name,
        "tasks": [episode["task_id"] for episode in episodes],
        "seeds": _parse_seeds(args.seeds),
        "episodes": episodes,
        "mean_scores": {},
    }
    for task_id in _parse_tasks(args.tasks):
        task_scores = [episode["score"] for episode in episodes if episode["task_id"] == task_id]
        if task_scores:
            summary["mean_scores"][task_id] = round(sum(task_scores) / len(task_scores), 4)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
