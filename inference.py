"""Submission inference runner with strict START/STEP/END stdout logging."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from latentgoalops.models import LatentGoalOpsAction, TaskId
from latentgoalops.server.environment import LatentGoalOpsEnvironment


BENCHMARK_NAME = "latentgoalops"
DEFAULT_TASKS = ",".join(
    [
        TaskId.TASK1.value,
        TaskId.TASK2.value,
        TaskId.TASK3.value,
    ]
)
DEFAULT_API_BASE_URL = "https://inference.do-ai.run/v1"
DEFAULT_MODEL_NAME = "openai-gpt-oss-20b"
SUCCESS_SCORE_THRESHOLD = 0.10
MAX_NOTE_TOKENS = 32
API_BASE_URL = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")


def _parse_tasks(task_arg: str) -> list[str]:
    return [task.strip() for task in task_arg.split(",") if task.strip()]


def _parse_seeds(seed_arg: str) -> list[int]:
    if ":" in seed_arg:
        start, count = [int(part) for part in seed_arg.split(":", 1)]
        return list(range(start, start + count))
    return [int(part) for part in seed_arg.split(",") if part.strip()]


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
    load_dotenv(".env")

    api_base_url = cli_api_base_url or os.getenv("API_BASE_URL") or API_BASE_URL
    model_name = cli_model or os.getenv("MODEL_NAME") or MODEL_NAME
    token = _resolve_token(api_base_url=api_base_url, cli_token=cli_token)

    if not token:
        raise ValueError(
            "HF_TOKEN is required for submission inference. "
            "For local Gradient runs, you can also set DIGITALOCEAN_API_TOKEN."
        )

    os.environ["API_BASE_URL"] = api_base_url
    os.environ["MODEL_NAME"] = model_name
    os.environ["HF_TOKEN"] = token
    return api_base_url, model_name, token


def _sanitize_single_line(value: str | None) -> str:
    if not value:
        return "null"
    compact = re.sub(r"\s+", " ", value).strip()
    return compact if compact else "null"


def _compact_action(action: LatentGoalOpsAction) -> str:
    return json.dumps(action.model_dump(mode="json"), separators=(",", ":"), sort_keys=True)


def _clamp_score(score: float | None) -> float:
    if score is None:
        return 0.0
    return max(0.0, min(1.0, float(score)))


def _score_from_observation(observation) -> float:
    metadata = observation.metadata or {}
    grader = metadata.get("grader") or {}
    if "score" in grader:
        return _clamp_score(grader["score"])
    return _clamp_score(observation.reward)


def _compact_observation(observation) -> dict:
    payload = {
        "task_id": observation.task_id.value if hasattr(observation.task_id, "value") else str(observation.task_id),
        "task_summary": observation.task_summary,
        "sim_day_label": observation.sim_day_label,
        "dashboard": {
            "dau": observation.dashboard.dau,
            "d30_retention": observation.dashboard.d30_retention,
            "mrr": observation.dashboard.mrr,
            "ops_margin": observation.dashboard.ops_margin,
            "support_ticket_volume": observation.dashboard.support_ticket_volume,
        },
        "alerts": observation.alerts[:4],
        "stakeholder_notes": observation.stakeholder_notes[:4],
        "budget_remaining": observation.budget_remaining,
        "capacity_remaining": observation.capacity_remaining,
    }
    if observation.narrative:
        payload["narrative"] = observation.narrative
    if observation.inbox:
        payload["inbox"] = [
            {
                "item_id": item.item_id,
                "text": item.text,
                "sender": item.sender,
                "metadata": {
                    "severity": item.metadata.get("severity"),
                    "segment": item.metadata.get("segment"),
                    "annual_contract_value": item.metadata.get("annual_contract_value"),
                    "renewal_window_days": item.metadata.get("renewal_window_days"),
                },
            }
            for item in observation.inbox[:4]
        ]
    if observation.backlog:
        payload["backlog"] = [
            {
                "item_id": item.item_id,
                "kind": item.kind,
                "cost": item.cost,
                "kpi_deltas": item.kpi_deltas,
                "implementation_risk": item.implementation_risk,
                "beneficiary_segments": item.beneficiary_segments[:3],
            }
            for item in observation.backlog[:6]
        ]
    return payload


def _model_note(client: OpenAI, model_name: str, observation) -> str:
    prompt = {
        "instruction": (
            "Infer the most likely operating objective from the visible context. "
            "Return a plain single-line note of 3 to 8 words only."
        ),
        "observation": _compact_observation(observation),
    }
    request_kwargs = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You summarize benchmark context into one short operating note. "
                    "No JSON, no markdown, one plain line only."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, separators=(",", ":"), sort_keys=True)},
        ],
        "temperature": 0.0,
        "max_completion_tokens": MAX_NOTE_TOKENS,
        "reasoning_effort": "low",
        "seed": 0,
    }
    try:
        response = client.chat.completions.create(**request_kwargs)
    except TypeError:
        request_kwargs.pop("seed", None)
        response = client.chat.completions.create(**request_kwargs)
    except Exception:
        return "visible-context heuristic"

    content = response.choices[0].message.content or ""
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    note = _sanitize_single_line(content)
    if note == "null":
        return "visible-context heuristic"
    return note[:96]


def _attach_note(action: LatentGoalOpsAction, note: str) -> LatentGoalOpsAction:
    if action.task_id == TaskId.TASK2:
        return action.model_copy(update={"rationale_summary": note})
    if action.task_id in {TaskId.TASK3, TaskId.TASK5}:
        return action.model_copy(update={"rationale": note})
    if action.task_id == TaskId.TASK4:
        return action.model_copy(update={"rationale_summary": note})
    return action


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
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=seed, task_id=task_id)
    note = _model_note(client, model_name, observation)

    log_start(task=task_id, env=BENCHMARK_NAME, model=model_name)

    rewards: list[float] = []
    steps_taken = 0
    terminal_observation = observation
    error_message: str | None = None

    try:
        max_steps = max(1, int(env.state.max_steps or observation.horizon or 1))
        for step in range(1, max_steps + 1):
            action = _attach_note(env.sample_heuristic_action(), note)
            action_str = _compact_action(action)
            try:
                terminal_observation = env.step(action)
                reward = float(terminal_observation.reward or 0.0)
                done = bool(terminal_observation.done)
                error_message = None
            except Exception as exc:
                reward = 0.0
                done = True
                error_message = str(exc)

            rewards.append(reward)
            steps_taken = step
            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=error_message,
            )

            if done:
                break
    finally:
        close_method = getattr(env, "close", None)
        if callable(close_method):
            try:
                close_method()
            except Exception:
                pass

    score = _score_from_observation(terminal_observation)
    success = score >= SUCCESS_SCORE_THRESHOLD
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "seed": seed,
        "model_name": model_name,
        "score": score,
        "steps": steps_taken,
        "rewards": rewards,
        "success": success,
        "note": note,
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
