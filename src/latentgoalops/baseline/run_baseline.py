"""Baseline runner for LatentGoalOps."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from latentgoalops.experiment import ExperimentConfig
from latentgoalops.baseline.heuristics import choose_action
from latentgoalops.baseline.parsers import parse_action
from latentgoalops.baseline.prompts import system_prompt, user_prompt
from latentgoalops.baseline.providers import (
    GradientChatProvider,
    ProviderCredentialError,
    ProviderExhaustedError,
)
from latentgoalops.baseline.synthetic_operator import (
    apply_operator_guardrails,
    operator_style_choices,
    resolve_operator_persona,
    stabilize_model_action,
)
from latentgoalops.logging_.jsonl_logger import JsonlLogger
from latentgoalops.logging_.schemas import CheckpointState, EpisodeSummary, StepLog
from latentgoalops.logging_.wandb_logger import WandbLogger
from latentgoalops.models import LatentGoalOpsAction, TaskId
from latentgoalops.server.environment import LatentGoalOpsEnvironment

DEFAULT_AGENT_MODEL = "openai-gpt-oss-20b"
DEFAULT_PERSONA_MODEL = "openai-gpt-oss-20b"


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "unknown"
    whole = max(0, int(round(value)))
    hours, rem = divmod(whole, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_tasks(task_arg: str) -> list[str]:
    if task_arg == "all":
        return [task.value for task in TaskId]
    return [task.strip() for task in task_arg.split(",") if task.strip()]


def _parse_seeds(seed_arg: str) -> list[int]:
    if ":" in seed_arg:
        start, count = [int(part) for part in seed_arg.split(":", 1)]
        return list(range(start, start + count))
    return [int(part) for part in seed_arg.split(",") if part.strip()]


def _experiment_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        enable_hidden_shift=not bool(getattr(args, "disable_hidden_shift", False)),
        enable_delayed_effects=not bool(getattr(args, "disable_delayed_effects", False)),
        expose_decision_ledger=not bool(getattr(args, "hide_decision_ledger", False)),
        reward_mode=str(getattr(args, "reward_mode", "shaped")),
        task3_horizon_override=getattr(args, "task3_horizon_override", None),
        scenario_split=str(getattr(args, "scenario_split", "core")),
    )


def _load_checkpoint(
    path: Path,
    run_id: str,
    model_name: str,
    policy: str,
    tasks: list[str],
    seeds: list[int],
) -> CheckpointState:
    if path.exists():
        return CheckpointState.model_validate_json(path.read_text(encoding="utf-8"))
    checkpoint = CheckpointState(
        run_id=run_id,
        model_name=model_name,
        policy=policy,
        tasks=tasks,
        seeds=seeds,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(checkpoint.model_dump_json(indent=2), encoding="utf-8")
    return checkpoint


def _completion_budget(task_id: TaskId, policy_mode: str = "model") -> int:
    if policy_mode == "synthetic_operator":
        if task_id in {TaskId.TASK3, TaskId.TASK5}:
            return 2600
        if task_id == TaskId.TASK4:
            return 2300
        return 2200
    if task_id in {TaskId.TASK3, TaskId.TASK5}:
        return 1800
    if task_id == TaskId.TASK4:
        return 1600
    return 1400


def _resolve_provider_model(args: argparse.Namespace) -> str:
    """Return the model that should actually make completions for this run."""
    if args.policy == "synthetic_operator":
        return str(getattr(args, "persona_model", DEFAULT_PERSONA_MODEL))
    return str(getattr(args, "model", DEFAULT_AGENT_MODEL))


def _display_model_name(policy: str, model: str, persona_model: str | None = None) -> str:
    if policy == "model":
        return model
    if policy == "synthetic_operator":
        resolved_persona = persona_model or DEFAULT_PERSONA_MODEL
        return f"persona::{resolved_persona}"
    return policy


def _save_checkpoint(path: Path, checkpoint: CheckpointState) -> None:
    path.write_text(checkpoint.model_dump_json(indent=2), encoding="utf-8")


def _model_action(
    provider: GradientChatProvider,
    env: LatentGoalOpsEnvironment,
    observation,
    policy_mode: str = "model",
    operator_profile=None,
) -> tuple[LatentGoalOpsAction, dict, dict]:
    system = system_prompt(observation.task_id, policy_mode=policy_mode, operator_profile=operator_profile)
    user = user_prompt(
        observation.model_dump(mode="json"),
        policy_mode=policy_mode,
        operator_profile=operator_profile,
    )
    messages = [
        {
            "role": "system",
            "content": system,
        },
        {
            "role": "user",
            "content": user,
        },
    ]
    response = provider.complete(messages, max_tokens=_completion_budget(observation.task_id, policy_mode=policy_mode))
    parse_fallback = False
    parse_repaired = False
    guardrail_adjusted = False
    stabilized_with_fallback = False
    try:
        action = parse_action(response.content)
    except Exception:
        repair_messages = [
            {
                "role": "system",
                "content": (
                    "You repair invalid benchmark action outputs. "
                    "Return exactly one valid JSON object for the active task and nothing else."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task_prompt": user,
                        "previous_invalid_response": response.content,
                    },
                    indent=2,
                    sort_keys=True,
                ),
            },
        ]
        try:
            repair_response = provider.complete(
                repair_messages,
                max_tokens=min(900, _completion_budget(observation.task_id, policy_mode=policy_mode)),
            )
            response = type(response)(
                content=repair_response.content,
                input_tokens=response.input_tokens + repair_response.input_tokens,
                output_tokens=response.output_tokens + repair_response.output_tokens,
                cost_usd=response.cost_usd + repair_response.cost_usd,
                raw={"initial": response.raw, "repair": repair_response.raw},
            )
            action = parse_action(repair_response.content)
            parse_repaired = True
        except Exception:
            parse_fallback = True
            action = choose_action(env, "heuristic")
    if policy_mode == "synthetic_operator" and operator_profile is not None:
        action, guardrail_adjusted = apply_operator_guardrails(
            action,
            observation.model_dump(mode="json"),
            operator_profile,
        )
    elif policy_mode == "model" and action.task_id == TaskId.TASK2 and not parse_fallback:
        heuristic_floor = env.sample_heuristic_action()
        action, stabilized_with_fallback = stabilize_model_action(
            action,
            observation.model_dump(mode="json"),
            heuristic_floor,
        )
    return action, {
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "cost_usd": response.cost_usd,
        "parse_fallback": parse_fallback,
        "parse_repaired": parse_repaired,
    }, {
        "operator_profile": operator_profile.as_dict() if operator_profile is not None else None,
        "operator_guardrail_adjusted": guardrail_adjusted,
        "task2_visible_floor_applied": stabilized_with_fallback,
        "policy_mode": policy_mode,
    }


def run(args: argparse.Namespace) -> dict[str, float]:
    """Execute a baseline batch and return mean score by task."""
    load_dotenv(".env")
    tasks = _parse_tasks(args.tasks)
    seeds = _parse_seeds(args.seeds)
    provider_model = _resolve_provider_model(args)
    benchmark_model = str(getattr(args, "model", DEFAULT_AGENT_MODEL))
    persona_model = str(getattr(args, "persona_model", DEFAULT_PERSONA_MODEL))
    run_id = args.run_id or (
        f"{args.policy}_{provider_model}_{getattr(args, 'operator_style', 'na')}"
        if args.policy == "synthetic_operator"
        else f"{args.policy}_{benchmark_model}"
    )
    output_dir = Path(args.output_dir)
    logger = JsonlLogger(output_dir / "runs.jsonl")
    checkpoint_path = output_dir / "checkpoint.json"
    checkpoint = _load_checkpoint(checkpoint_path, run_id, provider_model, args.policy, tasks, seeds)
    run_started_perf = time.perf_counter()
    experiment_config = _experiment_config_from_args(args)
    provider = None if args.policy not in {"model", "synthetic_operator"} else GradientChatProvider(
        provider_model,
        temperature=float(args.temperature),
    )
    budget_cap = float(args.budget_cap)
    score_table: dict[str, list[float]] = {task: [] for task in tasks}
    wandb_logger = WandbLogger(
        enabled=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=run_id,
        group=args.wandb_group or f"{args.policy}-{','.join(tasks)}",
        tags=[tag for tag in (args.wandb_tags.split(",") if args.wandb_tags else []) if tag]
        + [args.policy, provider_model, getattr(args, "operator_style", "na")],
        config={
            "policy": args.policy,
            "model": benchmark_model,
            "provider_model": provider_model,
            "persona_model": persona_model if args.policy == "synthetic_operator" else None,
            "operator_style": getattr(args, "operator_style", "na"),
            "tasks": tasks,
            "seeds": seeds,
            "budget_cap": budget_cap,
            "output_dir": str(output_dir),
            "temperature": float(args.temperature),
            "experiment_config": experiment_config.model_dump(mode="json"),
        },
    )

    jobs = [(seed, task) for seed in seeds for task in tasks]
    jobs_total = len(jobs)
    parse_fallback_count = 0
    parse_repaired_count = 0
    step_global = 0

    try:
        for index, (seed, task_id) in enumerate(jobs[checkpoint.next_index :], start=checkpoint.next_index):
            episode_started_at = _utc_now_iso()
            episode_started_perf = time.perf_counter()
            env = LatentGoalOpsEnvironment(experiment_config=experiment_config)
            observation = env.reset(seed=seed, task_id=task_id)
            provider_usage = {
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "parse_fallback": False,
                "parse_repaired": False,
                "parse_fallback_steps": 0,
                "parse_repaired_steps": 0,
            }
            step_index = 0
            operator_profile = None
            if args.policy == "synthetic_operator":
                operator_profile = resolve_operator_persona(getattr(args, "operator_style", "auto"), seed, TaskId(task_id))

            while not observation.done:
                step_started_at = _utc_now_iso()
                step_started_perf = time.perf_counter()
                if args.policy in {"model", "synthetic_operator"}:
                    try:
                        action, usage, model_metadata = _model_action(
                            provider,
                            env,
                            observation,
                            policy_mode=args.policy,
                            operator_profile=operator_profile,
                        )
                    except (ProviderExhaustedError, ProviderCredentialError) as exc:
                        checkpoint.failure_mode = type(exc).__name__
                        checkpoint.next_index = index
                        _save_checkpoint(checkpoint_path, checkpoint)
                        raise
                else:
                    action = choose_action(env, args.policy)
                    usage = {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": 0.0,
                        "parse_fallback": False,
                        "parse_repaired": False,
                    }
                    model_metadata = {"operator_profile": None, "operator_guardrail_adjusted": False, "policy_mode": args.policy}

                provider_usage["input_tokens"] += usage["input_tokens"]
                provider_usage["output_tokens"] += usage["output_tokens"]
                provider_usage["cost_usd"] += usage["cost_usd"]
                provider_usage["parse_fallback"] = bool(provider_usage["parse_fallback"] or usage.get("parse_fallback", False))
                provider_usage["parse_repaired"] = bool(provider_usage["parse_repaired"] or usage.get("parse_repaired", False))
                provider_usage["parse_fallback_steps"] += int(bool(usage.get("parse_fallback", False)))
                provider_usage["parse_repaired_steps"] += int(bool(usage.get("parse_repaired", False)))
                parse_fallback_count += int(bool(usage.get("parse_fallback", False)))
                parse_repaired_count += int(bool(usage.get("parse_repaired", False)))
                checkpoint.cumulative_cost_usd += usage["cost_usd"]
                checkpoint.cumulative_input_tokens += usage["input_tokens"]
                checkpoint.cumulative_output_tokens += usage["output_tokens"]

                if checkpoint.cumulative_cost_usd > budget_cap:
                    checkpoint.failure_mode = "budget_cap_exceeded"
                    checkpoint.next_index = index
                    _save_checkpoint(checkpoint_path, checkpoint)
                    raise ProviderExhaustedError("Configured budget cap exceeded.")

                next_observation = env.step(action)
                step_log = StepLog(
                    started_at=step_started_at,
                    finished_at=_utc_now_iso(),
                    elapsed_seconds=time.perf_counter() - step_started_perf,
                    run_id=run_id,
                    task_id=task_id,
                    seed=seed,
                    policy=args.policy,
                    model_name=_display_model_name(args.policy, benchmark_model, persona_model=persona_model),
                    step_index=step_index,
                    action=action.model_dump(mode="json", exclude_none=True),
                    reward=float(next_observation.reward or 0.0),
                    done=bool(next_observation.done),
                    observation=next_observation.model_dump(mode="json"),
                    provider_usage=usage,
                    metadata={
                        "policy": args.policy,
                        "experiment_config": experiment_config.model_dump(mode="json"),
                        "strict_step": not bool(usage.get("parse_fallback", False) or usage.get("parse_repaired", False)),
                        "benchmark_model": benchmark_model if args.policy == "model" else None,
                        "persona_source_model": persona_model if args.policy == "synthetic_operator" else None,
                        **model_metadata,
                    },
                )
                logger.write_step(step_log)
                wandb_logger.log_step(step_log, episode_index=index, step_global=step_global)
                observation = next_observation
                step_index += 1
                step_global += 1

            grade = env.last_grade.model_dump(mode="json") if env.last_grade else {}
            score = float(grade.get("score", observation.reward or 0.0))
            score_table[task_id].append(score)
            elapsed_run_seconds = time.perf_counter() - run_started_perf
            episodes_completed = index + 1
            progress_fraction = episodes_completed / jobs_total if jobs_total else 1.0
            average_episode_seconds = elapsed_run_seconds / episodes_completed if episodes_completed else 0.0
            eta_seconds = average_episode_seconds * max(0, jobs_total - episodes_completed)
            episode_summary = EpisodeSummary(
                started_at=episode_started_at,
                finished_at=_utc_now_iso(),
                elapsed_seconds=time.perf_counter() - episode_started_perf,
                run_id=run_id,
                task_id=task_id,
                seed=seed,
                policy=args.policy,
                model_name=_display_model_name(args.policy, benchmark_model, persona_model=persona_model),
                score=score,
                total_reward=float(env.state.cumulative_reward),
                total_steps=env.state.step_count,
                provider_usage=provider_usage,
                grade=grade,
                metadata={
                    "policy": args.policy,
                    "experiment_config": experiment_config.model_dump(mode="json"),
                    "operator_profile": operator_profile.as_dict() if operator_profile is not None else None,
                    "benchmark_model": benchmark_model if args.policy == "model" else None,
                    "persona_source_model": persona_model if args.policy == "synthetic_operator" else None,
                    "strict_episode": not bool(provider_usage["parse_fallback"] or provider_usage["parse_repaired"]),
                    "rescued_episode": bool(provider_usage["parse_fallback"] or provider_usage["parse_repaired"]),
                    "progress_fraction": progress_fraction,
                    "episodes_completed": episodes_completed,
                    "episodes_total": jobs_total,
                    "eta_seconds": eta_seconds,
                    "run_elapsed_seconds": elapsed_run_seconds,
                },
            )
            logger.write_episode(episode_summary)
            wandb_logger.log_episode(episode_summary, episode_index=index)
            print(
                "[progress] "
                f"{episodes_completed}/{jobs_total} "
                f"task={task_id} seed={seed} "
                f"score={score:.4f} "
                f"episode_time={_format_seconds(episode_summary.elapsed_seconds)} "
                f"run_time={_format_seconds(elapsed_run_seconds)} "
                f"eta={_format_seconds(eta_seconds)}",
                flush=True,
            )
            checkpoint.next_index = index + 1
            checkpoint.failure_mode = None
            _save_checkpoint(checkpoint_path, checkpoint)
    finally:
        means = {task: (sum(scores) / len(scores) if scores else 0.0) for task, scores in score_table.items()}
        strict_score_table: dict[str, list[float]] = {task: [] for task in tasks}
        rescued_score_table: dict[str, list[float]] = {task: [] for task in tasks}
        episodes_df = []
        for jsonl_line in logger.path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(jsonl_line)
            if "total_steps" not in payload or "score" not in payload:
                continue
            task = payload["task_id"]
            score = float(payload["score"])
            if payload.get("metadata", {}).get("strict_episode", False):
                strict_score_table.setdefault(task, []).append(score)
            else:
                rescued_score_table.setdefault(task, []).append(score)
            episodes_df.append(payload)
        summary_payload = {
            "mean_scores": means,
            "strict_mean_scores": {
                task: (sum(scores) / len(scores) if scores else 0.0)
                for task, scores in strict_score_table.items()
            },
            "rescued_mean_scores": {
                task: (sum(scores) / len(scores) if scores else 0.0)
                for task, scores in rescued_score_table.items()
            },
            "episode_count": len(episodes_df),
            "parse_repaired_count": parse_repaired_count,
            "parse_fallback_count": parse_fallback_count,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        wandb_logger.log_summary(
            means,
            extra={
                "summary/run_id": run_id,
                "summary/ablation": experiment_config.slug(),
                "summary/run_started_at": checkpoint.started_at,
                "summary/run_elapsed_seconds": time.perf_counter() - run_started_perf,
                "summary/parse_repaired_count": parse_repaired_count,
                "summary/parse_fallback_count": parse_fallback_count,
                "summary/cumulative_cost_usd": checkpoint.cumulative_cost_usd,
                "summary/cumulative_input_tokens": checkpoint.cumulative_input_tokens,
                "summary/cumulative_output_tokens": checkpoint.cumulative_output_tokens,
            },
        )
        wandb_logger.finish()
    return means


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["model", "synthetic_operator", "random", "heuristic", "oracle"], default="heuristic")
    parser.add_argument("--model", default=DEFAULT_AGENT_MODEL)
    parser.add_argument("--persona-model", default=DEFAULT_PERSONA_MODEL)
    parser.add_argument("--operator-style", choices=operator_style_choices(), default="auto")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--seeds", default="100:3")
    parser.add_argument("--output-dir", default="outputs/baseline")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--budget-cap", type=float, default=20.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--disable-hidden-shift", action="store_true")
    parser.add_argument("--disable-delayed-effects", action="store_true")
    parser.add_argument("--hide-decision-ledger", action="store_true")
    parser.add_argument("--reward-mode", choices=["shaped", "sparse"], default="shaped")
    parser.add_argument("--task3-horizon-override", type=int, default=None)
    parser.add_argument("--scenario-split", choices=["core", "heldout"], default="core")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="latentgoalops")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", default="")
    args = parser.parse_args()
    means = run(args)
    print(json.dumps(means, indent=2))


if __name__ == "__main__":
    main()
