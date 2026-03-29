"""Run the core ablation suite for LatentGoalOps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv

from latentgoalops.analysis.aggregate import load_episode_summaries
from latentgoalops.analysis.stats import bootstrap_mean_ci
from latentgoalops.baseline.run_baseline import DEFAULT_AGENT_MODEL, DEFAULT_PERSONA_MODEL, run
from latentgoalops.experiment import ExperimentConfig


ABLATONS = {
    "full": ExperimentConfig(),
    "no_shift": ExperimentConfig(enable_hidden_shift=False),
    "no_delay": ExperimentConfig(enable_delayed_effects=False),
    "no_ledger": ExperimentConfig(expose_decision_ledger=False),
    "sparse_reward": ExperimentConfig(reward_mode="sparse"),
}


DEFAULT_BENCHMARK_MODELS = [
    DEFAULT_AGENT_MODEL,
    "openai-gpt-oss-120b",
]


def _parse_models(raw: str, policy: str, persona_model: str) -> list[str]:
    if raw == "default":
        if policy == "synthetic_operator":
            return [persona_model]
        return DEFAULT_BENCHMARK_MODELS
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_ablation_names(raw: str) -> list[str]:
    if raw == "default":
        return list(ABLATONS.keys())
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["model", "synthetic_operator"], default="model")
    parser.add_argument("--models", default="default")
    parser.add_argument("--persona-model", default=DEFAULT_PERSONA_MODEL)
    parser.add_argument("--operator-style", default="auto")
    parser.add_argument("--ablations", default="default")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--seeds", default="100:10")
    parser.add_argument("--output-root", default="outputs/ablations")
    parser.add_argument("--budget-cap", type=float, default=20.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--scenario-split", choices=["core", "heldout"], default="core")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="latentgoalops")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default="ablations")
    parser.add_argument("--wandb-tags", default="ablation")
    args = parser.parse_args()

    load_dotenv(".env")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    models = _parse_models(args.models, args.policy, args.persona_model)
    ablation_names = _parse_ablation_names(args.ablations)

    for ablation_name in ablation_names:
        config = ABLATONS[ablation_name]
        for model in models:
            safe_model = model.replace("/", "-")
            output_dir = output_root / ablation_name / safe_model
            run(
                SimpleNamespace(
                    policy=args.policy,
                    model=model,
                    persona_model=model if args.policy == "synthetic_operator" else args.persona_model,
                    operator_style=args.operator_style,
                    tasks=args.tasks,
                    seeds=args.seeds,
                    output_dir=str(output_dir),
                    run_id=f"{ablation_name}-{safe_model}",
                    budget_cap=args.budget_cap,
                    temperature=args.temperature,
                    disable_hidden_shift=not config.enable_hidden_shift,
                    disable_delayed_effects=not config.enable_delayed_effects,
                    hide_decision_ledger=not config.expose_decision_ledger,
                    reward_mode=config.reward_mode,
                    task3_horizon_override=config.task3_horizon_override,
                    scenario_split=args.scenario_split,
                    wandb=args.wandb,
                    wandb_project=args.wandb_project,
                    wandb_entity=args.wandb_entity,
                    wandb_group=args.wandb_group,
                    wandb_tags=f"{args.wandb_tags},{ablation_name}",
                )
            )

    episodes = load_episode_summaries(output_root)
    report: dict[str, dict] = {}
    if not episodes.empty:
        episodes["ablation"] = episodes["run_id"].str.split("-").str[0]
        for (ablation, model_name, task_id), group in episodes.groupby(["ablation", "model_name", "task_id"]):
            scores = [float(value) for value in group["score"].tolist()]
            ci_low, ci_high = bootstrap_mean_ci(scores, samples=2000)
            report.setdefault(ablation, {}).setdefault(model_name, {})[task_id] = {
                "mean_score": sum(scores) / len(scores),
                "ci95": [ci_low, ci_high],
                "count": len(scores),
            }
    (output_root / "ablation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
