"""Launch a sequential model sweep with consistent logging and grouping."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv

from latentgoalops.baseline.run_baseline import DEFAULT_AGENT_MODEL, DEFAULT_PERSONA_MODEL, run

DEFAULT_BENCHMARK_MODELS = [
    DEFAULT_AGENT_MODEL,
    "openai-gpt-oss-120b",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_models(raw: str, policy: str, persona_model: str) -> list[str]:
    if raw == "default":
        if policy == "synthetic_operator":
            return [persona_model]
        return DEFAULT_BENCHMARK_MODELS
    return [model.strip() for model in raw.split(",") if model.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["model", "synthetic_operator"], default="model")
    parser.add_argument("--models", default="default")
    parser.add_argument("--persona-model", default=DEFAULT_PERSONA_MODEL)
    parser.add_argument("--operator-style", default="auto")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--seeds", default="100:10")
    parser.add_argument("--output-root", default="outputs/model-ladder")
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
    parser.add_argument("--wandb-group", default="model-ladder")
    parser.add_argument("--wandb-tags", default="model-ladder,validation")
    args = parser.parse_args()

    load_dotenv(".env")
    models = _parse_models(args.models, args.policy, args.persona_model)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now_iso()
    started_perf = time.perf_counter()
    aggregate: dict[str, dict[str, float] | str | list[str]] = {
        "started_at": started_at,
        "tasks": args.tasks,
        "seeds": args.seeds,
        "scenario_split": args.scenario_split,
        "models": models,
        "persona_models": models if args.policy == "synthetic_operator" else None,
    }

    for model in models:
        safe_name = model.replace("/", "-")
        output_dir = output_root / safe_name
        run_id = f"{safe_name}-{args.seeds.replace(':', '_')}"
        print(f"[model-ladder] starting model={model} output_dir={output_dir}", flush=True)
        model_started = time.perf_counter()
        summary = run(
            SimpleNamespace(
                policy=args.policy,
                model=model,
                persona_model=model if args.policy == "synthetic_operator" else args.persona_model,
                operator_style=args.operator_style,
                tasks=args.tasks,
                seeds=args.seeds,
                output_dir=str(output_dir),
                run_id=f"{args.policy}-{run_id}" if args.policy != "model" else run_id,
                budget_cap=args.budget_cap,
                temperature=args.temperature,
                disable_hidden_shift=args.disable_hidden_shift,
                disable_delayed_effects=args.disable_delayed_effects,
                hide_decision_ledger=args.hide_decision_ledger,
                reward_mode=args.reward_mode,
                task3_horizon_override=args.task3_horizon_override,
                scenario_split=args.scenario_split,
                wandb=args.wandb,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                wandb_group=args.wandb_group,
                wandb_tags=f"{args.wandb_tags},{safe_name}",
            )
        )
        aggregate[model] = {
            "elapsed_seconds": time.perf_counter() - model_started,
            "scores": summary,
        }
        aggregate_path = output_root / "aggregate_summary.json"
        aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(
            f"[model-ladder] finished model={model} "
            f"elapsed_seconds={aggregate[model]['elapsed_seconds']:.2f}",
            flush=True,
        )

    aggregate["finished_at"] = _utc_now_iso()
    aggregate["elapsed_seconds"] = time.perf_counter() - started_perf
    (output_root / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
