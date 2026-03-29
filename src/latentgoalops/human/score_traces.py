"""Replay and score collected human traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latentgoalops.experiment import ExperimentConfig
from latentgoalops.models import LatentGoalOpsAction
from latentgoalops.server.environment import LatentGoalOpsEnvironment


def _trace_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("trace.json"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/human")
    parser.add_argument("--output", default="outputs/human/human_scores.json")
    parser.add_argument("--disable-hidden-shift", action="store_true")
    parser.add_argument("--disable-delayed-effects", action="store_true")
    parser.add_argument("--hide-decision-ledger", action="store_true")
    parser.add_argument("--reward-mode", choices=["shaped", "sparse"], default="shaped")
    parser.add_argument("--task3-horizon-override", type=int, default=None)
    args = parser.parse_args()

    config = ExperimentConfig(
        enable_hidden_shift=not args.disable_hidden_shift,
        enable_delayed_effects=not args.disable_delayed_effects,
        expose_decision_ledger=not args.hide_decision_ledger,
        reward_mode=args.reward_mode,
        task3_horizon_override=args.task3_horizon_override,
    )

    summaries = []
    for trace_path in _trace_paths(Path(args.input)):
        records = json.loads(trace_path.read_text(encoding="utf-8"))
        action_rows = [row for row in records if "action" in row]
        if not action_rows:
            continue
        task_id = action_rows[0]["task_id"]
        seed = int(action_rows[0]["seed"])
        participant = action_rows[0]["participant"]
        env = LatentGoalOpsEnvironment(experiment_config=config)
        env.reset(seed=seed, task_id=task_id)
        for row in action_rows:
            env.step(LatentGoalOpsAction.model_validate(row["action"]))
        summaries.append(
            {
                "participant": participant,
                "task_id": task_id,
                "seed": seed,
                "score": env.last_grade.score if env.last_grade else env.state.last_score,
                "trace_path": str(trace_path),
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
