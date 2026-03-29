"""Interactive human-baseline trace collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latentgoalops.experiment import ExperimentConfig
from latentgoalops.models import (
    FeedbackLabel,
    ItemLabelAssignment,
    ItemPriorityAssignment,
    LatentGoalOpsAction,
    MessagingAction,
    SupportPolicy,
    TaskId,
)
from latentgoalops.server.environment import LatentGoalOpsEnvironment


def _input_with_default(prompt: str, default: str = "") -> str:
    raw = input(f"{prompt} ").strip()
    return raw or default


def _choose_enum(enum_values: list[tuple[str, str]], prompt: str, default_index: int = 0) -> str:
    print(prompt)
    for index, (_, label) in enumerate(enum_values, start=1):
        print(f"  {index}. {label}")
    choice = _input_with_default(f"Select [default {default_index + 1}]:", str(default_index + 1))
    selected = max(1, min(len(enum_values), int(choice))) - 1
    return enum_values[selected][0]


def _parse_ids(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _task1_action(observation) -> LatentGoalOpsAction:
    labels = [(label.value, label.value) for label in FeedbackLabel]
    label_assignments = []
    priority_assignments = []
    print("\nTask 1: label every item, set priorities, then choose escalations.")
    for item in observation.inbox:
        print(f"\n{item.item_id} | {item.sender}")
        print(item.text)
        label = _choose_enum(labels, "Choose label:")
        priority = int(_input_with_default("Priority 1-5 [default 3]:", "3"))
        label_assignments.append(ItemLabelAssignment(item_id=item.item_id, label=FeedbackLabel(label)))
        priority_assignments.append(ItemPriorityAssignment(item_id=item.item_id, priority=max(1, min(5, priority))))
    print("\nAvailable IDs for escalation:")
    print(", ".join(item.item_id for item in observation.inbox))
    escalate_ids = _parse_ids(_input_with_default("Escalate up to 3 IDs (comma-separated, blank for none):", ""))
    return LatentGoalOpsAction(
        task_id=TaskId.TASK1,
        labels=label_assignments,
        priorities=priority_assignments,
        escalate_ids=escalate_ids[:3],
    )


def _task2_action(observation) -> LatentGoalOpsAction:
    print("\nTask 2 backlog:")
    for item in observation.backlog:
        extras = []
        if item.requires_item_ids:
            extras.append(f"requires={','.join(item.requires_item_ids)}")
        if item.conflicts_with_ids:
            extras.append(f"conflicts={','.join(item.conflicts_with_ids)}")
        print(
            f"- {item.item_id} | cost={item.cost} | kind={item.kind} | "
            f"deltas={item.kpi_deltas} {' '.join(extras)}"
        )
        if item.risk_notes:
            print(f"  notes: {' | '.join(item.risk_notes)}")
    selected_ids = _parse_ids(
        _input_with_default("Choose item IDs within budget (comma-separated):", "")
    )
    rationale = _input_with_default("Short rationale:", "Human roadmap selection.")
    return LatentGoalOpsAction(
        task_id=TaskId.TASK2,
        selected_item_ids=selected_ids,
        rationale_summary=rationale,
    )


def _task4_action(observation) -> LatentGoalOpsAction:
    print("\nTask 4 capital programs:")
    for item in observation.backlog:
        print(
            f"- {item.item_id} | kind={item.kind} | alloc_max={item.allocation_max} | "
            f"saturation={item.saturation_point} | deltas={item.kpi_deltas}"
        )
        if item.risk_notes:
            print(f"  notes: {' | '.join(item.risk_notes)}")
    allocations = {}
    for item in observation.backlog:
        raw = _input_with_default(
            f"Allocate budget points to {item.item_id} [0-{int(item.allocation_max or 0)}] (blank for 0):",
            "0",
        )
        amount = max(0, min(int(raw), int(item.allocation_max or 0)))
        if amount > 0:
            allocations[item.item_id] = float(amount)
    rationale = _input_with_default("Short rationale:", "Human capital allocation.")
    return LatentGoalOpsAction(
        task_id=TaskId.TASK4,
        budget_allocations=allocations,
        rationale_summary=rationale,
    )


def _task3_like_action(observation, task_id: TaskId) -> LatentGoalOpsAction:
    label = observation.sim_day_label or "Decision"
    print(f"\n{label} | budget={observation.budget_remaining} | capacity={observation.capacity_remaining}")
    if observation.narrative:
        print(observation.narrative)
    print("\nVisible initiatives:")
    for item in observation.backlog:
        extras = []
        if item.requires_item_ids:
            extras.append(f"requires={','.join(item.requires_item_ids)}")
        if item.conflicts_with_ids:
            extras.append(f"conflicts={','.join(item.conflicts_with_ids)}")
        if item.synergy_item_ids:
            extras.append(f"synergy={','.join(item.synergy_item_ids)}")
        print(
            f"- {item.item_id} | cost={item.cost} | kind={item.kind} | "
            f"deltas={item.kpi_deltas} {' '.join(extras)}"
        )
        if item.risk_notes:
            print(f"  notes: {' | '.join(item.risk_notes)}")
    chosen_ids = _parse_ids(_input_with_default("Choose initiative IDs (comma-separated, blank for none):", ""))
    messaging = _choose_enum(
        [(value.value, value.value) for value in MessagingAction],
        "Choose messaging action:",
        default_index=4,
    )
    pricing = float(_input_with_default("Pricing change pct [-0.20, 0.20] [default 0.0]:", "0.0"))
    support = _choose_enum(
        [(value.value, value.value) for value in SupportPolicy],
        "Choose support policy:",
        default_index=1,
    )
    rationale = _input_with_default("Short rationale:", "Human startup-week action.")
    return LatentGoalOpsAction(
        task_id=task_id,
        chosen_initiatives=chosen_ids,
        messaging_action=MessagingAction(messaging),
        pricing_change_pct=max(-0.20, min(0.20, pricing)),
        support_policy=SupportPolicy(support),
        rationale=rationale,
    )


def _interactive_action(observation) -> LatentGoalOpsAction:
    task_id = TaskId(observation.task_id)
    if task_id == TaskId.TASK1:
        return _task1_action(observation)
    if task_id == TaskId.TASK2:
        return _task2_action(observation)
    if task_id == TaskId.TASK4:
        return _task4_action(observation)
    return _task3_like_action(observation, task_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--participant", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", default="outputs/human")
    parser.add_argument("--raw-json", action="store_true")
    parser.add_argument("--disable-hidden-shift", action="store_true")
    parser.add_argument("--disable-delayed-effects", action="store_true")
    parser.add_argument("--hide-decision-ledger", action="store_true")
    parser.add_argument("--reward-mode", choices=["shaped", "sparse"], default="shaped")
    parser.add_argument("--task3-horizon-override", type=int, default=None)
    parser.add_argument("--scenario-split", choices=["core", "heldout"], default="core")
    args = parser.parse_args()

    config = ExperimentConfig(
        enable_hidden_shift=not args.disable_hidden_shift,
        enable_delayed_effects=not args.disable_delayed_effects,
        expose_decision_ledger=not args.hide_decision_ledger,
        reward_mode=args.reward_mode,
        task3_horizon_override=args.task3_horizon_override,
        scenario_split=args.scenario_split,
    )
    env = LatentGoalOpsEnvironment(experiment_config=config)
    observation = env.reset(seed=args.seed, task_id=args.task)

    trace_dir = Path(args.output_dir) / args.participant / f"{args.task}_seed{args.seed}"
    trace_dir.mkdir(parents=True, exist_ok=True)
    records = []

    while True:
        step_path = trace_dir / f"observation_step{observation.step_index}.json"
        step_path.write_text(json.dumps(observation.model_dump(mode="json"), indent=2), encoding="utf-8")
        print(f"\nObservation written to {step_path}")

        if args.raw_json:
            print("Paste one JSON action object on a single line. Type 'quit' to stop.")
            raw = input("> ").strip()
            if raw.lower() == "quit":
                break
            action = LatentGoalOpsAction.model_validate_json(raw)
        else:
            action = _interactive_action(observation)

        records.append(
            {
                "task_id": args.task,
                "seed": args.seed,
                "participant": args.participant,
                "step_index": observation.step_index,
                "observation": observation.model_dump(mode="json"),
                "action": action.model_dump(mode="json", exclude_none=True),
            }
        )
        observation = env.step(action)
        if observation.done:
            records.append(
                {
                    "task_id": args.task,
                    "seed": args.seed,
                    "participant": args.participant,
                    "final_observation": observation.model_dump(mode="json"),
                    "final_score": env.last_grade.score if env.last_grade else observation.reward,
                }
            )
            break

    trace_path = trace_dir / "trace.json"
    trace_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\nSaved trace to {trace_path}")


if __name__ == "__main__":
    main()
