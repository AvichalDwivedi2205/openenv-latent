"""Build trajectory audit packets for manual review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latentgoalops.analysis.aggregate import load_run_records


def _episode_key(row: dict) -> tuple[str, str, int]:
    return (str(row["model_name"]), str(row["task_id"]), int(row["seed"]))


def _pick_examples(episodes):
    selections = []
    if episodes.empty:
        return selections
    for (model_name, task_id), group in episodes.groupby(["model_name", "task_id"]):
        ranked = group.sort_values("score").reset_index(drop=True)
        selections.append(ranked.iloc[0].to_dict())
        selections.append(ranked.iloc[len(ranked) // 2].to_dict())
        selections.append(ranked.iloc[-1].to_dict())
    deduped = {}
    for row in selections:
        deduped[(row["model_name"], row["task_id"], row["seed"])] = row
    return list(deduped.values())


def _write_packet(output_dir: Path, episode_row: dict, step_rows: list[dict]) -> None:
    safe_name = f"{episode_row['model_name']}_{episode_row['task_id']}_seed{episode_row['seed']}".replace("/", "-")
    packet_dir = output_dir / safe_name
    packet_dir.mkdir(parents=True, exist_ok=True)
    (packet_dir / "episode_summary.json").write_text(json.dumps(episode_row, indent=2), encoding="utf-8")
    (packet_dir / "steps.json").write_text(json.dumps(step_rows, indent=2), encoding="utf-8")

    lines = [
        f"# Audit Packet: {episode_row['model_name']} / {episode_row['task_id']} / seed {episode_row['seed']}",
        "",
        f"- score: {episode_row['score']}",
        f"- total_reward: {episode_row['total_reward']}",
        f"- total_steps: {episode_row['total_steps']}",
        f"- cost_usd: {episode_row.get('cost_usd', 0.0)}",
        "",
        "## Grader",
        "",
        "```json",
        json.dumps(episode_row.get("grade", {}), indent=2),
        "```",
        "",
        "## Trajectory",
        "",
    ]
    for step in step_rows:
        observation = step.get("observation", {})
        lines.extend(
            [
                f"### Step {step['step_index']} ({observation.get('sim_date', 'n/a')})",
                "",
                f"- reward: {step['reward']}",
                f"- elapsed_seconds: {step.get('elapsed_seconds', 0.0)}",
                f"- parse_repaired: {step.get('parse_repaired', 0)}",
                f"- parse_fallback: {step.get('parse_fallback', 0)}",
                "",
                "```json",
                json.dumps(
                    {
                        "action": step.get("action", {}),
                        "alerts": observation.get("alerts", []),
                        "realized_effects": observation.get("realized_effects", []),
                        "pending_effects": observation.get("pending_effects", []),
                        "decision_ledger": observation.get("decision_ledger", []),
                    },
                    indent=2,
                ),
                "```",
                "",
            ]
        )
    (packet_dir / "audit.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs")
    parser.add_argument("--output-dir", default="outputs/audit_packets")
    args = parser.parse_args()

    steps, episodes = load_run_records(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for episode_row in _pick_examples(episodes):
        step_rows = []
        if not steps.empty:
            mask = (
                (steps["model_name"] == episode_row["model_name"])
                & (steps["task_id"] == episode_row["task_id"])
                & (steps["seed"] == episode_row["seed"])
            )
            step_rows = steps[mask].sort_values("step_index").to_dict(orient="records")
        _write_packet(output_dir, episode_row, step_rows)
    print(f"Generated audit packets under {output_dir}")


if __name__ == "__main__":
    main()
