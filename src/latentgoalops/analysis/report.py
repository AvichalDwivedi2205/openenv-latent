"""CLI to generate a paper-friendly plot bundle from run logs."""

from __future__ import annotations

import argparse
from pathlib import Path

from latentgoalops.analysis.aggregate import load_run_records
from latentgoalops.analysis.plots import (
    plot_adaptation_distribution,
    plot_coherence_vs_score,
    plot_cost_vs_score,
    plot_heatmap,
    plot_oracle_gap,
    plot_parse_fallbacks,
    plot_score_distributions,
    plot_step_reward_trajectories,
    plot_task_scores,
    plot_token_usage,
)
from latentgoalops.analysis.tables import write_summary_tables


def generate_report(input_path: str, output_dir: str) -> None:
    """Generate a plot bundle from one log file or a directory tree of runs."""
    steps, episodes = load_run_records(input_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    plot_task_scores(episodes, output_root / "task_scores_bar.png")
    plot_score_distributions(episodes, output_root / "score_distributions_box.png")
    plot_heatmap(episodes, output_root / "score_heatmap.png")
    plot_cost_vs_score(episodes, output_root / "cost_vs_score.png")
    plot_step_reward_trajectories(steps, output_root / "reward_trajectories.png")
    plot_token_usage(episodes, output_root / "token_usage.png")
    plot_parse_fallbacks(episodes, output_root / "parse_fallbacks.png")
    plot_adaptation_distribution(episodes, output_root / "adaptation_distribution.png")
    plot_coherence_vs_score(episodes, output_root / "coherence_vs_score.png")
    plot_oracle_gap(episodes, output_root / "oracle_gap.png")
    write_summary_tables(episodes, output_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs", help="A runs.jsonl file or a directory containing many run outputs.")
    parser.add_argument("--output-dir", default="outputs/plots")
    args = parser.parse_args()
    generate_report(args.input, args.output_dir)
    print(f"Generated plots under {args.output_dir}")


if __name__ == "__main__":
    main()
