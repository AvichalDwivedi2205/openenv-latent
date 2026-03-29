"""Core task smoke tests."""

from latentgoalops.experiment import ExperimentConfig
from latentgoalops.server.environment import LatentGoalOpsEnvironment


def test_task1_oracle_scores_perfectly():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=7, task_id="task1_feedback_triage")
    assert observation.accounts
    assert observation.stakeholders
    assert observation.market_context is not None
    assert observation.governance_constraints
    assert "annual_contract_value" in observation.inbox[0].metadata
    observation = env.step(env.sample_oracle_action())
    assert observation.done is True
    assert env.last_grade is not None
    assert env.last_grade.score == 1.0


def test_task2_oracle_scores_perfectly():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=7, task_id="task2_roadmap_priority")
    assert observation.accounts
    assert observation.stakeholders
    assert observation.market_context is not None
    assert observation.governance_constraints
    assert observation.backlog[0].beneficiary_segments
    assert observation.backlog[0].policy_tags is not None
    observation = env.step(env.sample_oracle_action())
    assert observation.done is True
    assert env.last_grade is not None
    assert env.last_grade.score == 1.0


def test_task2_notes_do_not_name_hidden_goal():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=9, task_id="task2_roadmap_priority")
    hidden_goal = env._hidden_goal
    assert hidden_goal is not None
    combined_notes = " ".join(observation.stakeholder_notes).lower()
    assert hidden_goal.archetype.value not in combined_notes


def test_task2_backlog_exposes_bundle_structure():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=10, task_id="task2_roadmap_priority")
    assert any(
        item.requires_item_ids or item.conflicts_with_ids or item.synergy_item_ids or item.risk_notes
        for item in observation.backlog
    )


def test_task2_heldout_split_uses_alt_initiative_family():
    env = LatentGoalOpsEnvironment(experiment_config=ExperimentConfig(scenario_split="heldout"))
    observation = env.reset(seed=14, task_id="task2_roadmap_priority")
    visible_ids = {item.item_id.rsplit("_", 1)[0] for item in observation.backlog}
    assert visible_ids
    assert any(
        item_id in {
            "partner_marketplace_seed",
            "developer_conversion_grants",
            "security_assurance_pool",
            "enterprise_procurement_desk",
        }
        for item_id in visible_ids
    )


def test_task3_runs_to_completion():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=7, task_id="task3_startup_week")
    assert observation.sim_date is not None
    assert observation.sim_day_label == "Day 1"
    assert isinstance(observation.calendar_events, list)
    assert isinstance(observation.pending_effects, list)
    assert isinstance(observation.decision_ledger, list)
    while not observation.done:
        observation = env.step(env.sample_heuristic_action())
    assert env.last_grade is not None
    assert 0.0 <= env.last_grade.score <= 1.0
    assert env.state.step_count == 10


def test_task3_temporal_ledger_populates_after_step():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=11, task_id="task3_startup_week")
    observation = env.step(env.sample_heuristic_action())
    assert observation.sim_date is not None
    assert observation.decision_ledger
    assert observation.decision_ledger[-1].sim_date is not None


def test_task3_visible_text_does_not_name_hidden_goal():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=12, task_id="task3_startup_week")
    hidden_goal = env._hidden_goal
    assert hidden_goal is not None
    visible_text = " ".join([observation.narrative or ""] + [item.text for item in observation.inbox]).lower()
    assert hidden_goal.archetype.value not in visible_text


def test_task3_accounts_and_constraints_are_visible():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=15, task_id="task3_startup_week")
    assert observation.accounts
    assert observation.stakeholders
    assert observation.teams
    assert observation.market_context is not None
    assert observation.governance_constraints


def test_task3_backlog_exposes_bundle_structure():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=16, task_id="task3_startup_week")
    assert any(
        item.requires_item_ids or item.conflicts_with_ids or item.synergy_item_ids or item.risk_notes
        for item in observation.backlog
    )


def test_task3_account_renewal_windows_progress_over_time():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=17, task_id="task3_startup_week")
    before = {account.account_id: account.renewal_window_days for account in observation.accounts}
    observation = env.step(env.sample_heuristic_action())
    after = {account.account_id: account.renewal_window_days for account in observation.accounts}
    shared_ids = set(before) & set(after)
    assert shared_ids
    assert any(after[account_id] <= before[account_id] - 1 for account_id in shared_ids)


def test_task3_ablation_can_hide_ledger():
    env = LatentGoalOpsEnvironment(experiment_config=ExperimentConfig(expose_decision_ledger=False))
    observation = env.reset(seed=12, task_id="task3_startup_week")
    observation = env.step(env.sample_heuristic_action())
    assert observation.decision_ledger == []
    assert observation.pending_effects == []
    assert observation.realized_effects == []


def test_task3_sparse_reward_only_pays_on_terminal_step():
    env = LatentGoalOpsEnvironment(experiment_config=ExperimentConfig(reward_mode="sparse"))
    observation = env.reset(seed=13, task_id="task3_startup_week")
    while not observation.done:
        observation = env.step(env.sample_heuristic_action())
        if not observation.done:
            assert observation.reward == 0.0
    assert observation.reward is not None
    assert observation.reward >= 0.0


def test_task4_oracle_scores_perfectly():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=19, task_id="task4_capital_allocation")
    assert observation.backlog
    assert observation.sprint_budget is not None
    assert observation.stakeholder_notes
    observation = env.step(env.sample_oracle_action())
    assert observation.done is True
    assert env.last_grade is not None
    assert env.last_grade.score == 1.0


def test_task4_backlog_exposes_allocation_fields():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=20, task_id="task4_capital_allocation")
    assert all(item.allocation_max is not None and item.saturation_point is not None for item in observation.backlog)


def test_task5_runs_to_completion():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=21, task_id="task5_crisis_response")
    assert observation.narrative is not None
    assert observation.inbox
    observation = env.step(env.sample_heuristic_action())
    assert observation.done is True
    assert env.last_grade is not None
    assert 0.0 <= env.last_grade.score <= 1.0


def test_task5_visible_text_does_not_name_hidden_goal():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=22, task_id="task5_crisis_response")
    hidden_goal = env._hidden_goal
    assert hidden_goal is not None
    visible_text = " ".join([observation.narrative or ""] + [item.text for item in observation.inbox]).lower()
    assert hidden_goal.archetype.value not in visible_text
