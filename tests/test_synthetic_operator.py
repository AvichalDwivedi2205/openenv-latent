"""Synthetic operator baseline tests."""

from latentgoalops.baseline.prompts import system_prompt, user_prompt
from latentgoalops.baseline.synthetic_operator import (
    apply_operator_guardrails,
    resolve_operator_persona,
    stabilize_model_action,
)
from latentgoalops.models import LatentGoalOpsAction, SupportPolicy, TaskId


def test_operator_persona_is_deterministic():
    persona_a = resolve_operator_persona("auto", 21, TaskId.TASK3)
    persona_b = resolve_operator_persona("auto", 21, TaskId.TASK3)
    assert persona_a == persona_b


def test_operator_guardrails_clamp_risky_task3_actions():
    persona = resolve_operator_persona("support", 7, TaskId.TASK3)
    action = LatentGoalOpsAction(
        task_id=TaskId.TASK3,
        chosen_initiatives=["a", "b", "c"],
        pricing_change_pct=0.08,
        support_policy=SupportPolicy.AUTOMATION_FIRST,
    )
    observation = {
        "task_id": TaskId.TASK3.value,
        "accounts": [
            {
                "account_id": "acct_1",
                "segment": "strategic",
                "support_tier": "premium",
                "renewal_window_days": 12,
            }
        ],
        "governance_constraints": [
            {"constraint_id": "pricing_guardrail"},
            {"constraint_id": "sla_guardrail"},
        ],
    }
    guarded, adjusted = apply_operator_guardrails(action, observation, persona)
    assert adjusted is True
    assert guarded.chosen_initiatives == []
    assert guarded.pricing_change_pct <= 0.02
    assert guarded.support_policy == SupportPolicy.BALANCED_TRIAGE


def test_synthetic_operator_prompts_include_persona_constraints():
    persona = resolve_operator_persona("finance", 9, TaskId.TASK2)
    system = system_prompt(TaskId.TASK2, policy_mode="synthetic_operator", operator_profile=persona)
    user = user_prompt(
        {
            "task_id": TaskId.TASK2.value,
            "task_summary": "test",
            "dashboard": {"dau": 1, "d30_retention": 0.2, "mrr": 10, "ops_margin": 0.4, "support_ticket_volume": 5},
            "sprint_budget": 8,
            "market_context": {"cash_runway_months": 12, "board_pressure_level": 0.5, "gross_margin_target": 0.55},
            "stakeholder_notes": ["note"],
            "governance_constraints": [],
            "accounts": [],
            "backlog": [],
        },
        policy_mode="synthetic_operator",
        operator_profile=persona,
    )
    assert "synthetic startup operator" in system
    assert "max_parallel_bets" in user


def test_operator_guardrails_fill_underbuilt_task2_plan():
    persona = resolve_operator_persona("gm", 3, TaskId.TASK2)
    action = LatentGoalOpsAction(
        task_id=TaskId.TASK2,
        selected_item_ids=["item_a"],
    )
    observation = {
        "task_id": TaskId.TASK2.value,
        "sprint_budget": 8,
        "backlog": [
            {"item_id": "item_a", "cost": 2, "kpi_deltas": {"retention": 0.04}, "implementation_risk": 0.1},
            {"item_id": "item_b", "cost": 2, "kpi_deltas": {"revenue": 0.06}, "implementation_risk": 0.1},
            {"item_id": "item_c", "cost": 2, "kpi_deltas": {"growth": 0.05}, "implementation_risk": 0.2},
        ],
    }
    guarded, adjusted = apply_operator_guardrails(action, observation, persona)
    assert adjusted is True
    assert len(guarded.selected_item_ids) >= 2


def test_operator_guardrails_clip_task4_allocations():
    persona = resolve_operator_persona("finance", 11, TaskId.TASK4)
    action = LatentGoalOpsAction(
        task_id=TaskId.TASK4,
        budget_allocations={"prog_a": 5, "prog_b": 4, "prog_c": 3},
    )
    observation = {
        "task_id": TaskId.TASK4.value,
        "sprint_budget": 6,
        "backlog": [
            {"item_id": "prog_a", "allocation_max": 3, "saturation_point": 2, "kpi_deltas": {"revenue": 0.06}, "implementation_risk": 0.1},
            {"item_id": "prog_b", "allocation_max": 3, "saturation_point": 2, "kpi_deltas": {"efficiency": 0.05}, "implementation_risk": 0.1},
            {"item_id": "prog_c", "allocation_max": 3, "saturation_point": 2, "kpi_deltas": {"growth": 0.05}, "implementation_risk": 0.2},
        ],
    }
    guarded, adjusted = apply_operator_guardrails(action, observation, persona)
    assert adjusted is True
    assert sum(guarded.budget_allocations.values()) <= 6
    assert len(guarded.budget_allocations) <= persona.max_parallel_bets


def test_task2_visible_floor_prefers_stronger_heuristic_bundle():
    weak_action = LatentGoalOpsAction(
        task_id=TaskId.TASK2,
        selected_item_ids=["growth_only"],
        rationale_summary="Weak single bet.",
    )
    heuristic_action = LatentGoalOpsAction(
        task_id=TaskId.TASK2,
        selected_item_ids=["retention_core", "revenue_companion"],
        rationale_summary="Visible-context heuristic bundle.",
    )
    observation = {
        "task_id": TaskId.TASK2.value,
        "sprint_budget": 8,
        "backlog": [
            {
                "item_id": "growth_only",
                "cost": 2,
                "kind": "growth",
                "kpi_deltas": {"growth": 0.04},
                "implementation_risk": 0.35,
                "beneficiary_segments": ["self_serve"],
                "beneficiary_account_ids": [],
                "requires_item_ids": [],
                "conflicts_with_ids": [],
                "synergy_item_ids": [],
            },
            {
                "item_id": "retention_core",
                "cost": 3,
                "kind": "retention",
                "kpi_deltas": {"retention": 0.08, "revenue": 0.02},
                "implementation_risk": 0.1,
                "beneficiary_segments": ["enterprise", "strategic"],
                "beneficiary_account_ids": ["acct_1"],
                "requires_item_ids": [],
                "conflicts_with_ids": [],
                "synergy_item_ids": ["revenue_companion"],
            },
            {
                "item_id": "revenue_companion",
                "cost": 3,
                "kind": "revenue",
                "kpi_deltas": {"revenue": 0.06, "retention": 0.01},
                "implementation_risk": 0.1,
                "beneficiary_segments": ["enterprise"],
                "beneficiary_account_ids": ["acct_2"],
                "requires_item_ids": [],
                "conflicts_with_ids": [],
                "synergy_item_ids": ["retention_core"],
            },
        ],
    }
    stabilized, adjusted = stabilize_model_action(weak_action, observation, heuristic_action)
    assert adjusted is True
    assert stabilized.selected_item_ids == heuristic_action.selected_item_ids
