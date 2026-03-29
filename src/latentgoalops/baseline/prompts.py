"""Prompt builders for model baselines."""

from __future__ import annotations

import json
from typing import Any

from latentgoalops.models import TaskId


def _shared_system_prefix() -> str:
    return (
        "You are an expert operator inside a research benchmark for startup operations. "
        "Your real task is not just to react locally, but to infer the latent business objective from indirect evidence "
        "such as KPI dashboards, stakeholder messages, backlog structure, alerts, and resource constraints. "
        "Maintain one coherent working hypothesis unless new evidence strongly contradicts it. "
        "Do not explain your reasoning. Return exactly one JSON object and nothing else."
    )


def _operator_prefix(operator_profile: Any | None) -> str:
    if operator_profile is None:
        return _shared_system_prefix()
    return (
        f"You are a synthetic startup operator acting as {operator_profile.label} ({operator_profile.role}). "
        "Do not optimize for benchmark score directly. Make realistic, defendable decisions under uncertainty. "
        f"Risk posture: {operator_profile.risk_posture}. Change tolerance: {operator_profile.change_tolerance}. "
        f"Keep plans focused: roadmap <= {operator_profile.max_parallel_bets} items, startup initiatives <= {operator_profile.max_parallel_initiatives}, "
        f"pricing moves roughly within +/-{operator_profile.max_abs_pricing_change:.2f}. "
        "Prefer continuity, lower-regret choices, and careful handling of premium or renewal-risk accounts. "
        "Do not explain hidden reasoning. Return exactly one JSON object and nothing else."
    )


def _task1_system_prompt(policy_mode: str = "model", operator_profile: Any | None = None) -> str:
    return (
        f"{_operator_prefix(operator_profile) if policy_mode == 'synthetic_operator' else _shared_system_prefix()} "
        "Task 1 is weighted feedback triage. Label every inbox item, set a 1-5 priority for every item, and escalate at most 3 IDs. "
        "Use the hidden objective only to influence prioritization and escalation, never to change the underlying label semantics. "
        "Pay attention to account economics such as annual_contract_value, renewal_window_days, support_tier, and relationship_health. "
        "Allowed labels: bug, feature_request, churn_risk, praise, billing_issue, latency_complaint. "
        "Output schema: "
        "{"
        "\"task_id\":\"task1_feedback_triage\","
        "\"labels\":[{\"item_id\":\"...\",\"label\":\"bug\"}],"
        "\"priorities\":[{\"item_id\":\"...\",\"priority\":3}],"
        "\"escalate_ids\":[\"...\"]"
        "}."
    )


def _task2_system_prompt(policy_mode: str = "model", operator_profile: Any | None = None) -> str:
    return (
        f"{_operator_prefix(operator_profile) if policy_mode == 'synthetic_operator' else _shared_system_prefix()} "
        "Task 2 is roadmap prioritization under a hard sprint budget. Select only visible backlog item IDs. "
        "Prefer a coherent bundle of initiatives that all support the same latent objective rather than a noisy mix of conflicting bets. "
        "Use customer segments, beneficiary accounts, stakeholder agendas, implementation risk, and governance constraints as part of the decision. "
        "Never exceed the budget. If uncertain, choose the highest visible KPI gain per unit cost that matches your inferred objective. "
        "Output schema: "
        "{"
        "\"task_id\":\"task2_roadmap_priority\","
        "\"selected_item_ids\":[\"...\"],"
        "\"rationale_summary\":\"one short sentence\""
        "}."
    )


def _task3_system_prompt(policy_mode: str = "model", operator_profile: Any | None = None) -> str:
    return (
        f"{_operator_prefix(operator_profile) if policy_mode == 'synthetic_operator' else _shared_system_prefix()} "
        "Task 3 is a startup operations simulation with an explicit calendar, prior decision ledger, pending delayed effects, "
        "and realized consequences from earlier choices. You must act coherently across dates, preserve budget and capacity, "
        "manage a portfolio of customer accounts, recurring stakeholder agendas, team execution constraints, and governance guardrails, "
        "and adapt if the evidence suggests the objective has silently shifted. "
        "Avoid mixing growth, retention, revenue, and efficiency actions without a clear reason. Prefer plans whose delayed effects remain aligned over multiple days. "
        "Choose only visible initiative IDs. Keep pricing_change_pct in [-0.20, 0.20]. "
        "Allowed messaging_action values: growth_push, retention_campaign, revenue_upsell, cost_comms, none. "
        "Allowed support_policy values: premium_sla, balanced_triage, automation_first, incident_swarm. "
        "Output schema: "
        "{"
        "\"task_id\":\"task3_startup_week\","
        "\"chosen_initiatives\":[\"...\"],"
        "\"messaging_action\":\"growth_push\","
        "\"pricing_change_pct\":0.0,"
        "\"support_policy\":\"balanced_triage\","
        "\"rationale\":\"one short sentence\""
        "}."
    )


def _task4_system_prompt(policy_mode: str = "model", operator_profile: Any | None = None) -> str:
    return (
        f"{_operator_prefix(operator_profile) if policy_mode == 'synthetic_operator' else _shared_system_prefix()} "
        "Task 4 is capital allocation under uncertainty. Allocate discrete budget points across visible program IDs using budget_allocations. "
        "Programs have visible allocation_max, saturation hints, dependencies, conflicts, and stakeholder pressure. "
        "Prefer one focused capital story rather than sprinkling tiny amounts across every program. "
        "Never exceed the total budget. Use integer budget points only. "
        "Output schema: "
        "{"
        "\"task_id\":\"task4_capital_allocation\","
        "\"budget_allocations\":{\"program_id\":2},"
        "\"rationale_summary\":\"one short sentence\""
        "}."
    )


def _task5_system_prompt(policy_mode: str = "model", operator_profile: Any | None = None) -> str:
    return (
        f"{_operator_prefix(operator_profile) if policy_mode == 'synthetic_operator' else _shared_system_prefix()} "
        "Task 5 is a one-shot crisis response package. Choose a small set of visible initiative IDs and combine them with pricing, messaging, and support policy. "
        "Treat it like one executive operating decision: the best package should be coherent, defendable, and safe for high-touch accounts. "
        "Avoid aggressive pricing or automation-heavy support when the visible evidence suggests enterprise trust is already fragile. "
        "Allowed messaging_action values: growth_push, retention_campaign, revenue_upsell, cost_comms, none. "
        "Allowed support_policy values: premium_sla, balanced_triage, automation_first, incident_swarm. "
        "Output schema: "
        "{"
        "\"task_id\":\"task5_crisis_response\","
        "\"chosen_initiatives\":[\"...\"],"
        "\"messaging_action\":\"retention_campaign\","
        "\"pricing_change_pct\":0.0,"
        "\"support_policy\":\"balanced_triage\","
        "\"rationale\":\"one short sentence\""
        "}."
    )


def system_prompt(task_id: TaskId, policy_mode: str = "model", operator_profile: Any | None = None) -> str:
    """Return the system prompt for a given task."""
    if task_id == TaskId.TASK1:
        return _task1_system_prompt(policy_mode=policy_mode, operator_profile=operator_profile)
    if task_id == TaskId.TASK2:
        return _task2_system_prompt(policy_mode=policy_mode, operator_profile=operator_profile)
    if task_id == TaskId.TASK3:
        return _task3_system_prompt(policy_mode=policy_mode, operator_profile=operator_profile)
    if task_id == TaskId.TASK4:
        return _task4_system_prompt(policy_mode=policy_mode, operator_profile=operator_profile)
    return _task5_system_prompt(policy_mode=policy_mode, operator_profile=operator_profile)


def _allowed_ids(observation: dict) -> list[str]:
    task_id = TaskId(observation["task_id"])
    if task_id == TaskId.TASK1:
        return [item["item_id"] for item in observation.get("inbox", [])]
    return [item["item_id"] for item in observation.get("backlog", [])]


def _json_template(task_id: TaskId) -> dict:
    if task_id == TaskId.TASK1:
        return {
            "task_id": "task1_feedback_triage",
            "labels": [{"item_id": "fb_1", "label": "bug"}],
            "priorities": [{"item_id": "fb_1", "priority": 3}],
            "escalate_ids": ["fb_1"],
        }
    if task_id == TaskId.TASK2:
        return {
            "task_id": "task2_roadmap_priority",
            "selected_item_ids": ["item_1"],
            "rationale_summary": "Short rationale.",
        }
    if task_id == TaskId.TASK3:
        return {
            "task_id": "task3_startup_week",
            "chosen_initiatives": ["optimize_infra"],
            "messaging_action": "cost_comms",
            "pricing_change_pct": 0.0,
            "support_policy": "automation_first",
            "rationale": "Short rationale.",
        }
    if task_id == TaskId.TASK4:
        return {
            "task_id": "task4_capital_allocation",
            "budget_allocations": {"renewal_rescue_pod": 3},
            "rationale_summary": "Short rationale.",
        }
    return {
        "task_id": "task5_crisis_response",
        "chosen_initiatives": ["fix_login_bug"],
        "messaging_action": "retention_campaign",
        "pricing_change_pct": 0.0,
        "support_policy": "balanced_triage",
        "rationale": "Short rationale.",
    }


def _dashboard_brief(observation: dict) -> dict:
    dashboard = observation.get("dashboard", {})
    return {
        "dau": dashboard.get("dau"),
        "d30_retention": dashboard.get("d30_retention"),
        "mrr": dashboard.get("mrr"),
        "ops_margin": dashboard.get("ops_margin"),
        "support_ticket_volume": dashboard.get("support_ticket_volume"),
    }


def _compact_task1_observation(observation: dict) -> dict:
    accounts_by_id = {account["account_id"]: account for account in observation.get("accounts", [])}
    return {
        "task_id": observation["task_id"],
        "task_summary": observation.get("task_summary"),
        "dashboard": _dashboard_brief(observation),
        "governance_constraints": [
            {
                "constraint_id": constraint.get("constraint_id"),
                "title": constraint.get("title"),
                "severity": constraint.get("severity"),
            }
            for constraint in observation.get("governance_constraints", [])
        ],
        "inbox": [
            {
                "item_id": item.get("item_id"),
                "sender": item.get("sender"),
                "text": item.get("text"),
                "severity": item.get("metadata", {}).get("severity"),
                "channel": item.get("metadata", {}).get("channel"),
                "account": {
                    "account_id": item.get("metadata", {}).get("account_id"),
                    "segment": item.get("metadata", {}).get("segment"),
                    "acv": item.get("metadata", {}).get("annual_contract_value"),
                    "renewal_window_days": item.get("metadata", {}).get("renewal_window_days"),
                    "support_tier": item.get("metadata", {}).get("support_tier"),
                    "relationship_health": item.get("metadata", {}).get("relationship_health"),
                    "strategic_importance": accounts_by_id.get(
                        item.get("metadata", {}).get("account_id"), {}
                    ).get("strategic_importance"),
                },
            }
            for item in observation.get("inbox", [])
        ],
    }


def _compact_task2_observation(observation: dict) -> dict:
    return {
        "task_id": observation["task_id"],
        "task_summary": observation.get("task_summary"),
        "dashboard": _dashboard_brief(observation),
        "sprint_budget": observation.get("sprint_budget"),
        "market_context": {
            "cash_runway_months": (observation.get("market_context") or {}).get("cash_runway_months"),
            "board_pressure_level": (observation.get("market_context") or {}).get("board_pressure_level"),
            "gross_margin_target": (observation.get("market_context") or {}).get("gross_margin_target"),
        },
        "stakeholder_notes": observation.get("stakeholder_notes", []),
        "governance_constraints": [
            {
                "constraint_id": constraint.get("constraint_id"),
                "title": constraint.get("title"),
                "severity": constraint.get("severity"),
            }
            for constraint in observation.get("governance_constraints", [])
        ],
        "accounts": [
            {
                "account_id": account.get("account_id"),
                "company_name": account.get("company_name"),
                "segment": account.get("segment"),
                "acv": account.get("annual_contract_value"),
                "renewal_window_days": account.get("renewal_window_days"),
                "relationship_health": account.get("relationship_health"),
            }
            for account in observation.get("accounts", [])[:5]
        ],
        "backlog": [
            {
                "item_id": item.get("item_id"),
                "kind": item.get("kind"),
                "cost": item.get("cost"),
                "kpi_deltas": item.get("kpi_deltas"),
                "beneficiary_segments": item.get("beneficiary_segments", []),
                "beneficiary_account_ids": item.get("beneficiary_account_ids", []),
                "implementation_risk": item.get("implementation_risk"),
                "policy_tags": item.get("policy_tags", []),
                "requires_item_ids": item.get("requires_item_ids", []),
                "conflicts_with_ids": item.get("conflicts_with_ids", []),
                "synergy_item_ids": item.get("synergy_item_ids", []),
                "risk_notes": item.get("risk_notes", []),
                "allocation_unit": item.get("allocation_unit"),
                "allocation_max": item.get("allocation_max"),
                "saturation_point": item.get("saturation_point"),
            }
            for item in observation.get("backlog", [])
        ],
    }


def _compact_task4_observation(observation: dict) -> dict:
    return {
        "task_id": observation["task_id"],
        "task_summary": observation.get("task_summary"),
        "dashboard": _dashboard_brief(observation),
        "sprint_budget": observation.get("sprint_budget"),
        "market_context": {
            "cash_runway_months": (observation.get("market_context") or {}).get("cash_runway_months"),
            "board_pressure_level": (observation.get("market_context") or {}).get("board_pressure_level"),
            "gross_margin_target": (observation.get("market_context") or {}).get("gross_margin_target"),
        },
        "stakeholder_notes": observation.get("stakeholder_notes", []),
        "accounts": [
            {
                "account_id": account.get("account_id"),
                "company_name": account.get("company_name"),
                "segment": account.get("segment"),
                "acv": account.get("annual_contract_value"),
                "renewal_window_days": account.get("renewal_window_days"),
                "relationship_health": account.get("relationship_health"),
            }
            for account in observation.get("accounts", [])[:6]
        ],
        "backlog": [
            {
                "item_id": item.get("item_id"),
                "kind": item.get("kind"),
                "allocation_unit": item.get("allocation_unit"),
                "allocation_max": item.get("allocation_max"),
                "saturation_point": item.get("saturation_point"),
                "kpi_deltas": item.get("kpi_deltas"),
                "beneficiary_segments": item.get("beneficiary_segments", []),
                "beneficiary_account_ids": item.get("beneficiary_account_ids", []),
                "implementation_risk": item.get("implementation_risk"),
                "requires_item_ids": item.get("requires_item_ids", []),
                "conflicts_with_ids": item.get("conflicts_with_ids", []),
                "synergy_item_ids": item.get("synergy_item_ids", []),
                "risk_notes": item.get("risk_notes", []),
            }
            for item in observation.get("backlog", [])
        ],
    }


def _compact_task3_observation(observation: dict) -> dict:
    return {
        "task_id": observation["task_id"],
        "task_summary": observation.get("task_summary"),
        "step_index": observation.get("step_index"),
        "horizon": observation.get("horizon"),
        "sim_date": observation.get("sim_date"),
        "sim_day_label": observation.get("sim_day_label"),
        "narrative": observation.get("narrative"),
        "dashboard": _dashboard_brief(observation),
        "budget_remaining": observation.get("budget_remaining"),
        "capacity_remaining": observation.get("capacity_remaining"),
        "market_context": {
            "cash_runway_months": (observation.get("market_context") or {}).get("cash_runway_months"),
            "board_pressure_level": (observation.get("market_context") or {}).get("board_pressure_level"),
            "sales_pipeline_health": (observation.get("market_context") or {}).get("sales_pipeline_health"),
            "competition_intensity": (observation.get("market_context") or {}).get("competition_intensity"),
        },
        "accounts": [
            {
                "account_id": account.get("account_id"),
                "company_name": account.get("company_name"),
                "segment": account.get("segment"),
                "acv": account.get("annual_contract_value"),
                "renewal_window_days": account.get("renewal_window_days"),
                "support_tier": account.get("support_tier"),
                "relationship_health": account.get("relationship_health"),
                "churn_propensity": account.get("churn_propensity"),
                "strategic_importance": account.get("strategic_importance"),
            }
            for account in observation.get("accounts", [])[:5]
        ],
        "stakeholders": [
            {
                "name": stakeholder.get("name"),
                "role": stakeholder.get("role"),
                "political_power": stakeholder.get("political_power"),
                "favorite_metrics": stakeholder.get("favorite_metrics", []),
            }
            for stakeholder in observation.get("stakeholders", [])[:4]
        ],
        "teams": [
            {
                "function": team.get("function"),
                "capacity": team.get("capacity"),
                "burnout_risk": team.get("burnout_risk"),
                "execution_reliability": team.get("execution_reliability"),
            }
            for team in observation.get("teams", [])
        ],
        "governance_constraints": [
            {
                "constraint_id": constraint.get("constraint_id"),
                "title": constraint.get("title"),
                "severity": constraint.get("severity"),
                "threshold": constraint.get("threshold"),
            }
            for constraint in observation.get("governance_constraints", [])
        ],
        "alerts": observation.get("alerts", []),
        "calendar_events": [
            {
                "name": event.get("name"),
                "sim_date": event.get("sim_date"),
                "status": event.get("status"),
                "summary": event.get("summary"),
            }
            for event in observation.get("calendar_events", [])
        ],
        "decision_ledger": [
            {
                "sim_date": entry.get("sim_date"),
                "chosen_initiatives": entry.get("chosen_initiatives", []),
                "messaging_action": entry.get("messaging_action"),
                "pricing_change_pct": entry.get("pricing_change_pct"),
                "support_policy": entry.get("support_policy"),
                "governance_flags": entry.get("governance_flags", []),
            }
            for entry in observation.get("decision_ledger", [])[-3:]
        ],
        "pending_effects": [
            {
                "summary": effect.get("summary"),
                "scheduled_for_date": effect.get("scheduled_for_date"),
                "affected_account_ids": effect.get("affected_account_ids", []),
                "affected_team_ids": effect.get("affected_team_ids", []),
            }
            for effect in observation.get("pending_effects", [])[:5]
        ],
        "realized_effects": [
            {
                "summary": effect.get("summary"),
                "realized_date": effect.get("realized_date"),
                "affected_account_ids": effect.get("affected_account_ids", []),
                "affected_team_ids": effect.get("affected_team_ids", []),
            }
            for effect in observation.get("realized_effects", [])[:5]
        ],
        "backlog": [
            {
                "item_id": item.get("item_id"),
                "kind": item.get("kind"),
                "cost": item.get("cost"),
                "kpi_deltas": item.get("kpi_deltas"),
                "beneficiary_segments": item.get("beneficiary_segments", []),
                "beneficiary_account_ids": item.get("beneficiary_account_ids", []),
                "implementation_risk": item.get("implementation_risk"),
                "policy_tags": item.get("policy_tags", []),
                "requires_item_ids": item.get("requires_item_ids", []),
                "conflicts_with_ids": item.get("conflicts_with_ids", []),
                "synergy_item_ids": item.get("synergy_item_ids", []),
                "risk_notes": item.get("risk_notes", []),
                "allocation_unit": item.get("allocation_unit"),
                "allocation_max": item.get("allocation_max"),
                "saturation_point": item.get("saturation_point"),
            }
            for item in observation.get("backlog", [])
        ],
    }


def _compact_task5_observation(observation: dict) -> dict:
    return {
        "task_id": observation["task_id"],
        "task_summary": observation.get("task_summary"),
        "narrative": observation.get("narrative"),
        "dashboard": _dashboard_brief(observation),
        "budget_remaining": observation.get("budget_remaining"),
        "capacity_remaining": observation.get("capacity_remaining"),
        "alerts": observation.get("alerts", []),
        "market_context": {
            "cash_runway_months": (observation.get("market_context") or {}).get("cash_runway_months"),
            "board_pressure_level": (observation.get("market_context") or {}).get("board_pressure_level"),
            "sales_pipeline_health": (observation.get("market_context") or {}).get("sales_pipeline_health"),
        },
        "inbox": [
            {
                "item_id": item.get("item_id"),
                "sender": item.get("sender"),
                "text": item.get("text"),
            }
            for item in observation.get("inbox", [])
        ],
        "accounts": [
            {
                "account_id": account.get("account_id"),
                "company_name": account.get("company_name"),
                "segment": account.get("segment"),
                "acv": account.get("annual_contract_value"),
                "renewal_window_days": account.get("renewal_window_days"),
                "support_tier": account.get("support_tier"),
                "relationship_health": account.get("relationship_health"),
            }
            for account in observation.get("accounts", [])[:6]
        ],
        "governance_constraints": [
            {
                "constraint_id": constraint.get("constraint_id"),
                "title": constraint.get("title"),
                "severity": constraint.get("severity"),
            }
            for constraint in observation.get("governance_constraints", [])
        ],
        "backlog": [
            {
                "item_id": item.get("item_id"),
                "kind": item.get("kind"),
                "cost": item.get("cost"),
                "kpi_deltas": item.get("kpi_deltas"),
                "beneficiary_segments": item.get("beneficiary_segments", []),
                "beneficiary_account_ids": item.get("beneficiary_account_ids", []),
                "implementation_risk": item.get("implementation_risk"),
                "policy_tags": item.get("policy_tags", []),
                "requires_item_ids": item.get("requires_item_ids", []),
                "conflicts_with_ids": item.get("conflicts_with_ids", []),
                "synergy_item_ids": item.get("synergy_item_ids", []),
                "risk_notes": item.get("risk_notes", []),
            }
            for item in observation.get("backlog", [])
        ],
    }


def user_prompt(observation: dict, policy_mode: str = "model", operator_profile: Any | None = None) -> str:
    """Serialize an observation as the user prompt."""
    task_id = TaskId(observation["task_id"])
    hints = [
        "Use only IDs that appear in the observation.",
        "Keep the policy internally coherent with one inferred objective.",
        "If evidence is ambiguous, prefer the strongest non-contradictory signal from dashboard + notes + alerts.",
        "Return one JSON object only with no markdown fences.",
    ]
    if policy_mode == "synthetic_operator" and operator_profile is not None:
        hints.append("Behave like a realistic human operator: do not over-rotate, over-parallelize, or use extreme pricing without strong evidence.")
        hints.append("Prefer one defendable operating plan over a benchmark-gaming move that a real leadership team would not take.")
    if task_id == TaskId.TASK2:
        hints.append("Do not exceed sprint_budget.")
        hints.append("Use beneficiary_account_ids, beneficiary_segments, implementation_risk, dependencies, conflicts, and stakeholder_notes to form one coherent portfolio bet.")
    if task_id == TaskId.TASK3:
        hints.append("Respect budget_remaining and capacity_remaining.")
        hints.append("If the narrative or alerts change sharply, consider that the hidden goal may have shifted.")
        hints.append("Use sim_date, calendar_events, decision_ledger, pending_effects, realized_effects, and visible initiative dependencies/conflicts to reason over delayed consequences.")
        hints.append("Use accounts, stakeholders, teams, market_context, and governance_constraints to understand who is affected and what is risky.")
        hints.append("Only backlog initiative IDs are actionable. Inbox or calendar IDs are context, never valid chosen_initiatives.")
        hints.append("Prefer decisions whose delayed effects stay coherent with the same latent objective over the next few dates.")
    if task_id == TaskId.TASK4:
        hints.append("Use integer budget points only and do not exceed sprint_budget.")
        hints.append("Prefer funding levels near visible saturation_point before spreading tiny allocations across too many programs.")
        hints.append("Use dependencies, conflicts, synergies, and stakeholder_notes to form one coherent capital narrative.")
    if task_id == TaskId.TASK5:
        hints.append("Respect both budget_remaining and capacity_remaining.")
        hints.append("Treat pricing, messaging, support policy, and initiatives as one package, not four unrelated choices.")
        hints.append("Avoid policies that visibly clash with enterprise renewals, premium support obligations, or margin pressure.")

    compact_observation = observation
    if task_id == TaskId.TASK1:
        compact_observation = _compact_task1_observation(observation)
    elif task_id == TaskId.TASK2:
        compact_observation = _compact_task2_observation(observation)
    elif task_id == TaskId.TASK3:
        compact_observation = _compact_task3_observation(observation)
    elif task_id == TaskId.TASK4:
        compact_observation = _compact_task4_observation(observation)
    elif task_id == TaskId.TASK5:
        compact_observation = _compact_task5_observation(observation)

    prompt_payload = {
        "decision_hints": hints,
        "allowed_item_ids": _allowed_ids(observation),
        "output_template": _json_template(task_id),
        "observation": compact_observation,
    }
    if policy_mode == "synthetic_operator" and operator_profile is not None:
        prompt_payload["operator_profile"] = {
            "persona_id": operator_profile.persona_id,
            "label": operator_profile.label,
            "risk_posture": operator_profile.risk_posture,
            "change_tolerance": operator_profile.change_tolerance,
            "default_focus": operator_profile.default_focus,
            "max_parallel_bets": operator_profile.max_parallel_bets,
            "max_parallel_initiatives": operator_profile.max_parallel_initiatives,
            "max_escalations": operator_profile.max_escalations,
            "max_abs_pricing_change": operator_profile.max_abs_pricing_change,
        }
    return json.dumps(prompt_payload, indent=2, sort_keys=True)
