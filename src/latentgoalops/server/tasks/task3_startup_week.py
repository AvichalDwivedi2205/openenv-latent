"""Task 3: startup week with explicit simulation time and delayed effects."""

from __future__ import annotations

import itertools
import random
from copy import deepcopy
from datetime import date, datetime, timedelta

from latentgoalops.models import (
    CustomerAccount,
    DashboardState,
    DecisionLedgerEntry,
    GovernanceConstraint,
    InboxItem,
    InitiativeItem,
    InternalTeamState,
    LatentGoalOpsAction,
    MarketContext,
    MessagingAction,
    SimCalendarEvent,
    StakeholderPersona,
    SupportPolicy,
    TaskId,
    TemporalEffectRecord,
)
from latentgoalops.server.config import load_config
from latentgoalops.server.hidden_goals import HiddenGoal, active_weights, compute_utility
from latentgoalops.server.tasks.template_bank import load_events, load_initiative_effects


def initial_dashboard() -> DashboardState:
    """Base startup operating state."""
    return DashboardState(
        dau=16000.0,
        mau=76000.0,
        d7_retention=0.41,
        d30_retention=0.24,
        mrr=38000.0,
        arpu=92.0,
        cac=67.0,
        churn_rate=0.048,
        ops_margin=0.34,
        infra_cost_per_unit=2.10,
        support_ticket_volume=142,
    )


def _parse_date(raw: str) -> date:
    return datetime.fromisoformat(raw).date()


def sim_date_for_step(start_date: str, step_index: int) -> str:
    """Return ISO simulation date for a given zero-based step."""
    return (_parse_date(start_date) + timedelta(days=step_index)).isoformat()


def sim_day_label(step_index: int) -> str:
    """Return a readable day label."""
    return f"Day {step_index + 1}"


def _task3_config() -> dict:
    return load_config("tasks.yaml")["task3"]


def _beneficiary_segments(kind: str) -> list[str]:
    if kind == "growth":
        return ["self_serve", "smb", "mid_market"]
    if kind == "retention":
        return ["mid_market", "enterprise", "strategic"]
    if kind == "revenue":
        return ["mid_market", "enterprise", "strategic"]
    return ["smb", "enterprise", "strategic"]


def _policy_tags(name: str, kind: str) -> list[str]:
    tags = []
    if "pricing" in name:
        tags.append("pricing_guardrail")
    if kind == "efficiency":
        tags.append("margin_guardrail")
    if "support" in name or "incident" in name:
        tags.append("sla_guardrail")
    return tags


def _load_initiatives(accounts: list[CustomerAccount], rng: random.Random, split: str) -> list[InitiativeItem]:
    templates = load_initiative_effects(split)
    items: list[InitiativeItem] = []
    for name, template in templates.items():
        kind = str(template["kind"])
        beneficiary_segments = _beneficiary_segments(kind)
        candidate_accounts = [account for account in accounts if account.segment in beneficiary_segments]
        linked_accounts = [
            account.account_id
            for account in rng.sample(candidate_accounts, k=min(len(candidate_accounts), rng.randint(1, 3)))
        ] if candidate_accounts else []
        items.append(
            InitiativeItem(
                item_id=name,
                title=name.replace("_", " ").title(),
                description=f"Operational initiative to {name.replace('_', ' ')}.",
                cost=float(template["cost"]),
                kind=kind,
                kpi_deltas={key: float(value) for key, value in template["deltas"].items()},
                uncertainty_band=0.10,
                stakeholder_tag=str(template.get("stakeholder_tag", "leadership")),
                lag_steps=int(template.get("lag_steps", 1)),
                effect_window=int(template.get("effect_window", 1)),
                delivery_note=str(template.get("delivery_note", "") or ""),
                beneficiary_segments=beneficiary_segments,
                beneficiary_account_ids=linked_accounts,
                implementation_risk=round(rng.uniform(0.08, 0.35), 3),
                policy_tags=_policy_tags(name, kind),
            )
        )
    _annotate_initiative_structure(items, {account.account_id: account for account in accounts})
    return items


def _initiative_prefix(item_id: str) -> str:
    return item_id


def _annotate_initiative_structure(
    items: list[InitiativeItem],
    accounts_by_id: dict[str, CustomerAccount],
) -> None:
    by_prefix = {_initiative_prefix(item.item_id): item for item in items}
    referral = by_prefix.get("launch_referral_loop")
    onboarding = by_prefix.get("improve_onboarding")
    pricing = by_prefix.get("ship_usage_pricing")
    analytics = by_prefix.get("launch_admin_analytics")
    infra = by_prefix.get("optimize_infra")
    incident = by_prefix.get("refactor_incident_tooling")
    triage = by_prefix.get("automate_support_triage")
    login_fix = by_prefix.get("fix_login_bug")

    if referral and onboarding:
        referral.requires_item_ids = [onboarding.item_id]
        referral.synergy_item_ids = [onboarding.item_id]
        onboarding.synergy_item_ids = [referral.item_id]
    if pricing and analytics:
        pricing.requires_item_ids = [analytics.item_id]
        pricing.synergy_item_ids = [analytics.item_id]
        analytics.synergy_item_ids = [pricing.item_id]
    if pricing and referral:
        pricing.conflicts_with_ids = [referral.item_id]
        referral.conflicts_with_ids = [pricing.item_id]
    if infra and incident:
        infra.synergy_item_ids = [incident.item_id]
        incident.synergy_item_ids = [infra.item_id]
    if login_fix and onboarding:
        login_fix.synergy_item_ids = sorted(set(login_fix.synergy_item_ids + [onboarding.item_id]))
    if triage:
        triage.risk_notes.append("Can lower queue cost quickly, but service quality risk rises if premium accounts are already tense.")

    for item in items:
        linked_accounts = [accounts_by_id[account_id] for account_id in item.beneficiary_account_ids if account_id in accounts_by_id]
        risk_notes = list(item.risk_notes)
        if item.requires_item_ids:
            risk_notes.append("Some upside is gated on a visible prerequisite landing first.")
        if item.conflicts_with_ids:
            risk_notes.append("Can create mixed operating signals if paired with a conflicting initiative.")
        if any(account.renewal_window_days <= 30 for account in linked_accounts):
            risk_notes.append("Several linked accounts are already close to renewal, so rollout quality matters.")
        if any(account.segment in {"enterprise", "strategic"} for account in linked_accounts):
            risk_notes.append("A meaningful part of the impact lands on high-touch enterprise accounts.")
        item.risk_notes = risk_notes


def _schedule_events(rng: random.Random, horizon: int, start_date: str, split: str) -> dict[int, list[dict]]:
    config = load_events(split)
    names = list(config.keys())
    rng.shuffle(names)
    max_count = min(3, len(names))
    count = rng.randint(2, max_count) if max_count >= 2 else max_count
    event_steps = sorted(rng.sample(range(1, max(horizon - 1, 2)), k=count))
    schedule: dict[int, list[dict]] = {}
    for idx, (step, name) in enumerate(zip(event_steps, names[:count])):
        raw = config[name]
        schedule.setdefault(step, []).append(
            {
                "event_id": f"event_{idx}_{name}",
                "name": name,
                "summary": str(raw.get("summary", name.replace("_", " ").title())),
                "clue_goal": str(raw["clue_goal"]),
                "alerts": list(raw.get("alerts", [])),
                "effects": {key: float(value) for key, value in raw.get("effects", {}).items()},
                "lag_steps": int(raw.get("lag_steps", 1)),
                "sim_date": sim_date_for_step(start_date, step),
            }
        )
    return schedule


def _calendar_events(episode: dict, current_step: int) -> list[SimCalendarEvent]:
    entries: list[SimCalendarEvent] = []
    for step in sorted(episode["events"]):
        if step < current_step or step > current_step + 2:
            continue
        for event in episode["events"][step]:
            entries.append(
                SimCalendarEvent(
                    event_id=event["event_id"],
                    name=event["name"],
                    sim_date=event["sim_date"],
                    status="today" if step == current_step else "upcoming",
                    summary=event["summary"],
                    alerts=list(event["alerts"]),
                )
            )
    return entries


def _select_lead(stakeholders: list[StakeholderPersona]) -> StakeholderPersona:
    ceo = [stakeholder for stakeholder in stakeholders if stakeholder.role == "CEO"]
    if ceo:
        return max(ceo, key=lambda stakeholder: (stakeholder.political_power, stakeholder.credibility))
    return max(stakeholders, key=lambda stakeholder: (stakeholder.political_power, stakeholder.credibility))


def _goal_signal_lines(goal_name: str, episode: dict) -> list[str]:
    accounts: list[CustomerAccount] = sorted(
        episode["accounts"],
        key=lambda account: (account.renewal_window_days, -account.relationship_health, -account.annual_contract_value),
    )
    market_context: MarketContext = episode["market_context"]
    dashboard: DashboardState = episode["dashboard"]
    hot_accounts = ", ".join(account.company_name for account in accounts[:2]) or "top accounts"
    if goal_name == "growth":
        return [
            f"Self-serve activation looks softer than expected while competition intensity sits near {market_context.competition_intensity:.2f}.",
            "At the same time, finance is watching margin discipline closely enough that reckless spend would be hard to defend.",
        ]
    if goal_name == "retention":
        return [
            f"Several high-value accounts, including {hot_accounts}, are inside tighter renewal windows and support strain is rising.",
            "At the same time, the board is still asking for a cleaner monetization story, so the team cannot overcorrect blindly.",
        ]
    if goal_name == "revenue":
        return [
            f"Board pressure is at {market_context.board_pressure_level:.2f} and monetization credibility is starting to dominate the staff narrative.",
            "At the same time, several customer relationships remain fragile enough that an aggressive move could backfire.",
        ]
    return [
        f"Ops margin is {dashboard.ops_margin:.2f} with infra cost per unit at {dashboard.infra_cost_per_unit:.2f}, and runway discipline is back in focus.",
        "At the same time, top-line softness means the team still needs a plan that does not stall customer momentum entirely.",
    ]


def _build_inbox(
    rng: random.Random,
    episode: dict,
    goal_name: str,
    step_index: int,
    sim_date: str,
    realized_effects: list[TemporalEffectRecord],
    calendar_events: list[SimCalendarEvent],
) -> list[InboxItem]:
    stakeholders: list[StakeholderPersona] = episode["stakeholders"]
    accounts: list[CustomerAccount] = sorted(
        episode["accounts"],
        key=lambda account: (
            account.renewal_window_days <= 30,
            account.churn_propensity,
            account.strategic_importance,
        ),
        reverse=True,
    )
    lead = _select_lead(stakeholders)
    hot_account = accounts[0]
    signal_lines = _goal_signal_lines(goal_name, episode)
    items = [
        InboxItem(
            item_id=f"msg_{step_index}_0",
            text=(
                f"{lead.name} ({lead.role}) wants a cleaner plan. "
                f"{signal_lines[0]} {signal_lines[1]}"
            ),
            sender=lead.role,
            metadata={"importance": "high", "timestamp": f"{sim_date}T09:00:00Z", "persona_id": lead.persona_id},
        )
    ]
    items.append(
        InboxItem(
            item_id=f"msg_{step_index}_1",
            text=(
                f"{hot_account.company_name} is a {hot_account.segment.replace('_', ' ')} account "
                f"worth about ${hot_account.annual_contract_value:,.0f} ARR with renewal in "
                f"{hot_account.renewal_window_days} days and relationship health {hot_account.relationship_health:.2f}."
            ),
            sender=hot_account.company_name,
            metadata={"importance": "high", "timestamp": f"{sim_date}T10:00:00Z", "account_id": hot_account.account_id},
        )
    )
    if realized_effects:
        effect = realized_effects[0]
        items.append(
            InboxItem(
                item_id=f"msg_{step_index}_2",
                text=f"Analytics: {effect.summary}",
                sender="analytics",
                metadata={"importance": "medium", "timestamp": f"{sim_date}T11:00:00Z"},
            )
        )
    elif calendar_events:
        event = calendar_events[0]
        items.append(
            InboxItem(
                item_id=f"msg_{step_index}_2",
                text=f"Chief of staff note: {event.summary}",
                sender="chief_of_staff",
                metadata={"importance": "medium", "timestamp": f"{sim_date}T11:00:00Z"},
            )
        )
    return items


def _memory_summary(episode: dict) -> str:
    recent_decisions = episode["decision_ledger"][-2:]
    recent_effects = episode["realized_effects"][-2:]
    decision_line = "No prior decisions logged yet."
    if recent_decisions:
        decision_parts = []
        for decision in recent_decisions:
            if decision.chosen_initiatives:
                decision_parts.append(
                    f"{decision.sim_date}: launched {', '.join(decision.chosen_initiatives)}"
                )
            elif decision.messaging_action:
                decision_parts.append(f"{decision.sim_date}: shifted messaging to {decision.messaging_action.value}")
        if decision_parts:
            decision_line = "Recent decisions: " + "; ".join(decision_parts) + "."
    effect_line = "No delayed effects landed today."
    if recent_effects:
        effect_line = "Recent realized effects: " + "; ".join(effect.summary for effect in recent_effects) + "."
    pending_count = len(episode["pending_effects"])
    pending_line = f"{pending_count} delayed effect(s) remain in flight."
    return " ".join([decision_line, effect_line, pending_line])


def _narrative(
    goal_name: str,
    step_index: int,
    sim_date: str,
    dashboard: DashboardState,
    accounts: list[CustomerAccount],
    alerts: list[str],
    realized_effects: list[TemporalEffectRecord],
    pending_effects: list[TemporalEffectRecord],
    market_context: MarketContext,
) -> str:
    alert_line = " ".join(alerts) if alerts else "No critical alerts have fired yet."
    effect_line = (
        " ".join(effect.summary for effect in realized_effects)
        if realized_effects
        else "No delayed rollouts landed overnight."
    )
    signal_lines = _goal_signal_lines(
        goal_name,
        {
            "accounts": accounts,
            "market_context": market_context,
            "dashboard": dashboard,
        },
    )
    return (
        f"{sim_day_label(step_index)} ({sim_date}) begins with DAU at {dashboard.dau:.0f}, MRR at {dashboard.mrr:.0f}, "
        f"and ops margin at {dashboard.ops_margin:.2f}. Runway is {market_context.cash_runway_months} months and board pressure "
        f"is {market_context.board_pressure_level:.2f}. Stakeholders are sending mixed signals. "
        f"{signal_lines[0]} {signal_lines[1]} {alert_line} {effect_line} "
        f"There are {len(pending_effects)} delayed effect(s) still in flight, so plan across dates rather than just today's KPI move."
    )


def build_task3_episode(
    rng: random.Random,
    hidden_goal: HiddenGoal,
    horizon: int,
    budget: float,
    capacity: float,
    world: dict,
    split: str = "core",
) -> dict:
    """Create a startup-week episode with scheduled events."""
    config = _task3_config()
    start_date = str(config.get("start_date", "2026-03-02"))
    accounts = [account.model_copy(deep=True) for account in world["accounts"]]
    stakeholders = [stakeholder.model_copy(deep=True) for stakeholder in world["stakeholders"]]
    teams = [team.model_copy(deep=True) for team in world["teams"]]
    market_context = world["market_context"].model_copy(deep=True)
    governance_constraints = [constraint.model_copy(deep=True) for constraint in world["governance_constraints"]]
    backlog = _load_initiatives(accounts, rng, split)
    rng.shuffle(backlog)
    backlog = backlog[: rng.randint(5, min(10, len(backlog)))]
    schedule = _schedule_events(rng, horizon, start_date, split)
    dashboard = initial_dashboard()
    return {
        "dashboard": dashboard,
        "backlog": backlog,
        "accounts": accounts,
        "stakeholders": stakeholders,
        "teams": teams,
        "market_context": market_context,
        "governance_constraints": governance_constraints,
        "budget_remaining": float(budget),
        "capacity_remaining": float(capacity),
        "completed_ids": set(),
        "step_index": 0,
        "horizon": horizon,
        "start_date": start_date,
        "events": schedule,
        "alerts": [],
        "action_history": [],
        "decision_ledger": [],
        "pending_effects": [],
        "realized_effects": [],
        "latent_utility_history": [compute_utility(dashboard.to_metric_vector(), hidden_goal)],
        "goal_history": [hidden_goal.archetype.value],
        "decision_quality_history": [],
        "invalid_actions": 0,
        "policy_violations": 0,
        "task_summary": (
            "Operate the startup across a dated operating calendar, track delayed effects from your decisions, "
            "manage high-value accounts and stakeholder agendas, infer the hidden objective, and adapt if it shifts."
        ),
    }


def current_goal_name(hidden_goal: HiddenGoal, step_index: int) -> str:
    """Return the active goal at this step."""
    if hidden_goal.shift_step is not None and hidden_goal.shift_goal is not None and step_index >= hidden_goal.shift_step:
        return hidden_goal.shift_goal.value
    return hidden_goal.archetype.value


def build_task3_view(rng: random.Random, hidden_goal: HiddenGoal, episode: dict) -> dict:
    """Create the current observable bundle."""
    goal_name = current_goal_name(hidden_goal, episode["step_index"])
    alerts = list(episode["alerts"])
    sim_date = sim_date_for_step(episode["start_date"], episode["step_index"])
    calendar_events = _calendar_events(episode, episode["step_index"])
    if episode["step_index"] in episode["events"]:
        alerts.extend(event_alert for event in episode["events"][episode["step_index"]] for event_alert in event["alerts"])
    realized_effects = [effect.model_copy(deep=True) for effect in episode["realized_effects"]]
    pending_effects = [
        effect.model_copy(deep=True)
        for effect in sorted(episode["pending_effects"], key=lambda effect: (effect.scheduled_for_step, effect.effect_id))
    ]
    return {
        "step_index": episode["step_index"],
        "horizon": episode["horizon"],
        "sim_date": sim_date,
        "sim_day_label": sim_day_label(episode["step_index"]),
        "dashboard": deepcopy(episode["dashboard"]),
        "backlog": [item for item in episode["backlog"] if item.item_id not in episode["completed_ids"]],
        "accounts": [
            account.model_copy(deep=True)
            for account in sorted(
                episode["accounts"],
                key=lambda account: (
                    account.renewal_window_days <= 30,
                    account.strategic_importance,
                    account.annual_contract_value,
                ),
                reverse=True,
            )[:6]
        ],
        "stakeholders": [stakeholder.model_copy(deep=True) for stakeholder in episode["stakeholders"]],
        "teams": [team.model_copy(deep=True) for team in episode["teams"]],
        "market_context": episode["market_context"].model_copy(deep=True),
        "governance_constraints": [constraint.model_copy(deep=True) for constraint in episode["governance_constraints"]],
        "alerts": alerts,
        "calendar_events": calendar_events,
        "inbox": _build_inbox(rng, episode, goal_name, episode["step_index"], sim_date, realized_effects, calendar_events),
        "narrative": _narrative(
            goal_name,
            episode["step_index"],
            sim_date,
            episode["dashboard"],
            episode["accounts"],
            alerts,
            realized_effects,
            pending_effects,
            episode["market_context"],
        ),
        "budget_remaining": episode["budget_remaining"],
        "capacity_remaining": episode["capacity_remaining"],
        "task_summary": episode["task_summary"],
        "decision_ledger": [entry.model_copy(deep=True) for entry in episode["decision_ledger"]],
        "pending_effects": pending_effects,
        "realized_effects": realized_effects,
        "memory_summary": _memory_summary(episode),
    }


def _apply_channel_deltas(dashboard: DashboardState, channel_deltas: dict[str, float]) -> DashboardState:
    updated = dashboard.model_copy(deep=True)
    updated.dau += channel_deltas.get("growth", 0.0) * 2400.0
    updated.mau += channel_deltas.get("growth", 0.0) * 6200.0
    updated.d7_retention = max(0.0, min(1.0, updated.d7_retention + channel_deltas.get("retention", 0.0) * 0.15))
    updated.d30_retention = max(0.0, min(1.0, updated.d30_retention + channel_deltas.get("retention", 0.0) * 0.12))
    updated.mrr += channel_deltas.get("revenue", 0.0) * 18000.0
    updated.arpu += channel_deltas.get("revenue", 0.0) * 18.0
    updated.churn_rate = max(
        0.0,
        updated.churn_rate - channel_deltas.get("retention", 0.0) * 0.06 + max(0.0, -channel_deltas.get("revenue", 0.0)) * 0.02,
    )
    updated.ops_margin = max(0.0, min(1.0, updated.ops_margin + channel_deltas.get("efficiency", 0.0) * 0.20))
    updated.infra_cost_per_unit = max(0.1, updated.infra_cost_per_unit - channel_deltas.get("efficiency", 0.0) * 1.1)
    updated.support_ticket_volume = max(
        0,
        int(updated.support_ticket_volume - channel_deltas.get("efficiency", 0.0) * 60 - channel_deltas.get("retention", 0.0) * 25 + max(0.0, -channel_deltas.get("growth", 0.0)) * 20),
    )
    return updated


def _apply_dashboard_deltas(dashboard: DashboardState, dashboard_deltas: dict[str, float]) -> DashboardState:
    updated = dashboard.model_copy(deep=True)
    for key, value in dashboard_deltas.items():
        if key == "support_ticket_volume":
            updated.support_ticket_volume = max(0, int(updated.support_ticket_volume + value))
        elif hasattr(updated, key):
            setattr(updated, key, getattr(updated, key) + value)
    updated.d7_retention = max(0.0, min(1.0, updated.d7_retention))
    updated.d30_retention = max(0.0, min(1.0, updated.d30_retention))
    updated.churn_rate = max(0.0, updated.churn_rate)
    updated.ops_margin = max(0.0, min(1.0, updated.ops_margin))
    updated.infra_cost_per_unit = max(0.1, updated.infra_cost_per_unit)
    return updated


def _effect_record(
    *,
    effect_id: str,
    decision_id: str | None,
    source_type: str,
    source_id: str,
    summary: str,
    scheduled_for_step: int,
    start_date: str,
    channel_deltas: dict[str, float] | None = None,
    dashboard_deltas: dict[str, float] | None = None,
    affected_account_ids: list[str] | None = None,
    affected_team_ids: list[str] | None = None,
) -> TemporalEffectRecord:
    return TemporalEffectRecord(
        effect_id=effect_id,
        decision_id=decision_id,
        source_type=source_type,
        source_id=source_id,
        summary=summary,
        channel_deltas=channel_deltas or {},
        dashboard_deltas=dashboard_deltas or {},
        affected_account_ids=affected_account_ids or [],
        affected_team_ids=affected_team_ids or [],
        scheduled_for_step=scheduled_for_step,
        scheduled_for_date=sim_date_for_step(start_date, scheduled_for_step),
    )


def _message_effect(action) -> dict[str, float]:
    mapping = {
        MessagingAction.GROWTH_PUSH: {"growth": 0.03},
        MessagingAction.RETENTION_CAMPAIGN: {"retention": 0.03},
        MessagingAction.REVENUE_UPSELL: {"revenue": 0.04},
        MessagingAction.COST_COMMS: {"efficiency": 0.03},
    }
    return mapping.get(action.messaging_action, {})


def _support_immediate_effect(action) -> dict[str, float]:
    if action.support_policy == SupportPolicy.PREMIUM_SLA:
        return {"retention": 0.02, "efficiency": -0.01}
    if action.support_policy == SupportPolicy.AUTOMATION_FIRST:
        return {"efficiency": 0.02, "retention": -0.005}
    if action.support_policy == SupportPolicy.INCIDENT_SWARM:
        return {"retention": 0.01, "efficiency": 0.01}
    return {"retention": 0.01, "efficiency": 0.01}


def _schedule_initiative_effects(
    chosen_items: list[InitiativeItem],
    current_step: int,
    start_date: str,
    decision_id: str,
) -> tuple[dict[str, float], list[TemporalEffectRecord]]:
    immediate = {"growth": 0.0, "retention": 0.0, "revenue": 0.0, "efficiency": 0.0}
    scheduled: list[TemporalEffectRecord] = []
    for item in chosen_items:
        for channel, value in item.kpi_deltas.items():
            immediate[channel] += float(value) * 0.25
        delayed_fraction = 0.75
        window = max(1, item.effect_window)
        for offset in range(window):
            scheduled_step = current_step + item.lag_steps + offset
            if scheduled_step > _task3_config()["horizon"]:
                continue
            scheduled.append(
                _effect_record(
                    effect_id=f"{decision_id}_{item.item_id}_{offset}",
                    decision_id=decision_id,
                    source_type="initiative",
                    source_id=item.item_id,
                    summary=(
                        item.delivery_note
                        or f"{item.title} shipped and is now moving operating metrics."
                    ),
                    scheduled_for_step=scheduled_step,
                    start_date=start_date,
                    channel_deltas={
                        channel: float(value) * delayed_fraction / window
                        for channel, value in item.kpi_deltas.items()
                    },
                    affected_account_ids=item.beneficiary_account_ids,
                    affected_team_ids=[
                        "team_growth" if item.kind == "growth" else (
                            "team_support" if item.kind == "retention" else (
                                "team_product" if item.kind == "revenue" else "team_infra"
                            )
                        )
                    ],
                )
            )
    return immediate, scheduled


def _schedule_support_effect(action, current_step: int, start_date: str, decision_id: str) -> list[TemporalEffectRecord]:
    if action.support_policy == SupportPolicy.PREMIUM_SLA:
        delayed = {"retention": 0.015}
        summary = "Premium SLA follow-through stabilized a portion of at-risk accounts."
    elif action.support_policy == SupportPolicy.AUTOMATION_FIRST:
        delayed = {"efficiency": 0.02}
        summary = "Automation-first support policy reduced manual queue load."
    elif action.support_policy == SupportPolicy.INCIDENT_SWARM:
        delayed = {"retention": 0.01, "efficiency": 0.01}
        summary = "Incident swarm created cross-functional momentum after the firefight."
    else:
        delayed = {"retention": 0.005, "efficiency": 0.005}
        summary = "Balanced support policy produced a modest follow-through improvement."
    return [
        _effect_record(
            effect_id=f"{decision_id}_support",
            decision_id=decision_id,
            source_type="support_policy",
            source_id=action.support_policy.value if action.support_policy else "none",
            summary=summary,
            scheduled_for_step=current_step + 1,
            start_date=start_date,
            channel_deltas=delayed,
            affected_team_ids=["team_support"],
        )
    ]


def _schedule_pricing_effect(action, current_step: int, start_date: str, decision_id: str) -> tuple[dict[str, float], list[TemporalEffectRecord]]:
    immediate = {"growth": 0.0, "retention": 0.0, "revenue": 0.0, "efficiency": 0.0}
    if action.pricing_change_pct is None or abs(action.pricing_change_pct) < 1e-9:
        return immediate, []
    if action.pricing_change_pct > 0:
        immediate["revenue"] += action.pricing_change_pct * 0.18
        summary = "A delayed churn response arrived after the price increase."
        delayed = {"retention": -action.pricing_change_pct * 0.10}
    else:
        immediate["growth"] += abs(action.pricing_change_pct) * 0.14
        summary = "A delayed monetization headwind arrived after the discounting move."
        delayed = {"revenue": action.pricing_change_pct * 0.08}
    return immediate, [
        _effect_record(
            effect_id=f"{decision_id}_pricing",
            decision_id=decision_id,
            source_type="pricing",
            source_id="pricing_change",
            summary=summary,
            scheduled_for_step=current_step + 2,
            start_date=start_date,
            channel_deltas=delayed,
        )
    ]


def _schedule_event_effects(step_events: list[dict], current_step: int, start_date: str) -> list[TemporalEffectRecord]:
    records: list[TemporalEffectRecord] = []
    for event in step_events:
        records.append(
            _effect_record(
                effect_id=f"{event['event_id']}_effect",
                decision_id=None,
                source_type="event",
                source_id=event["event_id"],
                summary=event["summary"],
                scheduled_for_step=current_step + int(event.get("lag_steps", 1)),
                start_date=start_date,
                dashboard_deltas=event.get("effects", {}),
            )
        )
    return records


def _aggregate_effect_channels(effects: list[TemporalEffectRecord]) -> dict[str, float]:
    aggregate = {"growth": 0.0, "retention": 0.0, "revenue": 0.0, "efficiency": 0.0}
    for effect in effects:
        for channel, value in effect.channel_deltas.items():
            aggregate[channel] += float(value)
    return aggregate


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _beneficiary_account_ids(
    episode: dict,
    chosen_items: list[InitiativeItem],
    realized_effects: list[TemporalEffectRecord],
) -> set[str]:
    account_ids = {account_id for item in chosen_items for account_id in item.beneficiary_account_ids}
    for effect in realized_effects:
        account_ids.update(effect.affected_account_ids)
    return account_ids


def _governance_flags(episode: dict, action) -> list[GovernanceConstraint]:
    by_id = {constraint.constraint_id: constraint for constraint in episode["governance_constraints"]}
    flags: list[GovernanceConstraint] = []
    strategic_renewals = [
        account
        for account in episode["accounts"]
        if account.segment in {"enterprise", "strategic"} and account.renewal_window_days <= 30
    ]
    if (
        action.pricing_change_pct is not None
        and action.pricing_change_pct > 0.05
        and strategic_renewals
        and "pricing_guardrail" in by_id
    ):
        flags.append(by_id["pricing_guardrail"])
    if (
        action.support_policy == SupportPolicy.AUTOMATION_FIRST
        and any(account.support_tier == "premium" or account.segment == "strategic" for account in strategic_renewals)
        and "sla_guardrail" in by_id
    ):
        flags.append(by_id["sla_guardrail"])
    if (
        action.messaging_action == MessagingAction.GROWTH_PUSH
        and "margin_guardrail" in by_id
        and episode["dashboard"].ops_margin < float(by_id["margin_guardrail"].threshold or 0.0)
    ):
        flags.append(by_id["margin_guardrail"])
    return flags


def _schedule_governance_effects(
    flags: list[GovernanceConstraint],
    episode: dict,
    current_step: int,
    start_date: str,
    decision_id: str,
) -> list[TemporalEffectRecord]:
    risk_accounts = [
        account.account_id
        for account in episode["accounts"]
        if account.segment in {"enterprise", "strategic"} or account.renewal_window_days <= 30
    ][:4]
    records: list[TemporalEffectRecord] = []
    for flag in flags:
        if flag.constraint_id == "pricing_guardrail":
            records.append(
                _effect_record(
                    effect_id=f"{decision_id}_gov_pricing",
                    decision_id=decision_id,
                    source_type="governance",
                    source_id=flag.constraint_id,
                    summary="Strategic accounts reacted poorly to a pricing move near renewal windows.",
                    scheduled_for_step=current_step + 1,
                    start_date=start_date,
                    channel_deltas={"retention": -0.05, "revenue": -0.02},
                    affected_account_ids=risk_accounts,
                )
            )
        elif flag.constraint_id == "sla_guardrail":
            records.append(
                _effect_record(
                    effect_id=f"{decision_id}_gov_sla",
                    decision_id=decision_id,
                    source_type="governance",
                    source_id=flag.constraint_id,
                    summary="Automation pressure spilled into premium support queues and trust slipped.",
                    scheduled_for_step=current_step + 1,
                    start_date=start_date,
                    channel_deltas={"retention": -0.04, "efficiency": -0.01},
                    affected_account_ids=risk_accounts,
                    affected_team_ids=["team_support"],
                )
            )
        elif flag.constraint_id == "margin_guardrail":
            records.append(
                _effect_record(
                    effect_id=f"{decision_id}_gov_margin",
                    decision_id=decision_id,
                    source_type="governance",
                    source_id=flag.constraint_id,
                    summary="Board pressure increased because growth spend ran ahead of margin reality.",
                    scheduled_for_step=current_step + 1,
                    start_date=start_date,
                    channel_deltas={"growth": -0.02, "efficiency": -0.04},
                    affected_team_ids=["team_growth", "team_infra"],
                )
            )
    return records


def _support_total_effect_channels(action) -> dict[str, float]:
    total = {"growth": 0.0, "retention": 0.0, "revenue": 0.0, "efficiency": 0.0}
    for channel, value in _support_immediate_effect(action).items():
        total[channel] += float(value)
    delayed_map = {
        SupportPolicy.PREMIUM_SLA: {"retention": 0.015},
        SupportPolicy.AUTOMATION_FIRST: {"efficiency": 0.02},
        SupportPolicy.INCIDENT_SWARM: {"retention": 0.01, "efficiency": 0.01},
        SupportPolicy.BALANCED_TRIAGE: {"retention": 0.005, "efficiency": 0.005},
        None: {"retention": 0.005, "efficiency": 0.005},
    }
    for channel, value in delayed_map.get(action.support_policy, {}).items():
        total[channel] += float(value)
    return total


def _pricing_total_effect_channels(action) -> dict[str, float]:
    total = {"growth": 0.0, "retention": 0.0, "revenue": 0.0, "efficiency": 0.0}
    if action.pricing_change_pct is None or abs(action.pricing_change_pct) < 1e-9:
        return total
    if action.pricing_change_pct > 0:
        total["revenue"] += action.pricing_change_pct * 0.18
        total["retention"] += -action.pricing_change_pct * 0.10
    else:
        total["growth"] += abs(action.pricing_change_pct) * 0.14
        total["revenue"] += action.pricing_change_pct * 0.08
    return total


def _initiative_bundle_value(chosen_items: list[InitiativeItem], weights: dict[str, float]) -> float:
    chosen_ids = {item.item_id for item in chosen_items}
    score = 0.0
    for item in chosen_items:
        base = sum(float(item.kpi_deltas.get(channel, 0.0)) * float(weights.get(channel, 0.0)) for channel in weights)
        if item.requires_item_ids and not any(required in chosen_ids for required in item.requires_item_ids):
            base *= 0.55
        if item.synergy_item_ids and any(synergy in chosen_ids for synergy in item.synergy_item_ids):
            base *= 1.12
        score += base
    for item in chosen_items:
        for conflict_id in item.conflicts_with_ids:
            if conflict_id in chosen_ids and item.item_id < conflict_id:
                score -= 0.18 * sum(
                    float(item.kpi_deltas.get(channel, 0.0)) * float(weights.get(channel, 0.0))
                    for channel in weights
                )
    return max(0.0, score)


def evaluate_task3_action_value(episode: dict, action: LatentGoalOpsAction, weights: dict[str, float]) -> float:
    """Approximate the strategic value of a task-3 action under the active goal."""
    visible_backlog = {item.item_id: item for item in episode["backlog"] if item.item_id not in episode["completed_ids"]}
    chosen_items = [visible_backlog[item_id] for item_id in action.chosen_initiatives if item_id in visible_backlog]
    spent_budget = sum(item.cost for item in chosen_items)
    spent_capacity = sum(max(1.0, item.cost / 2.0) for item in chosen_items)
    if spent_budget > episode["budget_remaining"] + 1e-9 or spent_capacity > episode["capacity_remaining"] + 1e-9:
        return 0.0

    score = _initiative_bundle_value(chosen_items, weights)
    score += sum(float(_message_effect(action).get(channel, 0.0)) * float(weights.get(channel, 0.0)) for channel in weights)
    score += sum(
        float(_support_total_effect_channels(action).get(channel, 0.0)) * float(weights.get(channel, 0.0))
        for channel in weights
    )
    score += sum(
        float(_pricing_total_effect_channels(action).get(channel, 0.0)) * float(weights.get(channel, 0.0))
        for channel in weights
    )

    governance_flags = _governance_flags(episode, action)
    governance_penalty = 0.03 * len(governance_flags)
    execution_penalty = sum(item.implementation_risk * 0.04 for item in chosen_items)
    concentration_bonus = 0.02 if chosen_items and len({item.kind for item in chosen_items}) == 1 else 0.0
    idle_penalty = 0.01 if not chosen_items and action.messaging_action == MessagingAction.NONE else 0.0
    return max(0.0, score + concentration_bonus - governance_penalty - execution_penalty - idle_penalty)


def solve_task3_oracle_action(episode: dict, weights: dict[str, float]) -> tuple[LatentGoalOpsAction, float]:
    """Brute-force a high-quality visible action under the active goal."""
    visible = [item for item in episode["backlog"] if item.item_id not in episode["completed_ids"]]
    best_action = LatentGoalOpsAction(
        task_id=TaskId.TASK3,
        chosen_initiatives=[],
        messaging_action=MessagingAction.NONE,
        pricing_change_pct=0.0,
        support_policy=SupportPolicy.BALANCED_TRIAGE,
        rationale="Oracle action using active-goal rollout value.",
    )
    best_value = evaluate_task3_action_value(episode, best_action, weights)
    pricing_choices = [-0.05, 0.0, 0.05]

    for subset_size in range(len(visible) + 1):
        for subset in itertools.combinations(visible, subset_size):
            subset_ids = [item.item_id for item in subset]
            subset_budget = sum(item.cost for item in subset)
            subset_capacity = sum(max(1.0, item.cost / 2.0) for item in subset)
            if subset_budget > episode["budget_remaining"] + 1e-9:
                continue
            if subset_capacity > episode["capacity_remaining"] + 1e-9:
                continue
            for messaging_action in MessagingAction:
                for support_policy in SupportPolicy:
                    for pricing_change_pct in pricing_choices:
                        candidate = LatentGoalOpsAction(
                            task_id=TaskId.TASK3,
                            chosen_initiatives=subset_ids,
                            messaging_action=messaging_action,
                            pricing_change_pct=pricing_change_pct,
                            support_policy=support_policy,
                            rationale="Oracle action using active-goal rollout value.",
                        )
                        candidate_value = evaluate_task3_action_value(episode, candidate, weights)
                        if candidate_value > best_value:
                            best_value = candidate_value
                            best_action = candidate
    return best_action, best_value


def _update_accounts(
    episode: dict,
    chosen_items: list[InitiativeItem],
    action,
    immediate_channels: dict[str, float],
    delayed_channels: dict[str, float],
    governance_flags: list[GovernanceConstraint],
    realized_effects: list[TemporalEffectRecord],
) -> list[TemporalEffectRecord]:
    targeted_ids = _beneficiary_account_ids(episode, chosen_items, realized_effects)
    renewal_records: list[TemporalEffectRecord] = []
    for account in episode["accounts"]:
        multiplier = 1.0 if account.account_id in targeted_ids else 0.35
        segment_weight = 1.0
        if account.segment in {"self_serve", "smb"}:
            segment_weight = 1.0 + max(0.0, immediate_channels["growth"] + delayed_channels["growth"]) * 1.2
        elif account.segment in {"enterprise", "strategic"}:
            segment_weight = 1.0 + max(0.0, immediate_channels["retention"] + delayed_channels["retention"]) * 1.1
        account.relationship_health = _clamp(
            account.relationship_health
            + (immediate_channels["retention"] * 0.10 + delayed_channels["retention"] * 0.08) * multiplier * segment_weight
            + (immediate_channels["growth"] * 0.03) * multiplier
            - max(0.0, immediate_channels["efficiency"]) * (0.025 if account.support_tier == "premium" else -0.01) * multiplier
        )
        account.churn_propensity = _clamp(
            account.churn_propensity
            - (immediate_channels["retention"] * 0.10 + delayed_channels["retention"] * 0.08) * multiplier
            + max(0.0, -delayed_channels["retention"]) * 0.06 * multiplier
        )
        account.expansion_potential = _clamp(
            account.expansion_potential
            + (immediate_channels["revenue"] * 0.07 + delayed_channels["revenue"] * 0.05) * multiplier
            + max(0.0, immediate_channels["growth"]) * 0.03
        )
        if action.pricing_change_pct and action.pricing_change_pct > 0 and account.renewal_window_days <= 30:
            account.relationship_health = _clamp(account.relationship_health - action.pricing_change_pct * 0.45 * multiplier)
            account.churn_propensity = _clamp(account.churn_propensity + action.pricing_change_pct * 0.30 * multiplier)
        if action.support_policy == SupportPolicy.PREMIUM_SLA and account.support_tier == "premium":
            account.relationship_health = _clamp(account.relationship_health + 0.03 * multiplier)
            account.churn_propensity = _clamp(account.churn_propensity - 0.025 * multiplier)
        if action.support_policy == SupportPolicy.AUTOMATION_FIRST and account.support_tier == "premium":
            account.relationship_health = _clamp(account.relationship_health - 0.04 * multiplier)
            account.churn_propensity = _clamp(account.churn_propensity + 0.03 * multiplier)
        if governance_flags and account.account_id in targeted_ids:
            account.relationship_health = _clamp(account.relationship_health - 0.03)
            account.churn_propensity = _clamp(account.churn_propensity + 0.03)

    next_step = episode["step_index"] + 1
    for account in episode["accounts"]:
        previous_window = account.renewal_window_days
        account.renewal_window_days = max(0, account.renewal_window_days - 1)
        if previous_window > 0 and account.renewal_window_days == 0:
            renewal_score = (
                account.relationship_health * 0.45
                + account.champion_strength * 0.20
                + (1.0 - account.churn_propensity) * 0.20
                + account.strategic_importance * 0.15
            )
            if renewal_score >= 0.62:
                renewal_delta = round(account.annual_contract_value / 12.0 * (0.04 + 0.05 * account.expansion_potential), 2)
                renewal_records.append(
                    TemporalEffectRecord(
                        effect_id=f"renewal_{account.account_id}_{next_step}",
                        source_type="renewal",
                        source_id=account.account_id,
                        summary=f"{account.company_name} renewed and modestly expanded after a stable operating stretch.",
                        dashboard_deltas={"mrr": renewal_delta, "churn_rate": -0.002},
                        affected_account_ids=[account.account_id],
                        scheduled_for_step=next_step,
                        scheduled_for_date=sim_date_for_step(episode["start_date"], next_step),
                        realized_step=next_step,
                        realized_date=sim_date_for_step(episode["start_date"], next_step),
                    )
                )
                account.relationship_health = _clamp(account.relationship_health + 0.04)
                account.adoption_stage = "steady"
            else:
                churn_delta = round(account.annual_contract_value / 12.0 * 0.08, 2)
                renewal_records.append(
                    TemporalEffectRecord(
                        effect_id=f"renewal_{account.account_id}_{next_step}",
                        source_type="renewal",
                        source_id=account.account_id,
                        summary=f"{account.company_name} entered a renewal scare and near-term revenue slipped.",
                        dashboard_deltas={"mrr": -churn_delta, "churn_rate": 0.004, "support_ticket_volume": 8.0},
                        affected_account_ids=[account.account_id],
                        scheduled_for_step=next_step,
                        scheduled_for_date=sim_date_for_step(episode["start_date"], next_step),
                        realized_step=next_step,
                        realized_date=sim_date_for_step(episode["start_date"], next_step),
                    )
                )
                account.relationship_health = _clamp(account.relationship_health - 0.05)
                account.adoption_stage = "at_risk"
    episode["accounts"].sort(
        key=lambda account: (
            account.renewal_window_days <= 30,
            account.strategic_importance,
            account.annual_contract_value,
        ),
        reverse=True,
    )
    return renewal_records


def _update_teams(
    episode: dict,
    chosen_items: list[InitiativeItem],
    action,
    invalid: bool,
    governance_flags: list[GovernanceConstraint],
    delayed_channels: dict[str, float],
) -> None:
    team_by_id = {team.team_id: team for team in episode["teams"]}
    if action.messaging_action == MessagingAction.GROWTH_PUSH and "team_growth" in team_by_id:
        team_by_id["team_growth"].burnout_risk = _clamp(team_by_id["team_growth"].burnout_risk + 0.04)
    if action.support_policy == SupportPolicy.PREMIUM_SLA and "team_support" in team_by_id:
        team_by_id["team_support"].burnout_risk = _clamp(team_by_id["team_support"].burnout_risk + 0.05)
        team_by_id["team_support"].capacity = _clamp(team_by_id["team_support"].capacity - 0.03)
    if action.support_policy == SupportPolicy.AUTOMATION_FIRST and "team_support" in team_by_id:
        team_by_id["team_support"].capacity = _clamp(team_by_id["team_support"].capacity + 0.04)
        team_by_id["team_support"].burnout_risk = _clamp(team_by_id["team_support"].burnout_risk - 0.03)
    for item in chosen_items:
        target_team = {
            "growth": "team_growth",
            "retention": "team_support",
            "revenue": "team_product",
            "efficiency": "team_infra",
        }.get(item.kind)
        if target_team and target_team in team_by_id:
            team = team_by_id[target_team]
            team.capacity = _clamp(team.capacity - 0.02)
            team.execution_reliability = _clamp(team.execution_reliability + 0.01 * max(0.0, 1.0 - item.implementation_risk))
    if invalid:
        for team in team_by_id.values():
            team.cross_team_friction = _clamp(team.cross_team_friction + 0.03)
    if governance_flags:
        for flag in governance_flags:
            if flag.constraint_id == "margin_guardrail" and "team_growth" in team_by_id:
                team_by_id["team_growth"].cross_team_friction = _clamp(team_by_id["team_growth"].cross_team_friction + 0.04)
            if flag.constraint_id == "sla_guardrail" and "team_support" in team_by_id:
                team_by_id["team_support"].execution_reliability = _clamp(team_by_id["team_support"].execution_reliability - 0.03)
    if delayed_channels["efficiency"] > 0 and "team_infra" in team_by_id:
        team_by_id["team_infra"].capacity = _clamp(team_by_id["team_infra"].capacity + 0.02)
        team_by_id["team_infra"].burnout_risk = _clamp(team_by_id["team_infra"].burnout_risk - 0.02)


def _update_market_context(
    episode: dict,
    immediate_channels: dict[str, float],
    delayed_channels: dict[str, float],
    governance_flags: list[GovernanceConstraint],
) -> None:
    market_context: MarketContext = episode["market_context"]
    board_delta = 0.0
    board_delta += -0.05 * max(0.0, immediate_channels["revenue"] + delayed_channels["revenue"])
    board_delta += -0.04 * max(0.0, immediate_channels["efficiency"] + delayed_channels["efficiency"])
    board_delta += 0.05 * len(governance_flags)
    if episode["dashboard"].ops_margin < market_context.gross_margin_target:
        board_delta += 0.03
    market_context.board_pressure_level = _clamp(market_context.board_pressure_level + board_delta)
    pipeline_delta = 0.06 * max(0.0, immediate_channels["growth"] + delayed_channels["growth"])
    pipeline_delta += 0.04 * max(0.0, immediate_channels["revenue"] + delayed_channels["revenue"])
    pipeline_delta -= 0.03 * len(governance_flags)
    market_context.sales_pipeline_health = _clamp(market_context.sales_pipeline_health + pipeline_delta)
    if immediate_channels["efficiency"] + delayed_channels["efficiency"] > 0.08:
        market_context.cash_runway_months = min(36, market_context.cash_runway_months + 1)
    elif len(governance_flags) >= 2 or episode["dashboard"].ops_margin < max(0.20, market_context.gross_margin_target - 0.12):
        market_context.cash_runway_months = max(3, market_context.cash_runway_months - 1)


def apply_task3_action(rng: random.Random, hidden_goal: HiddenGoal, episode: dict, action) -> dict:
    """Mutate the task 3 episode with the selected action."""
    current_step = episode["step_index"]
    start_date = episode["start_date"]
    enable_delayed_effects = bool(episode.get("enable_delayed_effects", True))
    decision_id = f"decision_{current_step}"
    visible_backlog = {item.item_id: item for item in episode["backlog"] if item.item_id not in episode["completed_ids"]}
    chosen_items = [visible_backlog[item_id] for item_id in action.chosen_initiatives if item_id in visible_backlog]
    spent_budget = sum(item.cost for item in chosen_items)
    spent_capacity = sum(max(1.0, item.cost / 2.0) for item in chosen_items)
    invalid = spent_budget > episode["budget_remaining"] + 1e-9 or spent_capacity > episode["capacity_remaining"] + 1e-9

    step_events = episode["events"].get(current_step, [])
    previous_dashboard = episode["dashboard"].model_copy(deep=True)
    scheduled_effects: list[TemporalEffectRecord] = []
    immediate_channels = {"growth": 0.0, "retention": 0.0, "revenue": 0.0, "efficiency": 0.0}
    governance_flags: list[GovernanceConstraint] = []
    policy_alerts: list[str] = []

    if not invalid:
        episode["budget_remaining"] -= spent_budget
        episode["capacity_remaining"] -= spent_capacity
        for item in chosen_items:
            episode["completed_ids"].add(item.item_id)

        initiative_immediate, initiative_scheduled = _schedule_initiative_effects(
            chosen_items,
            current_step,
            start_date,
            decision_id,
        )
        for channel, value in initiative_immediate.items():
            immediate_channels[channel] += value
        scheduled_effects.extend(initiative_scheduled)

        for effect in (_message_effect(action), _support_immediate_effect(action)):
            for channel, value in effect.items():
                immediate_channels[channel] += float(value)

        pricing_immediate, pricing_scheduled = _schedule_pricing_effect(action, current_step, start_date, decision_id)
        for channel, value in pricing_immediate.items():
            immediate_channels[channel] += value
        scheduled_effects.extend(pricing_scheduled)
        scheduled_effects.extend(_schedule_support_effect(action, current_step, start_date, decision_id))

        governance_flags = _governance_flags(episode, action)
        if governance_flags:
            episode["policy_violations"] += len(governance_flags)
            policy_alerts = [f"Governance risk: {constraint.title}." for constraint in governance_flags]
            scheduled_effects.extend(
                _schedule_governance_effects(governance_flags, episode, current_step, start_date, decision_id)
            )

        if not enable_delayed_effects:
            immediate_from_scheduled = _aggregate_effect_channels(scheduled_effects)
            for channel, value in immediate_from_scheduled.items():
                immediate_channels[channel] += value
            scheduled_effects = []

        for channel in immediate_channels:
            immediate_channels[channel] += rng.uniform(-0.01, 0.01)
        episode["dashboard"] = _apply_channel_deltas(episode["dashboard"], immediate_channels)
    else:
        episode["invalid_actions"] += 1

    scheduled_effects.extend(_schedule_event_effects(step_events, current_step, start_date))
    if not enable_delayed_effects:
        event_effects = scheduled_effects[:]
        scheduled_effects = []
        for effect in event_effects:
            if effect.dashboard_deltas:
                episode["dashboard"] = _apply_dashboard_deltas(episode["dashboard"], effect.dashboard_deltas)
    episode["pending_effects"].extend(scheduled_effects)
    episode["pending_effects"].sort(key=lambda effect: (effect.scheduled_for_step, effect.effect_id))

    next_step = current_step + 1
    realized_effects: list[TemporalEffectRecord] = []
    remaining_pending: list[TemporalEffectRecord] = []
    for effect in episode["pending_effects"]:
        if effect.scheduled_for_step == next_step:
            realized = effect.model_copy(
                update={
                    "realized_step": next_step,
                    "realized_date": sim_date_for_step(start_date, next_step),
                },
                deep=True,
            )
            realized_effects.append(realized)
        else:
            remaining_pending.append(effect)
    episode["pending_effects"] = remaining_pending

    delayed_channels = _aggregate_effect_channels(realized_effects)
    for channel in delayed_channels:
        delayed_channels[channel] += rng.uniform(-0.005, 0.005)
    if any(abs(value) > 1e-9 for value in delayed_channels.values()):
        episode["dashboard"] = _apply_channel_deltas(episode["dashboard"], delayed_channels)
    for effect in realized_effects:
        if effect.dashboard_deltas:
            episode["dashboard"] = _apply_dashboard_deltas(episode["dashboard"], effect.dashboard_deltas)

    renewal_effects = _update_accounts(
        episode,
        chosen_items,
        action,
        immediate_channels,
        delayed_channels,
        governance_flags,
        realized_effects,
    )
    for effect in renewal_effects:
        if effect.dashboard_deltas:
            episode["dashboard"] = _apply_dashboard_deltas(episode["dashboard"], effect.dashboard_deltas)
    realized_effects.extend(renewal_effects)
    _update_teams(episode, chosen_items, action, invalid, governance_flags, delayed_channels)
    _update_market_context(episode, immediate_channels, delayed_channels, governance_flags)

    decision = DecisionLedgerEntry(
        decision_id=decision_id,
        step_index=current_step,
        sim_date=sim_date_for_step(start_date, current_step),
        chosen_initiatives=[item.item_id for item in chosen_items],
        messaging_action=action.messaging_action,
        pricing_change_pct=action.pricing_change_pct,
        support_policy=action.support_policy,
        rationale=action.rationale or action.rationale_summary,
        expected_channels=_aggregate_effect_channels(scheduled_effects),
        scheduled_effect_ids=[effect.effect_id for effect in scheduled_effects],
        observed_alerts=[alert for event in step_events for alert in event["alerts"]] + policy_alerts,
        governance_flags=[constraint.constraint_id for constraint in governance_flags],
    )
    episode["decision_ledger"].append(decision)
    decision_lookup = {entry.decision_id: entry for entry in episode["decision_ledger"]}
    for effect in realized_effects:
        if effect.decision_id and effect.decision_id in decision_lookup:
            decision_lookup[effect.decision_id].realized_effect_ids.append(effect.effect_id)

    renewal_alerts = [effect.summary for effect in renewal_effects]
    episode["alerts"] = [alert for event in step_events for alert in event["alerts"]] + policy_alerts + renewal_alerts
    episode["realized_effects"] = realized_effects
    episode["step_index"] = next_step
    episode["action_history"].append(action)
    episode["goal_history"].append(current_goal_name(hidden_goal, episode["step_index"]))
    episode["latent_utility_history"].append(
        compute_utility(
            episode["dashboard"].to_metric_vector(),
            HiddenGoal(
                archetype=hidden_goal.archetype,
                weights=active_weights(hidden_goal, episode["step_index"]),
                alpha=hidden_goal.alpha,
            ),
        )
    )

    return {
        "invalid": invalid,
        "previous_dashboard": previous_dashboard,
        "new_dashboard": episode["dashboard"].model_copy(deep=True),
        "spent_budget": spent_budget,
        "remaining_budget": episode["budget_remaining"],
        "step_events": step_events,
        "scheduled_effects": scheduled_effects,
        "realized_effects": realized_effects,
        "decision_id": decision_id,
        "governance_flags": [constraint.constraint_id for constraint in governance_flags],
        "policy_alerts": policy_alerts,
    }
