"""Task 2: roadmap prioritization under budget."""

from __future__ import annotations

import itertools
import random

from latentgoalops.models import CustomerAccount, DashboardState, InitiativeItem, StakeholderPersona
from latentgoalops.server.hidden_goals import HiddenGoal
from latentgoalops.server.tasks.template_bank import load_initiative_effects


def _initiative_value(kpi_deltas: dict[str, float], weights: dict[str, float]) -> float:
    return sum(float(kpi_deltas.get(channel, 0.0)) * float(weights.get(channel, 0.0)) for channel in weights)


def _beneficiary_segments(kind: str) -> list[str]:
    if kind == "growth":
        return ["self_serve", "smb", "mid_market"]
    if kind == "retention":
        return ["mid_market", "enterprise", "strategic"]
    if kind == "revenue":
        return ["mid_market", "enterprise", "strategic"]
    return ["enterprise", "strategic", "smb"]


def _policy_tags(name: str, kind: str) -> list[str]:
    tags = []
    if "pricing" in name:
        tags.append("pricing_guardrail")
    if kind == "efficiency":
        tags.append("margin_guardrail")
    if "support" in name:
        tags.append("sla_guardrail")
    return tags


def _goal_overlay(hidden_goal: HiddenGoal, accounts: list[CustomerAccount], market_context) -> str:
    at_risk_accounts = [account for account in accounts if account.renewal_window_days <= 30][:2]
    if hidden_goal.archetype.value == "growth":
        pipeline_gap = max(0.0, 0.65 - float(market_context.sales_pipeline_health))
        return (
            f"Activation in self-serve looks softer than expected, and the pipeline gap is around {pipeline_gap:.2f}. "
            "A coherent top-of-funnel story would help."
        )
    if hidden_goal.archetype.value == "retention":
        names = ", ".join(account.company_name for account in at_risk_accounts) or "top enterprise accounts"
        return (
            f"Several renewal decisions are tightening, especially around {names}. "
            "Leadership wants the roadmap to reduce avoidable trust erosion."
        )
    if hidden_goal.archetype.value == "revenue":
        expansion_accounts = sorted(accounts, key=lambda account: account.expansion_potential, reverse=True)[:2]
        names = ", ".join(account.company_name for account in expansion_accounts) or "larger accounts"
        return (
            f"The board conversation is shifting toward monetization leverage, with expansion potential concentrated in {names}. "
            "They want moves that improve the revenue narrative without reckless churn risk."
        )
    return (
        f"Ops margin is sitting near {market_context.gross_margin_target:.2f} target pressure while infra cost remains elevated. "
        "Teams are being pushed to reduce cost-to-serve and avoid brittle execution."
    )


def _stakeholder_note(
    persona: StakeholderPersona,
    hidden_goal: HiddenGoal,
    accounts: list[CustomerAccount],
    market_context,
) -> str:
    role_defaults = {
        "CEO": "wants a tight portfolio that can be defended in the next staff review.",
        "CFO": "is scanning for initiatives with visible payback and limited downside.",
        "CTO": "is worried about execution drag, reliability debt, and delivery sequencing.",
        "Head of CS": "keeps pointing at renewal pressure, trust-sensitive accounts, and support load.",
        "Growth Lead": "cares about activation quality and whether new bets can compound.",
    }
    role_specific = {
        "CEO": _goal_overlay(hidden_goal, accounts, market_context),
        "CFO": f"Board pressure is {market_context.board_pressure_level:.2f}, so sequencing and downside control matter.",
        "CTO": "The team can absorb a focused bundle, but messy cross-team sequencing will show up quickly.",
        "Head of CS": "Several linked accounts are sensitive to rollout quality, especially when trust is already thin.",
        "Growth Lead": "The best bundle should compound rather than scatter effort across unrelated bets.",
    }
    return (
        f"{persona.name} ({persona.role}) {role_defaults.get(persona.role, 'is pushing for a coherent plan.')} "
        f"{role_specific.get(persona.role, '')}"
    )


def _build_variants(rng: random.Random, accounts: list[CustomerAccount], split: str) -> list[InitiativeItem]:
    templates = load_initiative_effects(split)
    title_suffixes = [
        "for enterprise segment",
        "with analytics refresh",
        "for self-serve funnel",
        "for power users",
        "with AI-assisted ops",
        "for board narrative",
    ]
    items: list[InitiativeItem] = []
    counter = 1
    for name, template in templates.items():
        kind = str(template["kind"])
        beneficiary_segments = _beneficiary_segments(kind)
        candidate_accounts = [account for account in accounts if account.segment in beneficiary_segments]
        linked_accounts = (
            [
                account.account_id
                for account in rng.sample(candidate_accounts, k=min(len(candidate_accounts), rng.randint(1, 3)))
            ]
            if candidate_accounts
            else []
        )
        items.append(
            InitiativeItem(
                item_id=f"{name}_{counter}",
                title=name.replace("_", " ").title(),
                description=f"Core initiative to {name.replace('_', ' ')}.",
                cost=float(template["cost"]),
                kind=kind,
                kpi_deltas={key: float(value) for key, value in template["deltas"].items()},
                uncertainty_band=0.08,
                stakeholder_tag="core_team",
                lag_steps=int(template.get("lag_steps", 1)),
                effect_window=int(template.get("effect_window", 1)),
                delivery_note=str(template.get("delivery_note", "") or ""),
                beneficiary_segments=beneficiary_segments,
                beneficiary_account_ids=linked_accounts,
                implementation_risk=round(rng.uniform(0.08, 0.35), 3),
                policy_tags=_policy_tags(name, kind),
            )
        )
        counter += 1
        variant_scale = 0.75 + rng.random() * 0.5
        items.append(
            InitiativeItem(
                item_id=f"{name}_{counter}",
                title=f"{name.replace('_', ' ').title()} {rng.choice(title_suffixes)}",
                description=f"Variant execution path for {name.replace('_', ' ')}.",
                cost=float(template["cost"]) + rng.choice([0, 1]),
                kind=kind,
                kpi_deltas={key: round(float(value) * variant_scale, 4) for key, value in template["deltas"].items()},
                uncertainty_band=0.12,
                stakeholder_tag=rng.choice(["CEO", "CTO", "Head of CS", "Growth Lead"]),
                lag_steps=int(template.get("lag_steps", 1)) + rng.choice([0, 1]),
                effect_window=int(template.get("effect_window", 1)),
                delivery_note=str(template.get("delivery_note", "") or ""),
                beneficiary_segments=beneficiary_segments,
                beneficiary_account_ids=linked_accounts,
                implementation_risk=round(rng.uniform(0.12, 0.42), 3),
                policy_tags=_policy_tags(name, kind),
            )
        )
        counter += 1
    return items


def _dashboard_baseline() -> DashboardState:
    return DashboardState(
        dau=22000.0,
        mau=97000.0,
        d7_retention=0.43,
        d30_retention=0.26,
        mrr=51000.0,
        arpu=101.0,
        cac=63.0,
        churn_rate=0.041,
        ops_margin=0.38,
        infra_cost_per_unit=1.92,
        support_ticket_volume=120,
    )


def _item_prefix(item_id: str) -> str:
    prefix, _, suffix = item_id.rpartition("_")
    if suffix.isdigit() and prefix:
        return prefix
    return item_id


def _attach_bundle_structure(backlog: list[InitiativeItem], accounts_by_id: dict[str, CustomerAccount]) -> None:
    by_prefix: dict[str, list[InitiativeItem]] = {}
    for item in backlog:
        by_prefix.setdefault(_item_prefix(item.item_id), []).append(item)

    def first(prefix: str) -> InitiativeItem | None:
        candidates = by_prefix.get(prefix, [])
        return candidates[0] if candidates else None

    referral = first("launch_referral_loop")
    onboarding = first("improve_onboarding")
    pricing = first("ship_usage_pricing")
    analytics = first("launch_admin_analytics")
    infra = first("optimize_infra")
    incident = first("refactor_incident_tooling")
    triage = first("automate_support_triage")
    login_fix = first("fix_login_bug")

    if referral and onboarding:
        referral.requires_item_ids = [onboarding.item_id]
        referral.synergy_item_ids = [onboarding.item_id]
        onboarding.synergy_item_ids = sorted(set(onboarding.synergy_item_ids + [referral.item_id]))
    if pricing and analytics:
        pricing.requires_item_ids = [analytics.item_id]
        pricing.synergy_item_ids = [analytics.item_id]
        analytics.synergy_item_ids = sorted(set(analytics.synergy_item_ids + [pricing.item_id]))
    if pricing and referral:
        pricing.conflicts_with_ids = sorted(set(pricing.conflicts_with_ids + [referral.item_id]))
        referral.conflicts_with_ids = sorted(set(referral.conflicts_with_ids + [pricing.item_id]))
    if infra and incident:
        infra.synergy_item_ids = sorted(set(infra.synergy_item_ids + [incident.item_id]))
        incident.synergy_item_ids = sorted(set(incident.synergy_item_ids + [infra.item_id]))
    if login_fix and onboarding:
        login_fix.synergy_item_ids = sorted(set(login_fix.synergy_item_ids + [onboarding.item_id]))

    for item in backlog:
        risk_notes: list[str] = []
        linked_accounts = [accounts_by_id[account_id] for account_id in item.beneficiary_account_ids if account_id in accounts_by_id]
        if any(account.renewal_window_days <= 30 for account in linked_accounts):
            risk_notes.append("Touches accounts that are already inside a near-term renewal window.")
        if any(account.segment in {"enterprise", "strategic"} for account in linked_accounts):
            risk_notes.append("Rollout quality matters because several linked accounts are high-touch or strategic.")
        if "pricing_guardrail" in item.policy_tags:
            risk_notes.append("Can create monetization upside, but sequencing matters near sensitive enterprise contracts.")
        if "sla_guardrail" in item.policy_tags:
            risk_notes.append("Operational efficiency wins can backfire if service expectations are already fragile.")
        if item.requires_item_ids:
            risk_notes.append("A visible prerequisite is missing if this ships in isolation.")
        if item.conflicts_with_ids:
            risk_notes.append("This portfolio can create mixed signals if paired with a conflicting initiative.")
        item.risk_notes = risk_notes


def _bundle_value(selected_item_ids: list[str], backlog: list[InitiativeItem], weights: dict[str, float]) -> float:
    chosen = {item.item_id: item for item in backlog if item.item_id in set(selected_item_ids)}
    if not chosen:
        return 0.0

    score = 0.0
    for item in chosen.values():
        base = _initiative_value(item.kpi_deltas, weights)
        if item.requires_item_ids and not any(required in chosen for required in item.requires_item_ids):
            base *= 0.55
        if item.synergy_item_ids and any(synergy in chosen for synergy in item.synergy_item_ids):
            base *= 1.12
        score += base

    conflict_penalty = 0.0
    for item in chosen.values():
        for conflict_id in item.conflicts_with_ids:
            if conflict_id in chosen and item.item_id < conflict_id:
                conflict_penalty += 0.18 * min(
                    _initiative_value(item.kpi_deltas, weights),
                    _initiative_value(chosen[conflict_id].kpi_deltas, weights),
                )

    execution_penalty = sum(item.implementation_risk * 0.03 for item in chosen.values())
    concentration_bonus = 0.02 * max(0, len({item.kind for item in chosen.values()}) == 1)
    return max(0.0, score - conflict_penalty - execution_penalty + concentration_bonus)


def solve_oracle_selection(backlog: list[InitiativeItem], budget: int, weights: dict[str, float]) -> tuple[list[str], float]:
    """Search over visible bundles to find the best subset under budget."""
    best_selection: list[str] = []
    best_value = 0.0
    for subset_size in range(len(backlog) + 1):
        for subset in itertools.combinations(backlog, subset_size):
            cost = sum(int(round(item.cost)) for item in subset)
            if cost > budget:
                continue
            selected_ids = [item.item_id for item in subset]
            value = _bundle_value(selected_ids, backlog, weights)
            if value > best_value:
                best_value = value
                best_selection = selected_ids
    return best_selection, best_value


def random_baseline_value(
    rng: random.Random,
    backlog: list[InitiativeItem],
    budget: int,
    weights: dict[str, float],
    samples: int = 128,
) -> float:
    """Estimate a reproducible random baseline for normalization."""
    values: list[float] = []
    for _ in range(samples):
        shuffled = backlog[:]
        rng.shuffle(shuffled)
        remaining = budget
        selected_ids: list[str] = []
        for item in shuffled:
            cost = int(round(item.cost))
            if cost <= remaining and rng.random() > 0.45:
                remaining -= cost
                selected_ids.append(item.item_id)
        values.append(_bundle_value(selected_ids, backlog, weights))
    return sum(values) / max(len(values), 1)


def build_task2_episode(
    rng: random.Random,
    hidden_goal: HiddenGoal,
    budget: int,
    world: dict,
    split: str = "core",
) -> dict:
    """Create the roadmap prioritization episode."""
    accounts: list[CustomerAccount] = world["accounts"]
    stakeholders: list[StakeholderPersona] = world["stakeholders"]
    backlog = _build_variants(rng, accounts, split)
    rng.shuffle(backlog)
    backlog = backlog[: rng.randint(10, 15)]
    accounts_by_id = {account.account_id: account for account in accounts}
    _attach_bundle_structure(backlog, accounts_by_id)

    oracle_selection, oracle_value = solve_oracle_selection(backlog, budget, hidden_goal.weights)
    random_value = random_baseline_value(
        random.Random(rng.randint(0, 10_000_000)),
        backlog,
        budget,
        hidden_goal.weights,
    )

    visible_stakeholders = stakeholders[:4]
    stakeholder_notes = [
        _stakeholder_note(persona, hidden_goal, accounts, world["market_context"])
        for persona in visible_stakeholders
    ]
    visible_accounts = accounts[:5]

    return {
        "dashboard": _dashboard_baseline(),
        "backlog": backlog,
        "accounts": visible_accounts,
        "stakeholders": visible_stakeholders,
        "teams": world["teams"],
        "market_context": world["market_context"],
        "governance_constraints": world["governance_constraints"],
        "sprint_budget": float(budget),
        "stakeholder_notes": stakeholder_notes,
        "oracle_selection": oracle_selection,
        "oracle_value": oracle_value,
        "random_value": random_value,
        "task_summary": "Select the strongest roadmap bundle under budget while balancing sequencing, conflicts, account exposure, and stakeholder pressure.",
    }


def selection_value(selected_item_ids: list[str], backlog: list[InitiativeItem], weights: dict[str, float]) -> float:
    """Score an agent selection under the hidden utility weights."""
    return _bundle_value(selected_item_ids, backlog, weights)
