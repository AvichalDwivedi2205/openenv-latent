"""Task 4: capital allocation under uncertainty."""

from __future__ import annotations

import random

from latentgoalops.models import CustomerAccount, DashboardState, InitiativeItem, StakeholderPersona
from latentgoalops.server.hidden_goals import HiddenGoal


def _initiative_value(kpi_deltas: dict[str, float], weights: dict[str, float]) -> float:
    return sum(float(kpi_deltas.get(channel, 0.0)) * float(weights.get(channel, 0.0)) for channel in weights)


def _beneficiary_segments(kind: str) -> list[str]:
    if kind == "growth":
        return ["self_serve", "smb", "mid_market"]
    if kind == "retention":
        return ["mid_market", "enterprise", "strategic"]
    if kind == "revenue":
        return ["mid_market", "enterprise", "strategic"]
    return ["smb", "enterprise", "strategic"]


def _dashboard_baseline() -> DashboardState:
    return DashboardState(
        dau=25500.0,
        mau=112000.0,
        d7_retention=0.45,
        d30_retention=0.28,
        mrr=59000.0,
        arpu=108.0,
        cac=69.0,
        churn_rate=0.039,
        ops_margin=0.36,
        infra_cost_per_unit=1.84,
        support_ticket_volume=128,
    )


def _program_bank(split: str) -> list[dict]:
    core = [
        {
            "item_id": "plg_onboarding_fund",
            "title": "PLG Onboarding Fund",
            "kind": "growth",
            "description": "Increase activation by funding guided onboarding improvements and lifecycle nudges.",
            "deltas": {"growth": 0.030, "retention": 0.010, "revenue": 0.006, "efficiency": -0.004},
            "allocation_max": 4.0,
            "saturation_point": 2.0,
            "requires": [],
            "conflicts": ["pricing_packaging_lab"],
            "synergy": ["referral_acceleration_pool"],
        },
        {
            "item_id": "referral_acceleration_pool",
            "title": "Referral Acceleration Pool",
            "kind": "growth",
            "description": "Layer referral incentives and ambassador loops on top of a stronger activation funnel.",
            "deltas": {"growth": 0.034, "retention": 0.004, "revenue": 0.008, "efficiency": -0.006},
            "allocation_max": 3.0,
            "saturation_point": 2.0,
            "requires": ["plg_onboarding_fund"],
            "conflicts": [],
            "synergy": ["plg_onboarding_fund"],
        },
        {
            "item_id": "renewal_rescue_pod",
            "title": "Renewal Rescue Pod",
            "kind": "retention",
            "description": "Fund executive outreach and success coverage for renewal-risk accounts.",
            "deltas": {"growth": 0.004, "retention": 0.034, "revenue": 0.012, "efficiency": -0.006},
            "allocation_max": 4.0,
            "saturation_point": 3.0,
            "requires": [],
            "conflicts": [],
            "synergy": ["reliability_hardening_reserve"],
        },
        {
            "item_id": "reliability_hardening_reserve",
            "title": "Reliability Hardening Reserve",
            "kind": "efficiency",
            "description": "Reduce incident risk and noisy escalations through focused reliability work.",
            "deltas": {"growth": 0.002, "retention": 0.018, "revenue": 0.006, "efficiency": 0.030},
            "allocation_max": 4.0,
            "saturation_point": 2.0,
            "requires": [],
            "conflicts": [],
            "synergy": ["renewal_rescue_pod"],
        },
        {
            "item_id": "pricing_packaging_lab",
            "title": "Pricing Packaging Lab",
            "kind": "revenue",
            "description": "Fund packaging analysis and monetization experiments for higher-value segments.",
            "deltas": {"growth": -0.004, "retention": -0.006, "revenue": 0.038, "efficiency": 0.006},
            "allocation_max": 3.0,
            "saturation_point": 2.0,
            "requires": [],
            "conflicts": ["plg_onboarding_fund"],
            "synergy": ["sales_enablement_grants"],
        },
        {
            "item_id": "sales_enablement_grants",
            "title": "Sales Enablement Grants",
            "kind": "revenue",
            "description": "Equip account teams with ROI assets and pilot support for expansion conversations.",
            "deltas": {"growth": 0.010, "retention": 0.008, "revenue": 0.030, "efficiency": 0.000},
            "allocation_max": 3.0,
            "saturation_point": 2.0,
            "requires": [],
            "conflicts": [],
            "synergy": ["pricing_packaging_lab"],
        },
        {
            "item_id": "workflow_automation_pool",
            "title": "Workflow Automation Pool",
            "kind": "efficiency",
            "description": "Automate repetitive support and finance operations to protect margin and team capacity.",
            "deltas": {"growth": 0.000, "retention": 0.012, "revenue": 0.008, "efficiency": 0.034},
            "allocation_max": 4.0,
            "saturation_point": 3.0,
            "requires": [],
            "conflicts": [],
            "synergy": [],
        },
    ]
    heldout = [
        {
            "item_id": "partner_marketplace_seed",
            "title": "Partner Marketplace Seed",
            "kind": "growth",
            "description": "Seed co-marketing and channel experiments through a partner marketplace program.",
            "deltas": {"growth": 0.032, "retention": 0.006, "revenue": 0.010, "efficiency": -0.004},
            "allocation_max": 4.0,
            "saturation_point": 2.0,
            "requires": [],
            "conflicts": ["enterprise_procurement_desk"],
            "synergy": ["developer_conversion_grants"],
        },
        {
            "item_id": "developer_conversion_grants",
            "title": "Developer Conversion Grants",
            "kind": "growth",
            "description": "Fund migration tooling and trial assistance for developer-led accounts.",
            "deltas": {"growth": 0.030, "retention": 0.010, "revenue": 0.006, "efficiency": -0.005},
            "allocation_max": 3.0,
            "saturation_point": 2.0,
            "requires": [],
            "conflicts": [],
            "synergy": ["partner_marketplace_seed"],
        },
        {
            "item_id": "security_assurance_pool",
            "title": "Security Assurance Pool",
            "kind": "retention",
            "description": "Fund security reviews, trust recovery, and hands-on assurance for sensitive accounts.",
            "deltas": {"growth": 0.002, "retention": 0.036, "revenue": 0.010, "efficiency": -0.003},
            "allocation_max": 4.0,
            "saturation_point": 3.0,
            "requires": [],
            "conflicts": [],
            "synergy": ["platform_resilience_reserve"],
        },
        {
            "item_id": "enterprise_procurement_desk",
            "title": "Enterprise Procurement Desk",
            "kind": "revenue",
            "description": "Fund commercial packaging support and procurement acceleration for expansion-ready accounts.",
            "deltas": {"growth": 0.004, "retention": 0.006, "revenue": 0.040, "efficiency": 0.004},
            "allocation_max": 4.0,
            "saturation_point": 2.0,
            "requires": [],
            "conflicts": ["partner_marketplace_seed"],
            "synergy": ["seat_tier_packaging_lab"],
        },
        {
            "item_id": "seat_tier_packaging_lab",
            "title": "Seat Tier Packaging Lab",
            "kind": "revenue",
            "description": "Support pricing ops and packaging tests for seat-tier monetization.",
            "deltas": {"growth": -0.004, "retention": -0.004, "revenue": 0.036, "efficiency": 0.008},
            "allocation_max": 3.0,
            "saturation_point": 2.0,
            "requires": [],
            "conflicts": [],
            "synergy": ["enterprise_procurement_desk"],
        },
        {
            "item_id": "platform_resilience_reserve",
            "title": "Platform Resilience Reserve",
            "kind": "efficiency",
            "description": "Fund resiliency and ops hardening to lower cost-to-serve and surprise incidents.",
            "deltas": {"growth": 0.000, "retention": 0.016, "revenue": 0.006, "efficiency": 0.036},
            "allocation_max": 4.0,
            "saturation_point": 2.0,
            "requires": [],
            "conflicts": [],
            "synergy": ["security_assurance_pool"],
        },
        {
            "item_id": "backoffice_standardization_pool",
            "title": "Backoffice Standardization Pool",
            "kind": "efficiency",
            "description": "Standardize support and finance workflows to reduce overhead.",
            "deltas": {"growth": 0.000, "retention": 0.010, "revenue": 0.008, "efficiency": 0.034},
            "allocation_max": 4.0,
            "saturation_point": 3.0,
            "requires": [],
            "conflicts": [],
            "synergy": [],
        },
    ]
    return heldout if split == "heldout" else core


def _stakeholder_note(
    persona: StakeholderPersona,
    accounts: list[CustomerAccount],
    market_context,
) -> str:
    near_term = sorted(accounts, key=lambda account: account.renewal_window_days)[:2]
    expansion = sorted(accounts, key=lambda account: account.expansion_potential, reverse=True)[:2]
    names = ", ".join(account.company_name for account in near_term) or "renewal-sensitive accounts"
    expansion_names = ", ".join(account.company_name for account in expansion) or "expansion-ready accounts"
    overlay = (
        f"The room keeps circling the same pressures: fragile renewals around {names}, "
        f"commercial upside in {expansion_names}, and runway discipline with {market_context.cash_runway_months} months left."
    )
    role_prefix = {
        "CEO": "wants one crisp capital story rather than a scattered budget.",
        "CFO": "is focused on payback discipline and how quickly each dollar turns into visible progress.",
        "CTO": "wants fewer fragmented bets and more programs that the team can execute cleanly.",
        "Head of CS": "keeps bringing the discussion back to renewals, trust, and support load.",
        "Growth Lead": "cares about whether the budget compounds into a durable demand engine.",
    }
    return f"{persona.name} ({persona.role}) {role_prefix.get(persona.role, 'wants a focused allocation plan.')} {overlay}"


def _build_programs(rng: random.Random, accounts: list[CustomerAccount], split: str) -> list[InitiativeItem]:
    items: list[InitiativeItem] = []
    for raw in _program_bank(split):
        candidate_accounts = [
            account for account in accounts if account.segment in _beneficiary_segments(str(raw["kind"]))
        ]
        linked_accounts = [
            account.account_id
            for account in rng.sample(candidate_accounts, k=min(len(candidate_accounts), rng.randint(1, 3)))
        ] if candidate_accounts else []
        risk_notes = [
            f"Capital can be deployed in 1-point increments up to {raw['allocation_max']:.0f}.",
            f"Visible returns look strongest through about {raw['saturation_point']:.0f} points before tapering.",
        ]
        if raw["requires"]:
            risk_notes.append("Part of the upside depends on another visible program being funded alongside it.")
        if raw["conflicts"]:
            risk_notes.append("Can create mixed strategic signals if heavily funded beside a conflicting program.")
        if any(
            account.renewal_window_days <= 30
            for account in candidate_accounts
            if account.account_id in linked_accounts
        ):
            risk_notes.append("Some beneficiary accounts are already near renewal, so rollout quality matters.")
        items.append(
            InitiativeItem(
                item_id=str(raw["item_id"]),
                title=str(raw["title"]),
                description=str(raw["description"]),
                cost=1.0,
                kind=str(raw["kind"]),
                kpi_deltas={key: float(value) for key, value in raw["deltas"].items()},
                uncertainty_band=0.10,
                stakeholder_tag="capital_committee",
                lag_steps=1,
                effect_window=2,
                delivery_note="Allocate discrete budget points rather than selecting the program all-or-nothing.",
                beneficiary_segments=_beneficiary_segments(str(raw["kind"])),
                beneficiary_account_ids=linked_accounts,
                implementation_risk=round(rng.uniform(0.08, 0.28), 3),
                policy_tags=[],
                requires_item_ids=list(raw["requires"]),
                conflicts_with_ids=list(raw["conflicts"]),
                synergy_item_ids=list(raw["synergy"]),
                risk_notes=risk_notes,
                allocation_unit=1.0,
                allocation_max=float(raw["allocation_max"]),
                saturation_point=float(raw["saturation_point"]),
            )
        )
    return items


def _curve_units(amount: float, saturation_point: float) -> float:
    if amount <= 0.0:
        return 0.0
    if amount <= saturation_point:
        return amount
    return saturation_point + 0.55 * (amount - saturation_point)


def _allocation_value(budget_allocations: dict[str, float], backlog: list[InitiativeItem], weights: dict[str, float]) -> float:
    chosen: dict[str, tuple[InitiativeItem, float]] = {}
    for item in backlog:
        raw_amount = float(budget_allocations.get(item.item_id, 0.0))
        amount = max(0.0, min(raw_amount, float(item.allocation_max or 0.0)))
        if amount > 0.0:
            chosen[item.item_id] = (item, amount)

    if not chosen:
        return 0.0

    score = 0.0
    risk_penalty = 0.0
    allocated_by_kind: dict[str, float] = {}
    for item_id, (item, amount) in chosen.items():
        effective_units = _curve_units(amount, float(item.saturation_point or amount))
        base = _initiative_value(item.kpi_deltas, weights) * effective_units
        if item.requires_item_ids and not any(required in chosen for required in item.requires_item_ids):
            base *= 0.60
        if item.synergy_item_ids and any(synergy in chosen for synergy in item.synergy_item_ids):
            base *= 1.10
        score += base
        risk_penalty += float(item.implementation_risk) * 0.016 * amount
        if amount > float(item.saturation_point or amount) + 1.0:
            risk_penalty += 0.018 * (amount - float(item.saturation_point or amount) - 1.0)
        allocated_by_kind[item.kind] = allocated_by_kind.get(item.kind, 0.0) + amount

    conflict_penalty = 0.0
    for item_id, (item, amount) in chosen.items():
        for conflict_id in item.conflicts_with_ids:
            if conflict_id in chosen and item_id < conflict_id:
                other_item, other_amount = chosen[conflict_id]
                conflict_penalty += 0.10 * min(
                    _initiative_value(item.kpi_deltas, weights) * amount,
                    _initiative_value(other_item.kpi_deltas, weights) * other_amount,
                )

    active_programs = len(chosen)
    spread_penalty = 0.018 * max(0, active_programs - 4)
    total_allocated = sum(amount for _, amount in chosen.values())
    focus_bonus = 0.0
    if total_allocated > 0.0:
        dominant_share = max(allocated_by_kind.values()) / total_allocated
        if dominant_share >= 0.55:
            focus_bonus = 0.03
    return max(0.0, score - risk_penalty - conflict_penalty - spread_penalty + focus_bonus)


def solve_oracle_allocations(
    backlog: list[InitiativeItem],
    budget: int,
    weights: dict[str, float],
) -> tuple[dict[str, float], float]:
    """Exhaustive integer search over visible allocation programs."""
    best_allocations: dict[str, float] = {}
    best_value = 0.0
    allocation_limits = [int(round(float(item.allocation_max or 0.0))) for item in backlog]

    def search(index: int, remaining: int, current: dict[str, float]) -> None:
        nonlocal best_allocations, best_value
        if index >= len(backlog):
            value = _allocation_value(current, backlog, weights)
            if value > best_value:
                best_value = value
                best_allocations = {key: float(value) for key, value in current.items() if value > 0.0}
            return

        item = backlog[index]
        limit = min(allocation_limits[index], remaining)
        for amount in range(limit, -1, -1):
            if amount > 0:
                current[item.item_id] = float(amount)
            else:
                current.pop(item.item_id, None)
            search(index + 1, remaining - amount, current)
        current.pop(item.item_id, None)

    search(0, int(budget), {})
    return best_allocations, best_value


def random_baseline_value(
    rng: random.Random,
    backlog: list[InitiativeItem],
    budget: int,
    weights: dict[str, float],
    samples: int = 160,
) -> float:
    """Estimate a reproducible random-allocation baseline."""
    values: list[float] = []
    for _ in range(samples):
        remaining = int(budget)
        allocations: dict[str, float] = {}
        shuffled = backlog[:]
        rng.shuffle(shuffled)
        for item in shuffled:
            if remaining <= 0:
                break
            upper = min(remaining, int(round(float(item.allocation_max or 0.0))))
            amount = rng.randint(0, upper)
            if amount > 0:
                allocations[item.item_id] = float(amount)
                remaining -= amount
        values.append(_allocation_value(allocations, backlog, weights))
    return sum(values) / max(len(values), 1)


def build_task4_episode(
    rng: random.Random,
    hidden_goal: HiddenGoal,
    budget: int,
    world: dict,
    split: str = "core",
) -> dict:
    """Create the capital-allocation episode."""
    accounts: list[CustomerAccount] = world["accounts"]
    stakeholders: list[StakeholderPersona] = world["stakeholders"]
    backlog = _build_programs(rng, accounts, split)
    oracle_allocations, oracle_value = solve_oracle_allocations(backlog, budget, hidden_goal.weights)
    random_value = random_baseline_value(
        random.Random(rng.randint(0, 10_000_000)),
        backlog,
        budget,
        hidden_goal.weights,
    )
    role_priority = {
        "growth": {"Growth Lead": 0, "CEO": 1, "CTO": 2, "CFO": 3, "Head of CS": 4},
        "retention": {"Head of CS": 0, "CEO": 1, "CFO": 2, "CTO": 3, "Growth Lead": 4},
        "revenue": {"CFO": 0, "CEO": 1, "Growth Lead": 2, "Head of CS": 3, "CTO": 4},
        "efficiency": {"CTO": 0, "CFO": 1, "CEO": 2, "Head of CS": 3, "Growth Lead": 4},
    }[hidden_goal.archetype.value]
    visible_stakeholders = sorted(
        stakeholders[:4],
        key=lambda persona: role_priority.get(persona.role, 10),
    )
    stakeholder_notes = [
        _stakeholder_note(persona, accounts, world["market_context"])
        for persona in visible_stakeholders
    ]
    return {
        "dashboard": _dashboard_baseline(),
        "backlog": backlog,
        "accounts": accounts[:6],
        "stakeholders": visible_stakeholders,
        "teams": world["teams"],
        "market_context": world["market_context"],
        "governance_constraints": world["governance_constraints"],
        "sprint_budget": float(budget),
        "stakeholder_notes": stakeholder_notes,
        "oracle_allocations": oracle_allocations,
        "oracle_value": oracle_value,
        "random_value": random_value,
        "task_summary": (
            "Allocate discrete budget points across visible operating programs. "
            "Returns taper after saturation, some programs interact, and leadership only indirectly reveals which business outcome matters most."
        ),
    }


def allocation_value(budget_allocations: dict[str, float], backlog: list[InitiativeItem], weights: dict[str, float]) -> float:
    """Score an allocation plan under the hidden utility weights."""
    return _allocation_value(budget_allocations, backlog, weights)
