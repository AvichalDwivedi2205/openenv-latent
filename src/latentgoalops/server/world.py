"""Deterministic synthetic startup-world generation."""

from __future__ import annotations

import random

from latentgoalops.models import (
    CustomerAccount,
    GovernanceConstraint,
    InternalTeamState,
    MarketContext,
    StakeholderPersona,
)


COMPANY_PREFIXES = [
    "Aster",
    "Blue",
    "North",
    "Nova",
    "Clear",
    "Summit",
    "Vector",
    "Copper",
    "Horizon",
    "Pioneer",
    "Atlas",
    "Meridian",
]
COMPANY_SUFFIXES = ["Cloud", "Analytics", "Works", "Dynamics", "Health", "Logistics", "Capital", "Energy"]
INDUSTRIES = ["fintech", "healthcare", "logistics", "saas", "retail", "cybersecurity", "education"]
REGIONS = ["us-east", "us-west", "eu", "uk", "apac"]
SEGMENTS = ["self_serve", "smb", "mid_market", "enterprise", "strategic"]
COMMUNICATION_STYLES = ["concise", "demanding", "collaborative", "skeptical", "formal"]
ADOPTION_STAGES = ["trial", "ramping", "steady", "power_user", "at_risk"]


def _company_name(rng: random.Random, index: int) -> str:
    return f"{rng.choice(COMPANY_PREFIXES)} {rng.choice(COMPANY_SUFFIXES)} {index + 1}"


def _segment_ranges(segment: str) -> tuple[int, int, int, int]:
    if segment == "self_serve":
        return 5, 30, 500, 4_000
    if segment == "smb":
        return 20, 120, 4_000, 18_000
    if segment == "mid_market":
        return 80, 350, 18_000, 65_000
    if segment == "enterprise":
        return 200, 900, 65_000, 180_000
    return 600, 2_000, 180_000, 600_000


def _generate_accounts(rng: random.Random) -> list[CustomerAccount]:
    accounts: list[CustomerAccount] = []
    for index in range(12):
        segment = rng.choices(SEGMENTS, weights=[2, 3, 3, 2, 1], k=1)[0]
        seats_low, seats_high, acv_low, acv_high = _segment_ranges(segment)
        annual_contract_value = float(rng.randint(acv_low, acv_high))
        renewal_window_days = rng.randint(7, 120)
        expansion_potential = round(rng.uniform(0.1, 1.0), 3)
        relationship_health = round(rng.uniform(0.35, 0.95), 3)
        influence_weight = round(min(1.0, 0.25 + annual_contract_value / 600_000 + rng.uniform(0.0, 0.2)), 3)
        strategic_importance = round(min(1.0, influence_weight + (0.12 if segment in {"enterprise", "strategic"} else 0.0)), 3)
        accounts.append(
            CustomerAccount(
                account_id=f"acct_{index + 1}",
                company_name=_company_name(rng, index),
                segment=segment,
                industry=rng.choice(INDUSTRIES),
                region=rng.choice(REGIONS),
                seat_count=rng.randint(seats_low, seats_high),
                annual_contract_value=annual_contract_value,
                expansion_potential=expansion_potential,
                renewal_window_days=renewal_window_days,
                payment_risk=round(rng.uniform(0.0, 0.5), 3),
                support_tier="premium" if segment in {"enterprise", "strategic"} else rng.choice(["standard", "priority"]),
                sla_level="24x7" if segment == "strategic" else rng.choice(["business_hours", "priority", "24x7"]),
                security_sensitivity=round(rng.uniform(0.1, 1.0), 3),
                integration_complexity=round(rng.uniform(0.1, 1.0), 3),
                adoption_stage=rng.choice(ADOPTION_STAGES),
                relationship_health=relationship_health,
                champion_strength=round(rng.uniform(0.1, 1.0), 3),
                decision_maker_involved=bool(rng.random() < 0.45),
                churn_propensity=round(rng.uniform(0.05, 0.8), 3),
                communication_style=rng.choice(COMMUNICATION_STYLES),
                influence_weight=influence_weight,
                strategic_importance=strategic_importance,
            )
        )
    accounts.sort(key=lambda account: (account.strategic_importance, account.annual_contract_value), reverse=True)
    return accounts


def _generate_stakeholders(rng: random.Random) -> list[StakeholderPersona]:
    role_specs = [
        ("st_1", "Mira Chen", "CEO", "portfolio_balance", ["dau", "mrr"]),
        ("st_2", "Jon Alvarez", "CFO", "margin_discipline", ["mrr", "ops_margin"]),
        ("st_3", "Priya Rao", "CTO", "platform_reliability", ["ops_margin", "infra_cost_per_unit"]),
        ("st_4", "Lena Brooks", "Head of CS", "renewal_protection", ["d30_retention", "support_ticket_volume"]),
        ("st_5", "Evan Park", "Growth Lead", "topline_expansion", ["dau", "cac"]),
    ]
    stakeholders: list[StakeholderPersona] = []
    for persona_id, name, role, bias, metrics in role_specs:
        stakeholders.append(
            StakeholderPersona(
                persona_id=persona_id,
                name=name,
                role=role,
                agenda_bias=bias,
                risk_tolerance=round(rng.uniform(0.2, 0.9), 3),
                time_horizon=rng.choice(["short", "medium", "long"]),
                communication_style=rng.choice(COMMUNICATION_STYLES),
                political_power=round(rng.uniform(0.4, 1.0), 3),
                credibility=round(rng.uniform(0.5, 1.0), 3),
                patience=round(rng.uniform(0.2, 0.9), 3),
                favorite_metrics=metrics,
                hard_constraints=[],
                soft_preferences=[],
            )
        )
    return stakeholders


def _generate_teams(rng: random.Random) -> list[InternalTeamState]:
    team_specs = [
        ("team_product", "product", ["onboarding", "pricing", "analytics"]),
        ("team_infra", "infra", ["latency", "reliability", "cost"]),
        ("team_support", "support", ["sla", "triage", "incident_response"]),
        ("team_growth", "growth", ["activation", "referrals", "campaigns"]),
    ]
    teams: list[InternalTeamState] = []
    for team_id, function, specialization in team_specs:
        teams.append(
            InternalTeamState(
                team_id=team_id,
                function=function,
                capacity=round(rng.uniform(0.4, 1.0), 3),
                burnout_risk=round(rng.uniform(0.1, 0.8), 3),
                execution_reliability=round(rng.uniform(0.45, 0.95), 3),
                specialization=specialization,
                cross_team_friction=round(rng.uniform(0.05, 0.4), 3),
                delivery_latency_days=rng.randint(1, 4),
            )
        )
    return teams


def _generate_market_context(rng: random.Random, accounts: list[CustomerAccount]) -> MarketContext:
    strategic_accounts_share = sum(1 for account in accounts if account.segment in {"enterprise", "strategic"}) / max(len(accounts), 1)
    board_pressure_level = round(rng.uniform(0.3, 0.95), 3)
    return MarketContext(
        funding_stage=rng.choice(["seed", "series_a", "series_b", "growth"]),
        cash_runway_months=rng.randint(8, 24),
        gross_margin_target=round(rng.uniform(0.45, 0.7), 3),
        board_pressure_level=board_pressure_level,
        competition_intensity=round(rng.uniform(0.25, 0.95), 3),
        seasonality=rng.choice(["steady", "end_of_quarter", "holiday_push", "budget_reset"]),
        compliance_exposure=round(rng.uniform(0.1, 0.9), 3),
        strategic_accounts_share=round(strategic_accounts_share, 3),
        sales_pipeline_health=round(rng.uniform(0.3, 0.9), 3),
    )


def _generate_governance_constraints(
    rng: random.Random,
    accounts: list[CustomerAccount],
    market_context: MarketContext,
) -> list[GovernanceConstraint]:
    has_strategic = any(account.segment == "strategic" for account in accounts)
    constraints = [
        GovernanceConstraint(
            constraint_id="pricing_guardrail",
            title="Pricing change guardrail",
            description="Avoid enterprise price changes above 5% when strategic renewals are within 30 days.",
            severity="high",
            affected_channels=["revenue", "retention"],
            threshold=0.05,
            penalty_hint="Higher churn and governance penalties if violated.",
        ),
        GovernanceConstraint(
            constraint_id="sla_guardrail",
            title="Strategic account SLA guardrail",
            description="Do not aggressively automate support for strategic or renewal-risk accounts.",
            severity="high" if has_strategic else "medium",
            affected_channels=["retention", "efficiency"],
            penalty_hint="Trust and renewal risk increase if violated.",
        ),
        GovernanceConstraint(
            constraint_id="margin_guardrail",
            title="Margin preservation guardrail",
            description="Do not run growth-heavy campaigns when ops margin is already below the board target.",
            severity="medium",
            affected_channels=["growth", "efficiency"],
            threshold=market_context.gross_margin_target,
            penalty_hint="Board pressure and cash efficiency worsen if violated.",
        ),
    ]
    rng.shuffle(constraints)
    return constraints


def build_world(rng: random.Random) -> dict:
    """Create the persistent startup world shared across tasks for one seed."""
    accounts = _generate_accounts(rng)
    stakeholders = _generate_stakeholders(rng)
    teams = _generate_teams(rng)
    market_context = _generate_market_context(rng, accounts)
    governance_constraints = _generate_governance_constraints(rng, accounts, market_context)
    return {
        "accounts": accounts,
        "stakeholders": stakeholders,
        "teams": teams,
        "market_context": market_context,
        "governance_constraints": governance_constraints,
    }
