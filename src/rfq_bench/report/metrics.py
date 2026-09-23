"""Separate-track metrics: outcome quality, validity/safety, operational cost.

None of these are folded into the strategy score. They are reported alongside it
so trade-offs (efficiency, agreement rate, safety, cost) stay visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rfq_bench.core.contracts import Scenario, Trace
from rfq_bench.core.outcome_space import enumerate_outcomes
from rfq_bench.core.utility import outcome_utility, reservation_utility


def joint_surplus(scenario: Scenario, outcome: dict[str, Any]) -> float:
    """Combined utility of both parties above their BATNAs."""
    ub = outcome_utility(scenario.buyer, scenario.issues, outcome)
    us = outcome_utility(scenario.seller, scenario.issues, outcome)
    return (ub - reservation_utility(scenario.buyer)) + (us - reservation_utility(scenario.seller))


def is_pareto_efficient(scenario: Scenario, outcome: dict[str, Any]) -> bool:
    """True iff no other legal outcome dominates ``outcome`` for both parties.

    (Higher ``pareto_rate`` = more agreements on the frontier = better.)
    """
    ub = outcome_utility(scenario.buyer, scenario.issues, outcome)
    us = outcome_utility(scenario.seller, scenario.issues, outcome)
    for other in enumerate_outcomes(scenario.issues):
        ob = outcome_utility(scenario.buyer, scenario.issues, other)
        os_ = outcome_utility(scenario.seller, scenario.issues, other)
        if ob >= ub and os_ >= us and (ob > ub or os_ > us):
            return False  # dominated by ``other`` -> not Pareto-efficient
    return True


# The issue whose agreed value is the money that changes hands.
PRICE_ISSUE = "price"


def revenue(scenario: Scenario, agreement: dict[str, Any] | None) -> float | None:
    """The agreed price: the seller's revenue (and the buyer's spend) for one episode.

    0 when there is no deal; None when the scenario has no price issue.
    """
    if not any(i.name == PRICE_ISSUE for i in scenario.issues):
        return None
    return 0.0 if agreement is None else float(agreement[PRICE_ISSUE])


def best_price(scenario: Scenario) -> float | None:
    """The seller's best price option (the most revenue a deal can bring)."""
    issue = next((i for i in scenario.issues if i.name == PRICE_ISSUE), None)
    if issue is None:
        return None
    utils = scenario.seller.preference_for(PRICE_ISSUE).value_utilities
    return float(max(zip(issue.values, utils, strict=True), key=lambda p: p[1])[0])


@dataclass(frozen=True)
class SideMetrics:
    strategy: str
    n: int
    agreement_rate: float
    mean_joint_surplus: float
    pareto_rate: float  # over agreements only
    mean_rounds_to_close: float
    mean_latency_s: float
    mean_token_cost: float
    walk_away_accuracy: float  # correct no-deal on no-ZOPA scenarios
    mean_cost_usd: float | None = None  # real USD/episode when the provider reports it
    # Sum of agreed prices over the episodes (no deal = 0), and the agreed price as a
    # share of the seller's best price option, averaged — comparable across scenarios
    # with different price scales. None when no scenario has a price issue.
    total_revenue: float | None = None
    mean_revenue_share: float | None = None


def side_metrics(strategy: str, traces: list[Trace], scenarios: dict[str, Scenario]) -> SideMetrics:
    n = len(traces)
    agreements = [t for t in traces if t.agreement is not None]
    joint = [joint_surplus(scenarios[t.scenario_id], t.agreement) for t in agreements]  # type: ignore[arg-type]
    pareto = [
        is_pareto_efficient(scenarios[t.scenario_id], t.agreement)  # type: ignore[arg-type]
        for t in agreements
    ]
    walk_cases = [
        t
        for t in traces
        if t.outcome_kind in ("walk_away", "no_deal") and not _has_zopa(scenarios[t.scenario_id])
    ]
    walk_correct = [t for t in walk_cases if t.validity.get("correct_walk_away", False)]

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    costs = [t.cost_usd for t in traces if t.cost_usd is not None]
    mean_cost = (sum(costs) / len(costs)) if costs else None

    revenues: list[float] = []
    shares: list[float] = []
    for t in traces:
        scenario = scenarios[t.scenario_id]
        r, best = revenue(scenario, t.agreement), best_price(scenario)
        if r is not None and best:
            revenues.append(r)
            shares.append(r / best)

    return SideMetrics(
        strategy=strategy,
        n=n,
        agreement_rate=len(agreements) / n if n else 0.0,
        mean_joint_surplus=mean(joint),
        pareto_rate=(sum(pareto) / len(pareto)) if pareto else 0.0,
        mean_rounds_to_close=mean([float(t.rounds_to_close) for t in traces]),
        mean_latency_s=mean([t.latency_s for t in traces]),
        mean_token_cost=mean([float(t.token_cost) for t in traces]),
        walk_away_accuracy=(len(walk_correct) / len(walk_cases)) if walk_cases else float("nan"),
        mean_cost_usd=mean_cost,
        total_revenue=sum(revenues) if revenues else None,
        mean_revenue_share=mean(shares) if shares else None,
    )


def _has_zopa(scenario: Scenario) -> bool:
    from rfq_bench.core.zopa import compute_zopa

    return compute_zopa(scenario).exists
