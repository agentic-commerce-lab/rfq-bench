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
    )


def _has_zopa(scenario: Scenario) -> bool:
    from rfq_bench.core.zopa import compute_zopa

    return compute_zopa(scenario).exists
