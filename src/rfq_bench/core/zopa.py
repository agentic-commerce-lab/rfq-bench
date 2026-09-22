"""Zone of Possible Agreement and scenario degeneracy checks."""

from __future__ import annotations

from typing import Any, NamedTuple

from rfq_bench.core.contracts import Scenario
from rfq_bench.core.outcome_space import enumerate_outcomes
from rfq_bench.core.utility import ideal_utility, outcome_utility, reservation_utility


class ZopaResult(NamedTuple):
    exists: bool
    outcomes: list[dict[str, Any]]  # outcomes acceptable to both parties


def compute_zopa(scenario: Scenario) -> ZopaResult:
    """Outcomes at least as good as the BATNA for *both* parties."""
    d_buyer = reservation_utility(scenario.buyer)
    d_seller = reservation_utility(scenario.seller)
    acceptable: list[dict[str, Any]] = []
    for outcome in enumerate_outcomes(scenario.issues):
        u_buyer = outcome_utility(scenario.buyer, scenario.issues, outcome)
        u_seller = outcome_utility(scenario.seller, scenario.issues, outcome)
        if u_buyer >= d_buyer and u_seller >= d_seller:
            acceptable.append(outcome)
    return ZopaResult(exists=bool(acceptable), outcomes=acceptable)


def is_degenerate(scenario: Scenario, role: str) -> bool:
    """True when ideal equals BATNA for ``role`` (score is undefined)."""
    prefs = scenario.preferences(role)  # type: ignore[arg-type]
    return ideal_utility(prefs, scenario.issues) == reservation_utility(prefs)
