"""Weighted-additive utility, ideal point, and reservation value.

All utilities are normalized to [0, 1] by dividing out the party's total issue
weight, so results are independent of the absolute weight scale.
"""

from __future__ import annotations

from typing import Any

from rfq_bench.core.contracts import Issue, PartyPreferences
from rfq_bench.core.outcome_space import enumerate_outcomes


def outcome_utility(prefs: PartyPreferences, issues: list[Issue], outcome: dict[str, Any]) -> float:
    """Normalized utility in [0, 1] of ``outcome`` for ``prefs``."""
    by_name = {i.name: i for i in issues}
    total_weight = sum(ip.weight for ip in prefs.issues)
    if total_weight <= 0:
        raise ValueError("total issue weight must be positive")
    acc = 0.0
    for ip in prefs.issues:
        issue = by_name[ip.issue]
        idx = issue.index_of(outcome[ip.issue])
        acc += ip.weight * ip.value_utilities[idx]
    return acc / total_weight


def ideal_utility(prefs: PartyPreferences, issues: list[Issue]) -> float:
    """Best attainable utility within the outcome space.

    Issues are additively independent, so the maximum of the sum equals the sum
    of per-issue maxima.
    """
    total_weight = sum(ip.weight for ip in prefs.issues)
    acc = sum(ip.weight * max(ip.value_utilities) for ip in prefs.issues)
    return acc / total_weight


def worst_utility(prefs: PartyPreferences, issues: list[Issue]) -> float:
    """Lowest attainable utility within the outcome space (diagnostic only)."""
    total_weight = sum(ip.weight for ip in prefs.issues)
    acc = sum(ip.weight * min(ip.value_utilities) for ip in prefs.issues)
    return acc / total_weight


def reservation_utility(prefs: PartyPreferences) -> float:
    """The party's BATNA utility: the disagreement outcome value."""
    return prefs.batna


def best_outcome(prefs: PartyPreferences, issues: list[Issue]) -> dict[str, Any]:
    """A concrete outcome achieving the ideal utility (ties broken by index)."""
    best: dict[str, Any] | None = None
    best_u = float("-inf")
    for outcome in enumerate_outcomes(issues):
        u = outcome_utility(prefs, issues, outcome)
        if u > best_u:
            best_u, best = u, outcome
    assert best is not None
    return best
