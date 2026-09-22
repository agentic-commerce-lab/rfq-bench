"""Separate-track metrics — notably that pareto efficiency is not inverted."""

from __future__ import annotations

from rfq_bench.core.contracts import (
    Issue,
    IssuePreference,
    PartyPreferences,
    Scenario,
    ScenarioKind,
)
from rfq_bench.core.outcome_space import OUTCOME_SPACE_VERSION
from rfq_bench.report.metrics import is_pareto_efficient


def _common_value_scenario() -> Scenario:
    """A one-issue scenario where BOTH parties prefer 'A' over 'B' (so B is dominated)."""
    issue = Issue(name="quality", values=["A", "B"])
    prefs = lambda role: PartyPreferences(  # noqa: E731
        role=role,
        issues=[IssuePreference(issue="quality", weight=1.0, value_utilities=[1.0, 0.0])],
        batna=0.0,
    )
    return Scenario(
        id="common_value",
        kind=ScenarioKind.single_issue,
        product="widget",
        outcome_space_version=OUTCOME_SPACE_VERSION,
        issues=[issue],
        buyer=prefs("buyer"),
        seller=prefs("seller"),
        deadline_rounds=5,
        seed=0,
    )


def test_pareto_efficiency_is_not_inverted() -> None:
    scn = _common_value_scenario()
    # A is best for both -> Pareto-efficient; B is dominated by A -> not efficient.
    assert is_pareto_efficient(scn, {"quality": "A"}) is True
    assert is_pareto_efficient(scn, {"quality": "B"}) is False


def test_single_issue_outcomes_are_all_efficient(single_issue_scenario) -> None:
    # Single-issue price is fully competitive (buyer wants low, seller high), so every
    # price is on the frontier — none is dominated. (The old inverted code returned
    # False for all of these.)
    for price in single_issue_scenario.issues[0].values:
        assert is_pareto_efficient(single_issue_scenario, {"price": price}) is True
