"""The NegMAS bindings must faithfully mirror our utilities and outcome space."""

from __future__ import annotations

import pytest

from rfq_bench.core.outcome_space import enumerate_outcomes
from rfq_bench.core.utility import outcome_utility
from rfq_bench.runners.negmas_engine import (
    build_outcome_space,
    build_ufun,
    to_dict,
    to_tuple,
)


def test_outcome_space_size_matches(multi_issue_scenario) -> None:
    os_, neg_issues = build_outcome_space(multi_issue_scenario)
    assert len(neg_issues) == len(multi_issue_scenario.issues)
    assert len(list(os_.enumerate())) == len(enumerate_outcomes(multi_issue_scenario.issues))


def test_tuple_dict_roundtrip(multi_issue_scenario) -> None:
    issues = list(multi_issue_scenario.issues)
    outcome = {"price": 150, "delivery_days": 14, "warranty_months": 12}
    assert to_dict(to_tuple(outcome, issues), issues) == outcome


def test_ufun_matches_core_utility_exactly(multi_issue_scenario) -> None:
    scenario = multi_issue_scenario
    os_, _ = build_outcome_space(scenario)
    ufun = build_ufun(scenario.buyer, scenario, os_)
    issues = list(scenario.issues)
    for outcome in enumerate_outcomes(issues):
        expected = outcome_utility(scenario.buyer, issues, outcome)
        assert float(ufun(to_tuple(outcome, issues))) == pytest.approx(expected)
    # Reserved value is the party's BATNA.
    assert float(ufun.reserved_value) == pytest.approx(scenario.buyer.batna)
