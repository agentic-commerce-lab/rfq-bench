"""Tests for utility, ideal point, and ZOPA/degeneracy."""

from __future__ import annotations

import pytest

from rfq_bench.core.outcome_space import enumerate_outcomes, is_legal
from rfq_bench.core.utility import (
    best_outcome,
    ideal_utility,
    outcome_utility,
    reservation_utility,
)
from rfq_bench.core.zopa import compute_zopa, is_degenerate


def test_outcome_utility_single_issue(single_issue_scenario) -> None:
    s = single_issue_scenario
    assert outcome_utility(s.buyer, s.issues, {"price": 80}) == pytest.approx(1.0)
    assert outcome_utility(s.buyer, s.issues, {"price": 120}) == pytest.approx(0.0)
    assert outcome_utility(s.seller, s.issues, {"price": 120}) == pytest.approx(1.0)


def test_ideal_and_reservation(single_issue_scenario) -> None:
    s = single_issue_scenario
    assert ideal_utility(s.buyer, s.issues) == pytest.approx(1.0)
    assert reservation_utility(s.buyer) == pytest.approx(0.25)


def test_best_outcome(single_issue_scenario) -> None:
    s = single_issue_scenario
    assert best_outcome(s.buyer, s.issues) == {"price": 80}
    assert best_outcome(s.seller, s.issues) == {"price": 120}


def test_zopa_exists(single_issue_scenario) -> None:
    result = compute_zopa(single_issue_scenario)
    assert result.exists
    # Buyer accepts utility >= 0.25 (price <= 110); seller accepts >= 0.25 (price >= 90).
    prices = sorted(o["price"] for o in result.outcomes)
    assert prices == [90, 100, 110]


def test_no_zopa(no_zopa_scenario) -> None:
    assert not compute_zopa(no_zopa_scenario).exists


def test_degeneracy_flag(single_issue_scenario) -> None:
    assert not is_degenerate(single_issue_scenario, "buyer")


def test_enumerate_and_legality(single_issue_scenario) -> None:
    outcomes = enumerate_outcomes(single_issue_scenario.issues)
    assert len(outcomes) == 5
    assert is_legal(single_issue_scenario.issues, {"price": 100})
    assert not is_legal(single_issue_scenario.issues, {"price": 999})
    assert not is_legal(single_issue_scenario.issues, {"price": 100, "extra": 1})
