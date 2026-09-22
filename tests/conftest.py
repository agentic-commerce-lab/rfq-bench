"""Shared fixtures: small, hand-checkable scenarios."""

from __future__ import annotations

from pathlib import Path

import pytest

import rfq_bench.cli as cli
from rfq_bench.core.contracts import (
    Issue,
    IssuePreference,
    PartyPreferences,
    Scenario,
    ScenarioKind,
)
from rfq_bench.core.outcome_space import OUTCOME_SPACE_VERSION
from rfq_bench.datasets import load_scenarios


@pytest.fixture(autouse=True)
def _isolate_auto_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep CLI tests hermetic.

    The suite runs in the repo cwd, where a real ``rfq-bench.toml`` may live.
    Auto-discovery would pull it into every ``run`` invocation (e.g. flipping the
    agent to ``llm`` and making live network calls), so disable *implicit*
    discovery here. Explicit ``--config`` still resolves normally; the one test
    that exercises auto-discovery restores the real function.
    """
    monkeypatch.setattr(
        cli, "discover_config", lambda explicit: Path(explicit) if explicit is not None else None
    )


@pytest.fixture
def multi_issue_scenario() -> Scenario:
    """The multi-issue anchor scenario (price, delivery, warranty)."""
    scenarios = {s.id: s for s in load_scenarios("data/scenarios")}
    return scenarios["multi_pdw_a"]


@pytest.fixture
def single_issue_scenario() -> Scenario:
    """Price over five levels. Buyer prefers low, seller prefers high; ZOPA exists."""
    price = Issue(name="price", unit="EUR", values=[80, 90, 100, 110, 120])
    # Buyer: utility 1.0 at 80 down to 0.0 at 120 (linear).
    buyer = PartyPreferences(
        role="buyer",
        issues=[
            IssuePreference(issue="price", weight=1.0, value_utilities=[1.0, 0.75, 0.5, 0.25, 0.0])
        ],
        batna=0.25,  # will not accept worse than utility 0.25 (i.e. price 110)
    )
    # Seller: mirror — utility 0.0 at 80 up to 1.0 at 120.
    seller = PartyPreferences(
        role="seller",
        issues=[
            IssuePreference(issue="price", weight=1.0, value_utilities=[0.0, 0.25, 0.5, 0.75, 1.0])
        ],
        batna=0.25,
    )
    return Scenario(
        id="single_price_a",
        kind=ScenarioKind.single_issue,
        product="widget",
        outcome_space_version=OUTCOME_SPACE_VERSION,
        issues=[price],
        buyer=buyer,
        seller=seller,
        deadline_rounds=10,
        seed=1,
    )


@pytest.fixture
def no_zopa_scenario() -> Scenario:
    """Buyer and seller reservations leave no mutually acceptable outcome."""
    price = Issue(name="price", unit="EUR", values=[80, 100, 120])
    buyer = PartyPreferences(
        role="buyer",
        issues=[IssuePreference(issue="price", weight=1.0, value_utilities=[1.0, 0.5, 0.0])],
        batna=0.9,  # only accepts near price 80
    )
    seller = PartyPreferences(
        role="seller",
        issues=[IssuePreference(issue="price", weight=1.0, value_utilities=[0.0, 0.5, 1.0])],
        batna=0.9,  # only accepts near price 120
    )
    return Scenario(
        id="no_zopa_a",
        kind=ScenarioKind.no_zopa_diagnostic,
        product="widget",
        outcome_space_version=OUTCOME_SPACE_VERSION,
        issues=[price],
        buyer=buyer,
        seller=seller,
        deadline_rounds=10,
        seed=2,
    )
