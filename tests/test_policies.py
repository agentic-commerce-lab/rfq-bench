"""Concession-schedule and outcome-selection behavior."""

from __future__ import annotations

from rfq_bench.strategies.base import NegotiationState, select_outcome
from rfq_bench.strategies.registry import STRATEGIES, get_strategy


def _state(scenario, role, round_, deadline=12):
    return NegotiationState(
        role=role,
        prefs=scenario.preferences(role),
        issues=list(scenario.issues),
        round=round_,
        deadline=deadline,
    )


def test_target_decreases_over_time(single_issue_scenario) -> None:
    pol = get_strategy("linear")
    early = pol.target_utility(_state(single_issue_scenario, "buyer", 0))  # type: ignore[attr-defined]
    late = pol.target_utility(_state(single_issue_scenario, "buyer", 11))  # type: ignore[attr-defined]
    assert early > late


def test_tit_for_tat_target_is_idempotent_per_round(single_issue_scenario) -> None:
    # Reciprocation must build on the previous *target*, memoized per round, so
    # repeated calls within a turn (propose + accepts) return the same target and
    # never double-advance the concession.
    pol = get_strategy("tit_for_tat")
    prev = None
    for rnd in range(4):
        offers = [{"price": p} for p in (120, 110, 100)[:rnd]]  # opponent concedes over time
        st = NegotiationState(
            role="seller",
            prefs=single_issue_scenario.preferences("seller"),
            issues=list(single_issue_scenario.issues),
            round=rnd,
            deadline=12,
            opponent_offers=offers,
        )
        t1 = pol.target_utility(st)  # type: ignore[attr-defined]
        t2 = pol.target_utility(st)  # type: ignore[attr-defined]
        assert t1 == t2  # idempotent within the same round
        prev = t1
    assert prev is not None


def test_boulware_holds_higher_than_conceder_midway(single_issue_scenario) -> None:
    b = get_strategy("boulware")
    c = get_strategy("conceder")
    mid = _state(single_issue_scenario, "buyer", 6)
    assert b.target_utility(mid) > c.target_utility(mid)  # type: ignore[attr-defined]


def test_all_strategies_constructible() -> None:
    for name in STRATEGIES:
        assert get_strategy(name).name == name or name == "tit_for_tat"


def test_integrative_tiebreak_prefers_opponent_offer(multi_issue_scenario) -> None:
    issues = list(multi_issue_scenario.issues)
    prefs = multi_issue_scenario.preferences("buyer")
    opponent_last = {"price": 200, "delivery_days": 30, "warranty_months": 6}
    # With a very low target both branches have many feasible outcomes; the
    # integrative branch should shift selection toward the opponent's offer.
    plain = select_outcome(prefs, issues, target=0.0, integrative=False)
    integ = select_outcome(prefs, issues, target=0.0, integrative=True, opponent_last=opponent_last)
    matches_plain = sum(1 for k, v in opponent_last.items() if plain[k] == v)
    matches_integ = sum(1 for k, v in opponent_last.items() if integ[k] == v)
    assert matches_integ >= matches_plain


def test_select_outcome_never_below_target_when_feasible(single_issue_scenario) -> None:
    from rfq_bench.core.utility import outcome_utility

    prefs = single_issue_scenario.preferences("buyer")
    issues = list(single_issue_scenario.issues)
    out = select_outcome(prefs, issues, target=0.5)
    assert outcome_utility(prefs, issues, out) >= 0.5 - 1e-9
