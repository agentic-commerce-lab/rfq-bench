"""LayaNegotiator: Laya picks its own offer (per-issue choice) + accept/walk gates."""

from __future__ import annotations

import json

from rfq_bench.agent.laya_framing import laya_questions, laya_state
from rfq_bench.agent.laya_negotiator import LayaNegotiator
from rfq_bench.core.utility import outcome_utility, reservation_utility
from rfq_bench.strategies.base import NegotiationState


class FakeLaya:
    def __init__(self, answers: dict, tokens: int = 0) -> None:
        self._answers = answers
        self._tokens = tokens
        self.seen: dict | None = None

    def predict(self, state, questions):  # noqa: ANN001
        self.seen = {"state": state, "questions": questions}
        return {"answers": self._answers, "usage": {"input_tokens": self._tokens}}


def _seller_state(scenario, standing=None, round_=1):
    return NegotiationState(
        role="seller",
        prefs=scenario.preferences("seller"),
        issues=list(scenario.issues),
        round=round_,
        deadline=scenario.deadline_rounds,
        opponent_offers=[standing] if standing else [],
    )


def _answers(*, accept=0.1, reachable=0.9, choices: dict) -> dict:
    a = {"accept": {"noul": accept}, "reachable": {"noul": reachable}}
    for k, v in choices.items():
        a[k] = {"choice": str(v)}
    return a


def test_laya_offer_is_its_own_choice(single_issue_scenario) -> None:
    # Opening move: Laya's chosen price is played as-is.
    move = LayaNegotiator(FakeLaya(_answers(choices={"price": 100}), tokens=7)).act(
        None, _seller_state(single_issue_scenario)
    )
    assert move.action == "offer" and move.outcome == {"price": 100}
    assert move.token_cost == 7 and move.error is False


def test_accept_above_floor_is_honored(single_issue_scenario) -> None:
    move = LayaNegotiator(FakeLaya(_answers(accept=0.95, choices={"price": 100}))).act(
        {"price": 120}, _seller_state(single_issue_scenario, {"price": 120})
    )
    assert move.action == "accept" and move.outcome == {"price": 120}


def test_below_floor_accept_is_not_taken_offers_instead(single_issue_scenario) -> None:
    # price 80 -> seller utility 0 < floor: cannot accept; Laya's own offer is played.
    move = LayaNegotiator(FakeLaya(_answers(accept=0.95, choices={"price": 110}))).act(
        {"price": 80}, _seller_state(single_issue_scenario, {"price": 80})
    )
    assert move.action == "offer" and move.outcome == {"price": 110}


def test_low_reachability_terminates(single_issue_scenario) -> None:
    move = LayaNegotiator(
        FakeLaya(_answers(accept=0.1, reachable=0.05, choices={"price": 100})), walk_threshold=0.25
    ).act({"price": 100}, _seller_state(single_issue_scenario, {"price": 100}))
    assert move.action == "terminate" and move.error is False


def test_below_floor_choice_is_raised_to_floor(single_issue_scenario) -> None:
    # Laya picks price 80 (seller utility 0, below floor) -> raised to the floor.
    move = LayaNegotiator(FakeLaya(_answers(choices={"price": 80}))).act(
        None, _seller_state(single_issue_scenario)
    )
    assert move.action == "offer" and move.adjusted is True
    assert move.intended_outcome == {"price": 80}
    prefs, issues = single_issue_scenario.seller, list(single_issue_scenario.issues)
    assert outcome_utility(prefs, issues, move.outcome) >= reservation_utility(prefs) - 1e-9


def test_multi_issue_package_is_assembled_from_choices(multi_issue_scenario) -> None:
    choices = {"price": 200, "delivery_days": 7, "warranty_months": 24}
    move = LayaNegotiator(FakeLaya(_answers(choices=choices))).act(
        None,
        NegotiationState(
            role="seller",
            prefs=multi_issue_scenario.preferences("seller"),
            issues=list(multi_issue_scenario.issues),
            round=0,
            deadline=multi_issue_scenario.deadline_rounds,
        ),
    )
    # A legal, at/above-floor package for the seller is played exactly as chosen.
    assert move.action == "offer"
    assert set(move.outcome) == {"price", "delivery_days", "warranty_months"}


def test_illegal_choice_is_a_hard_error(single_issue_scenario) -> None:
    move = LayaNegotiator(FakeLaya(_answers(choices={"price": 999}))).act(
        None, _seller_state(single_issue_scenario)
    )
    assert move.error is True and "not a legal option" in (move.error_reason or "")


def test_missing_issue_choice_is_a_hard_error(single_issue_scenario) -> None:
    move = LayaNegotiator(FakeLaya({"accept": {"noul": 0.1}, "reachable": {"noul": 0.9}})).act(
        None, _seller_state(single_issue_scenario)
    )
    assert move.error is True and "no choice for issue" in (move.error_reason or "")


def test_missing_accept_is_a_hard_error(single_issue_scenario) -> None:
    move = LayaNegotiator(FakeLaya({"price": {"choice": "100"}})).act(
        {"price": 100}, _seller_state(single_issue_scenario, {"price": 100})
    )
    assert move.error is True


def test_strategy_instruction_injected_into_state(single_issue_scenario) -> None:
    client = FakeLaya(_answers(choices={"price": 100}))
    LayaNegotiator(client, strategy="boulware", instruction="HOLD FIRM NEAR YOUR BEST").act(
        None, _seller_state(single_issue_scenario)
    )
    assert client.seen is not None
    assert client.seen["state"]["your_negotiation_approach"] == "HOLD FIRM NEAR YOUR BEST"
    # The offer questions include one choice per issue plus the two gates.
    assert set(client.seen["questions"]) == {"accept", "reachable", "price"}


def test_laya_state_has_no_leakage(multi_issue_scenario) -> None:
    state = NegotiationState(
        role="seller",
        prefs=multi_issue_scenario.preferences("seller"),
        issues=list(multi_issue_scenario.issues),
        round=1,
        deadline=multi_issue_scenario.deadline_rounds,
    )
    offer = {"price": 150, "delivery_days": 7, "warranty_months": 24}
    payload = json.dumps(laya_state(state, offer, "hold firm near your best target")).lower()
    assert "utilit" not in payload
    assert "weight" not in payload
    assert "batna" not in payload
    assert "buyer" not in payload
    prefs = multi_issue_scenario.preferences("seller")
    questions = laya_questions(list(multi_issue_scenario.issues), prefs)
    assert {"accept", "reachable", "price", "delivery_days", "warranty_months"} == set(questions)
    # Option valence labels must not leak the raw economics either.
    qjson = json.dumps(questions).lower()
    assert "utilit" not in qjson and "weight" not in qjson and "batna" not in qjson
