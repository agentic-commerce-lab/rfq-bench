"""LLMNegotiator: snap-to-grid, shop reservation floor, and hard errors.

No scripted-strategy value is ever substituted. The model's decision stands,
adjusted only by (1) snapping an off-grid value to the nearest legal tier and
(2) clamping a below-floor quote up to the shop's reservation floor. An unusable
reply is a hard error that aborts the episode.
"""

from __future__ import annotations

from rfq_bench.agent.llm_client import LLMResponse
from rfq_bench.agent.negotiator import LLMNegotiator
from rfq_bench.core.utility import outcome_utility, reservation_utility
from rfq_bench.strategies.base import NegotiationState


class StubClient:
    def __init__(self, data: dict) -> None:
        self.data = data

    def complete(self, system: str, user: str, tool: dict) -> LLMResponse:
        return LLMResponse(data=self.data, tokens=42)


def _state(scenario, role="buyer", round_=1):
    return NegotiationState(
        role=role,
        prefs=scenario.preferences(role),
        issues=list(scenario.issues),
        round=round_,
        deadline=scenario.deadline_rounds,
    )


def _act(scenario, data, *, standing=None, role="buyer"):
    return LLMNegotiator("linear", StubClient(data)).act(standing, _state(scenario, role))


def test_valid_offer_is_played_as_is(multi_issue_scenario) -> None:
    outcome = {"price": 150, "delivery_days": 7, "warranty_months": 24}
    move = _act(multi_issue_scenario, {"action": "offer", "outcome": outcome, "rationale": "x"})
    assert move.action == "offer"
    assert move.outcome == outcome
    assert move.adjusted is False and move.error is False
    assert move.token_cost == 42


def test_off_grid_value_is_snapped(multi_issue_scenario) -> None:
    raw = {"price": 190, "delivery_days": 7, "warranty_months": 24}
    move = _act(multi_issue_scenario, {"action": "offer", "outcome": raw, "rationale": "near-200"})
    assert move.action == "offer"
    assert move.outcome["price"] == 200  # nearest legal tier to 190 in {100,150,200}
    assert move.adjusted is True and move.error is False
    assert move.intended_action == "offer" and move.intended_outcome == raw
    assert "snap" in (move.adjust_reason or "").lower()
    assert move.rationale == "near-200"  # the model's words survive the adjustment


def test_type_coercion_is_not_a_snap(multi_issue_scenario) -> None:
    raw = {"price": "150", "delivery_days": 7, "warranty_months": 24}
    move = _act(multi_issue_scenario, {"action": "offer", "outcome": raw})
    assert move.outcome["price"] == 150
    assert move.adjusted is False  # exact legal value, only string-typed


def test_below_floor_offer_is_clamped_to_floor(multi_issue_scenario) -> None:
    bad = {"price": 200, "delivery_days": 30, "warranty_months": 6}  # buyer utility 0
    move = _act(multi_issue_scenario, {"action": "offer", "outcome": bad})
    assert move.action == "offer" and move.adjusted is True
    assert move.intended_action == "offer" and move.intended_outcome == bad
    assert "floor" in (move.adjust_reason or "").lower()
    prefs = multi_issue_scenario.buyer
    played = outcome_utility(prefs, list(multi_issue_scenario.issues), move.outcome)
    assert played >= reservation_utility(prefs) - 1e-9  # respects the floor


def test_below_floor_accept_is_blocked_and_counters(multi_issue_scenario) -> None:
    bad = {"price": 200, "delivery_days": 30, "warranty_months": 6}
    move = _act(multi_issue_scenario, {"action": "accept", "outcome": None}, standing=bad)
    assert move.action == "offer"  # cannot accept below floor -> counter at floor
    assert move.adjusted is True
    assert move.intended_action == "accept" and move.intended_outcome == bad


def test_above_floor_accept_is_honored(single_issue_scenario) -> None:
    standing = {"price": 100}  # buyer utility 0.5 >= BATNA 0.25
    move = _act(single_issue_scenario, {"action": "accept", "outcome": None}, standing=standing)
    assert move.action == "accept" and move.outcome == standing
    assert move.adjusted is False


def test_bad_but_legal_offer_above_floor_is_played(single_issue_scenario) -> None:
    # A weak buyer offer at the floor (price 110, utility 0.25) is allowed, not blocked.
    move = _act(single_issue_scenario, {"action": "offer", "outcome": {"price": 110}})
    assert move.action == "offer" and move.outcome == {"price": 110}
    assert move.adjusted is False and move.error is False


def test_terminate_passes_through(multi_issue_scenario) -> None:
    move = _act(multi_issue_scenario, {"action": "terminate", "outcome": None, "rationale": "no"})
    assert move.action == "terminate" and move.error is False


def test_missing_issue_is_a_hard_error(multi_issue_scenario) -> None:
    move = _act(multi_issue_scenario, {"action": "offer", "outcome": {"price": 100}})
    assert move.error is True and move.action == "terminate"
    assert "missing" in (move.error_reason or "").lower()


def test_non_numeric_value_is_a_hard_error(multi_issue_scenario) -> None:
    raw = {"price": "cheap", "delivery_days": 7, "warranty_months": 24}
    move = _act(multi_issue_scenario, {"action": "offer", "outcome": raw})
    assert move.error is True and move.action == "terminate"


def test_no_outcome_is_a_hard_error(multi_issue_scenario) -> None:
    move = _act(multi_issue_scenario, {"action": "offer"})
    assert move.error is True


def test_rationale_and_raw_response_captured(multi_issue_scenario) -> None:
    outcome = {"price": 100, "delivery_days": 7, "warranty_months": 24}
    client = StubClient({"action": "offer", "outcome": outcome, "rationale": "anchor"})
    move = LLMNegotiator("anchoring", client).act(None, _state(multi_issue_scenario))
    assert move.rationale == "anchor"
    assert move.adjusted is False


def test_tool_schema_is_flat_plain_typed_and_required(multi_issue_scenario) -> None:
    from rfq_bench.agent.prompts import build_move_tool

    issues = {"price", "delivery_days", "warranty_months"}
    tool = build_move_tool(list(multi_issue_scenario.issues))  # default: non-strict
    fn = tool["function"]
    assert "strict" not in fn
    params = fn["parameters"]
    # Issues are flat top-level fields, all required, no nested "outcome".
    assert issues <= set(params["properties"])
    assert {"action", "rationale", *issues} <= set(params["required"])
    assert "outcome" not in params["properties"]
    # Value fields are plainly typed with options in the description — NOT enum-
    # constrained (some providers return empty args for enum function params).
    for name in issues:
        field = params["properties"][name]
        assert "enum" not in field
        assert field["type"] == "integer"
        assert "choose one of" in field["description"]
    # Only `action` stays a (small string) enum.
    assert params["properties"]["action"]["enum"] == ["offer", "accept", "terminate"]
    # Strict build opts into function.strict + additionalProperties:false.
    strict = build_move_tool(list(multi_issue_scenario.issues), strict=True)["function"]
    assert strict["strict"] is True
    assert strict["parameters"]["additionalProperties"] is False


def test_flat_top_level_package_is_read(multi_issue_scenario) -> None:
    # A real (flat) tool reply: issues at the top level, no "outcome" wrapper.
    data = {
        "action": "offer",
        "price": 150,
        "delivery_days": 7,
        "warranty_months": 24,
        "rationale": "anchor",
    }
    move = _act(multi_issue_scenario, data)
    assert move.action == "offer" and move.error is False
    assert move.outcome == {"price": 150, "delivery_days": 7, "warranty_months": 24}


class _FinishClient:
    def __init__(self, finish_reason: str) -> None:
        self._fr = finish_reason

    def complete(self, system: str, user: str, tool: dict) -> LLMResponse:
        return LLMResponse(data={}, tokens=10, finish_reason=self._fr)


def test_truncation_is_labeled_distinctly(multi_issue_scenario) -> None:
    move = LLMNegotiator("linear", _FinishClient("length")).act(None, _state(multi_issue_scenario))
    assert move.error is True
    assert "truncat" in (move.error_reason or "").lower()
    assert "max_tokens" in (move.error_reason or "").lower()


def test_genuine_error_is_not_labeled_truncation(multi_issue_scenario) -> None:
    move = LLMNegotiator("linear", _FinishClient("stop")).act(None, _state(multi_issue_scenario))
    assert move.error is True
    assert "truncat" not in (move.error_reason or "").lower()
