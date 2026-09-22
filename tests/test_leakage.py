"""The prompt must never expose the opponent's private economics."""

from __future__ import annotations

import json

from rfq_bench.agent.personas import PERSONA_GUIDANCE, persona_instruction
from rfq_bench.agent.prompts import build_user_payload, strategy_instruction
from rfq_bench.strategies.base import NegotiationState


def test_opponent_private_values_absent_from_payload(multi_issue_scenario) -> None:
    scenario = multi_issue_scenario
    # Target = buyer; the seller's private preferences/BATNA must not leak.
    state = NegotiationState(
        role="buyer",
        prefs=scenario.preferences("buyer"),
        issues=list(scenario.issues),
        round=0,
        deadline=scenario.deadline_rounds,
    )
    payload = build_user_payload(state, strategy_instruction("logrolling"), standing_offer=None)
    data = json.loads(payload)

    # Structural: the payload's top-level keys are exactly the safe allowlist —
    # no opponent block can slip in.
    allowed = {
        "your_role",
        "issues",
        "your_priorities",
        "your_walk_away",
        "round",
        "deadline_rounds",
        "standing_offer_from_opponent",
        "opponent_offer_history",
        "your_offer_history",
        "approach_instruction",
    }
    assert set(data) == allowed
    assert data["your_role"] == "buyer"
    assert "seller" not in payload.lower()  # target is buyer: opponent role unnamed

    # The raw utility model is NEVER exposed — no 0..1 utilities, weights, or BATNA.
    assert "utilit" not in payload.lower()
    assert "weight" not in payload.lower()
    assert "batna" not in payload.lower()
    # Only business-terms fields: options, preference order, importance labels.
    for pv in data["your_priorities"]:
        assert set(pv) == {"issue", "unit", "options", "you_prefer_in_order", "importance"}
        assert pv["importance"] in {"high", "medium", "low"}


def test_payload_exposes_public_issue_space(multi_issue_scenario) -> None:
    scenario = multi_issue_scenario
    state = NegotiationState(
        role="seller",
        prefs=scenario.preferences("seller"),
        issues=list(scenario.issues),
        round=1,
        deadline=scenario.deadline_rounds,
    )
    data = json.loads(
        build_user_payload(state, strategy_instruction("control"), standing_offer={"price": 150})
    )
    names = {i["name"] for i in data["issues"]}
    assert names == {"price", "delivery_days", "warranty_months"}
    assert data["standing_offer_from_opponent"] == {"price": 150}


def test_persona_payload_never_leaks_private_economics(multi_issue_scenario) -> None:
    """A buyer persona is disposition only — it must not widen the leakage surface."""
    scenario = multi_issue_scenario
    state = NegotiationState(
        role="buyer",
        prefs=scenario.preferences("buyer"),
        issues=list(scenario.issues),
        round=0,
        deadline=scenario.deadline_rounds,
    )
    for persona in PERSONA_GUIDANCE:
        payload = build_user_payload(
            state, persona_instruction(persona), standing_offer=None
        ).lower()
        assert "utilit" not in payload
        assert "weight" not in payload
        assert "batna" not in payload
        assert "seller" not in payload  # opponent role stays unnamed for a buyer


def test_persona_guidance_is_pure_disposition() -> None:
    """Guidance strings themselves must not carry economics vocabulary."""
    for text in PERSONA_GUIDANCE.values():
        low = text.lower()
        assert "utilit" not in low
        assert "weight" not in low
        assert "batna" not in low


def test_unknown_persona_falls_back_but_get_persona_raises() -> None:
    assert persona_instruction("nope") == PERSONA_GUIDANCE["neutral"]
    from rfq_bench.agent.personas import get_persona

    try:
        get_persona("nope")
    except KeyError as exc:
        assert "unknown persona" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("get_persona should raise on an unknown name")
