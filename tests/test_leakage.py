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
        "your_walk_away",
        "approach_instruction",
        "history",
        "rounds_left",
        "standing_offer_from_opponent",
    }
    assert set(data) == allowed
    assert data["your_role"] == "buyer"
    assert "seller" not in payload.lower()  # target is buyer: opponent role unnamed

    # The raw utility model is NEVER exposed — no 0..1 utilities, weights, or BATNA.
    assert "utilit" not in payload.lower()
    assert "weight" not in payload.lower()
    assert "batna" not in payload.lower()
    # Only business-terms fields: options in preference order, importance labels.
    for iv in data["issues"]:
        assert set(iv) <= {"name", "unit", "options_best_first", "importance"}
        assert iv["importance"] in {"high", "medium", "low"}  # multi-issue: always present
    assert set(data["your_walk_away"]) == {"break_even_package"}


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


def test_effective_guidance_including_md_files_has_no_economics_vocabulary() -> None:
    """The guidance that actually runs — built-ins plus prompts/{strategies,personas}/*.md
    — goes into every payload, so the md files are held to the same rule."""
    from rfq_bench.agent.personas import load_persona_guidance
    from rfq_bench.agent.prompts import load_strategy_guidance

    effective = {**load_strategy_guidance(), **load_persona_guidance()}
    for name, text in effective.items():
        low = text.lower()
        for word in ("utilit", "weight", "batna"):
            assert word not in low, f"guidance {name!r} contains {word!r}"


def _state_with_history(scenario, role, history):
    return NegotiationState(
        role=role,
        prefs=scenario.preferences(role),
        issues=list(scenario.issues),
        round=2,
        deadline=scenario.deadline_rounds,
        history=history,
    )


def test_history_is_viewer_relative_and_names_no_role(multi_issue_scenario) -> None:
    offer = {"price": 150, "delivery_days": 14, "warranty_months": 12}
    history = [
        {"round": 0, "by": "buyer", "offer": offer, "message": None},
        {"round": 0, "by": "seller", "offer": offer, "message": None},
    ]
    payload = build_user_payload(
        _state_with_history(multi_issue_scenario, "buyer", history), "x", offer
    )
    data = json.loads(payload)
    assert [h[0] for h in data["history"]] == ["you", "them"]
    assert "seller" not in payload.lower()


def test_messages_are_delivered_only_with_the_channel_and_truncated(multi_issue_scenario) -> None:
    from rfq_bench.agent.prompts import MESSAGE_MAX_CHARS

    offer = {"price": 150, "delivery_days": 14, "warranty_months": 12}
    long = "a" * (MESSAGE_MAX_CHARS + 50)
    history = [{"round": 0, "by": "seller", "offer": offer, "message": long}]
    state = _state_with_history(multi_issue_scenario, "buyer", history)

    off = json.loads(build_user_payload(state, "x", offer))
    assert off["history"][0] == ["them", offer]  # --agent llm: no channel, nothing shown

    on = json.loads(build_user_payload(state, "x", offer, message_channel=True))
    assert on["history"][0] == ["them", offer, "a" * MESSAGE_MAX_CHARS]


def test_unknown_persona_falls_back_but_get_persona_raises() -> None:
    assert persona_instruction("nope") == PERSONA_GUIDANCE["neutral"]
    from rfq_bench.agent.personas import get_persona

    try:
        get_persona("nope")
    except KeyError as exc:
        assert "unknown persona" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("get_persona should raise on an unknown name")


def test_single_issue_walk_away_states_its_direction(single_issue_scenario) -> None:
    """A buyer's price limit is a ceiling, a seller's a floor — stated, not implied."""
    for role, expected in (
        ("buyer", {"price": {"at_most": 110}}),
        ("seller", {"price": {"at_least": 90}}),
    ):
        state = NegotiationState(
            role=role,
            prefs=single_issue_scenario.preferences(role),
            issues=list(single_issue_scenario.issues),
            round=0,
            deadline=single_issue_scenario.deadline_rounds,
        )
        payload = json.loads(build_user_payload(state, "x", None))
        assert payload["your_walk_away"] == expected
