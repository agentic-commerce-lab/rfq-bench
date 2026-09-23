"""A2A message channel, append-only history, and prompt-cache accounting.

Fake clients only (no network): each side records the tool and payload it was
sent, so the tests check what the *other* model actually received.
"""

from __future__ import annotations

import json

from rfq_bench.agent.llm_client import LLMResponse
from rfq_bench.agent.negotiator import LLMNegotiator
from rfq_bench.agent.prompts import MESSAGE_MAX_CHARS
from rfq_bench.report.usage import usage_summary
from rfq_bench.runners.offline import EpisodeSpec, run_episode


class RecordingClient:
    """Plays back queued move dicts; records every (user payload, tool) it receives."""

    def __init__(self, moves: list[dict], *, prompt: int = 100, cached: int = 60) -> None:
        self._moves = list(moves)
        self.calls: list[tuple[dict, dict]] = []
        self._prompt, self._cached = prompt, cached

    def complete(self, system: str, user: str, tool: dict) -> LLMResponse:
        self.calls.append((json.loads(user), tool))
        move = self._moves[min(len(self.calls) - 1, len(self._moves) - 1)]
        return LLMResponse(
            data=dict(move),
            tokens=self._prompt + 10,
            prompt_tokens=self._prompt,
            cached_tokens=self._cached,
        )


def _spec(scenario, *, mode="a2a") -> EpisodeSpec:
    return EpisodeSpec(
        scenario=scenario,
        strategy="control",
        opponent="neutral",
        target_role="seller",
        first_speaker="buyer",
        seed=0,
        mode=mode,
        persona="neutral" if mode == "a2a" else None,
    )


def _pair(buyer_moves, seller_moves, *, channel=True):
    buyer_client, seller_client = RecordingClient(buyer_moves), RecordingClient(seller_moves)
    buyer = LLMNegotiator("neutral", buyer_client, behavior_kind="persona", message_channel=channel)
    seller = LLMNegotiator("control", seller_client, message_channel=channel)
    return buyer, seller, buyer_client, seller_client


def test_message_reaches_the_other_side_and_is_recorded(single_issue_scenario) -> None:
    buyer, seller, buyer_c, seller_c = _pair(
        [
            {
                "action": "offer",
                "price": 90,
                "rationale": "open low",
                "message": "Budget is tight.",
            },
            {"action": "accept", "rationale": "fair", "message": ""},
        ],
        [{"action": "offer", "price": 100, "rationale": "counter", "message": "Can do 100."}],
    )
    trace = run_episode(_spec(single_issue_scenario), agent_policy=seller, opponent_policy=buyer)

    # The seller's tool offers a message field; its history shows the buyer's message
    # as the opponent's, with the private rationale nowhere in sight.
    seller_payload, seller_tool = seller_c.calls[0]
    assert "message" in seller_tool["function"]["parameters"]["properties"]
    assert seller_payload["history"] == [["them", {"price": 90}, "Budget is tight."]]
    assert "open low" not in json.dumps(seller_payload)

    # The buyer then sees both moves, oldest first, from its own point of view.
    buyer_payload, _ = buyer_c.calls[1]
    assert [h[0] for h in buyer_payload["history"]] == ["you", "them"]
    assert buyer_payload["history"][1] == ["them", {"price": 100}, "Can do 100."]
    assert buyer_payload["standing_offer_from_opponent"] == {"price": 100}

    # The trace keeps each move's public message; an empty one is no message.
    assert [s.message for s in trace.steps] == ["Budget is tight.", "Can do 100.", None]
    assert trace.outcome_kind == "agreement" and trace.agreement == {"price": 100}


def test_long_message_is_cut_on_delivery_but_kept_in_full(single_issue_scenario) -> None:
    long = "x" * (MESSAGE_MAX_CHARS + 25)
    buyer, seller, _, seller_c = _pair(
        [{"action": "offer", "price": 90, "rationale": "r", "message": long}],
        [{"action": "terminate", "rationale": "r", "message": ""}],
    )
    trace = run_episode(_spec(single_issue_scenario), agent_policy=seller, opponent_policy=buyer)
    delivered = seller_c.calls[0][0]["history"][0][2]
    assert delivered == "x" * MESSAGE_MAX_CHARS
    assert trace.steps[0].message == long and trace.steps[0].message_truncated is True


def test_without_the_channel_there_is_no_message_field_and_none_is_kept(
    single_issue_scenario,
) -> None:
    # --agent llm: the model may still return a message, but it goes nowhere.
    buyer, seller, _, seller_c = _pair(
        [{"action": "offer", "price": 90, "rationale": "r", "message": "hello"}],
        [{"action": "terminate", "rationale": "r"}],
        channel=False,
    )
    trace = run_episode(
        _spec(single_issue_scenario, mode="offline"), agent_policy=seller, opponent_policy=buyer
    )
    payload, tool = seller_c.calls[0]
    assert "message" not in tool["function"]["parameters"]["properties"]
    assert payload["history"][0] == ["them", {"price": 90}]
    assert trace.steps[0].message is None


def test_prompt_and_cached_tokens_are_recorded_per_role(single_issue_scenario) -> None:
    buyer, seller, _, _ = _pair(
        [{"action": "offer", "price": 90, "rationale": "r", "message": "hi"}],
        [{"action": "terminate", "rationale": "r", "message": ""}],
    )
    trace = run_episode(_spec(single_issue_scenario), agent_policy=seller, opponent_policy=buyer)
    assert trace.prompt_tokens_by_role == {"seller": 100, "buyer": 100}
    assert trace.cached_tokens_by_role == {"seller": 60, "buyer": 60}

    lines = usage_summary([trace])
    assert any("Prompt cache: 60%" in line for line in lines)
    assert any("buyer" in line and "sent on 100%" in line for line in lines)


def test_usage_summary_is_silent_for_traces_without_the_data(single_issue_scenario) -> None:
    spec = EpisodeSpec(
        scenario=single_issue_scenario,
        strategy="control",
        opponent="hardliner",
        target_role="seller",
        first_speaker="buyer",
        seed=0,
    )
    trace = run_episode(spec)  # scripted both sides: no cache counts, no messages
    assert usage_summary([trace]) == []


def test_message_field_says_only_the_offer_fields_bind(single_issue_scenario) -> None:
    from rfq_bench.agent.prompts import build_move_tool

    tool = build_move_tool(list(single_issue_scenario.issues), message_channel=True)
    desc = tool["function"]["parameters"]["properties"]["message"]["description"]
    assert "bind" in desc and "match" in desc


def test_message_on_a_corrected_move_is_withheld_but_kept(single_issue_scenario) -> None:
    # Seller's walk-away is 90; offering 80 fires the floor constraint (played 90).
    buyer, seller, buyer_c, _ = _pair(
        [
            {"action": "offer", "price": 80, "rationale": "r", "message": "Start low."},
            {"action": "terminate", "rationale": "r", "message": ""},
        ],
        [{"action": "offer", "price": 80, "rationale": "r", "message": "80 works for me."}],
    )
    trace = run_episode(_spec(single_issue_scenario), agent_policy=seller, opponent_policy=buyer)

    seller_step = trace.steps[1]
    assert seller_step.adjusted and seller_step.adjust_kind == "floor_offer"
    assert seller_step.outcome == {"price": 90}  # the corrected, binding offer
    assert seller_step.message == "80 works for me." and seller_step.message_withheld is True
    # The buyer saw the corrected offer, and no message quoting the invalid 80.
    assert buyer_c.calls[1][0]["history"][1] == ["them", {"price": 90}]

    lines = usage_summary([trace])
    assert any("seller  1 of" in line and "floor_offer 1" in line for line in lines)
    assert any("messages withheld from the opponent: 1" in line for line in lines)

    from rfq_bench.report.dashboard import build_payload

    ep = build_payload([trace], {single_issue_scenario.id: single_issue_scenario})["episodes"][0]
    assert ep["policy_fires"] == {"buyer": 0, "seller": 1}
    assert ep["steps"][1]["message_withheld"] is True
    assert ep["steps"][1]["adjust_kind"] == "floor_offer"


def test_the_author_is_told_its_move_was_corrected_the_opponent_is_not(
    single_issue_scenario,
) -> None:
    # Seller's walk-away is 90. It offers 80 (corrected to 90), the buyer counters,
    # then the seller moves again — and must see why its offer reads 90.
    buyer, seller, buyer_c, seller_c = _pair(
        [
            {"action": "offer", "price": 80, "rationale": "r", "message": "Start low."},
            {"action": "offer", "price": 85, "rationale": "r", "message": "Meet me at 85?"},
            {"action": "terminate", "rationale": "r", "message": ""},
        ],
        [
            {"action": "offer", "price": 80, "rationale": "r", "message": "80 works."},
            {"action": "terminate", "rationale": "r", "message": ""},
        ],
    )
    run_episode(_spec(single_issue_scenario), agent_policy=seller, opponent_policy=buyer)

    own = seller_c.calls[1][0]["history"][1]
    assert own[:2] == ["you", {"price": 90}]
    assert own[2] == (
        "policy: you offered price=80, below your minimum of 90; played price=90; "
        "your message was not delivered."
    )
    # The buyer saw only the corrected offer: no note, no message.
    theirs = buyer_c.calls[1][0]["history"][1]
    assert theirs == ["them", {"price": 90}]
    assert "policy" not in json.dumps(buyer_c.calls[1][0])


def test_the_policy_note_also_reaches_the_author_without_a_message_channel(
    single_issue_scenario,
) -> None:
    buyer, seller, _, seller_c = _pair(
        [
            {"action": "offer", "price": 80, "rationale": "r"},
            {"action": "offer", "price": 85, "rationale": "r"},
            {"action": "terminate", "rationale": "r"},
        ],
        [
            {"action": "offer", "price": 80, "rationale": "r"},
            {"action": "terminate", "rationale": "r"},
        ],
        channel=False,
    )
    run_episode(
        _spec(single_issue_scenario, mode="offline"), agent_policy=seller, opponent_policy=buyer
    )
    note = seller_c.calls[1][0]["history"][1][2]
    assert note == "policy: you offered price=80, below your minimum of 90; played price=90."


def test_policy_note_wording_for_each_kind(single_issue_scenario) -> None:
    from rfq_bench.agent.prompts import build_user_payload
    from rfq_bench.strategies.base import NegotiationState

    def note(kind, action, sent, played):
        history = [
            {
                "round": 0,
                "by": "seller",
                "offer": played,
                "message": None,
                "policy": {"kind": kind, "action": action, "sent": sent, "withheld": False},
            }
        ]
        state = NegotiationState(
            role="seller",
            prefs=single_issue_scenario.seller,
            issues=list(single_issue_scenario.issues),
            round=1,
            deadline=10,
            history=history,
        )
        return json.loads(build_user_payload(state, "x", None))["history"][0][2]

    assert note("snap", "offer", {"price": 97}, {"price": 100}) == (
        "policy: price=97 is not an allowed option; played the nearest, price=100."
    )
    assert note("floor_accept", "accept", {"price": 80}, {"price": 90}) == (
        "policy: you tried to accept price=80, below your minimum of 90; countered price=90."
    )
