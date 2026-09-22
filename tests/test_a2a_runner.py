"""A2A self-play: two LLM negotiators in one episode, both sides accounted for.

Uses fake clients (no network). The seller strategy is the scored agent; the
buyer persona is the opponent (the standard opponent axis), so A2A traces flow
through the offline report/dashboard unchanged.
"""

from __future__ import annotations

from rfq_bench.agent.llm_client import LLMResponse
from rfq_bench.agent.negotiator import LLMNegotiator
from rfq_bench.runners.offline import EpisodeSpec, build_a2a_matrix, run_episode


class ScriptedClient:
    """Returns a fixed move each call, with configurable token/cost accounting."""

    def __init__(self, data: dict, *, tokens: int = 0, cost_usd: float | None = None) -> None:
        self._data = data
        self._tokens = tokens
        self._cost = cost_usd

    def complete(self, system: str, user: str, tool: dict) -> LLMResponse:
        return LLMResponse(data=dict(self._data), tokens=self._tokens, cost_usd=self._cost)


def _a2a_spec(scenario, *, first_speaker="buyer") -> EpisodeSpec:
    # Seller strategy = scored agent; buyer persona = opponent.
    return EpisodeSpec(
        scenario=scenario,
        strategy="boulware",
        opponent="hardball",
        target_role="seller",
        first_speaker=first_speaker,
        seed=0,
        mode="a2a",
        persona="hardball",
    )


def test_two_llm_sides_reach_agreement_and_account_both(single_issue_scenario) -> None:
    # Buyer (opponent) opens with a mid price good for both; seller (agent) accepts.
    buyer = LLMNegotiator(
        "hardball",
        ScriptedClient(
            {"action": "offer", "price": 100, "rationale": "mid"}, tokens=30, cost_usd=0.02
        ),
        behavior_kind="persona",
    )
    seller = LLMNegotiator(
        "boulware",
        ScriptedClient({"action": "accept"}, tokens=20, cost_usd=0.01),
        behavior_kind="strategy",
    )

    trace = run_episode(
        _a2a_spec(single_issue_scenario, first_speaker="buyer"),
        agent_policy=seller,  # target_role = seller
        opponent_policy=buyer,
    )

    assert trace.mode == "a2a"
    assert trace.persona == "hardball"
    assert trace.strategy == "boulware"
    assert trace.opponent == "hardball"  # persona sits on the opponent axis
    assert trace.target_role == "seller"
    assert trace.outcome_kind == "agreement"
    assert trace.agreement == {"price": 100}
    # Both sides are LLMs, so both sides' spend is recorded per role and summed.
    assert trace.token_cost == 50
    assert trace.token_cost_by_role == {"seller": 20, "buyer": 30}
    assert trace.cost_usd is not None and abs(trace.cost_usd - 0.03) < 1e-9
    assert set(trace.cost_usd_by_role) == {"buyer", "seller"}
    assert trace.errored is False


def test_opponent_side_error_aborts_episode(single_issue_scenario) -> None:
    # A usable seller, but the buyer (opponent) returns an unusable reply -> excluded.
    seller = LLMNegotiator(
        "boulware",
        ScriptedClient({"action": "offer", "price": 100, "rationale": "mid"}),
        behavior_kind="strategy",
    )
    buyer = LLMNegotiator(
        "hardball",
        ScriptedClient({"action": "offer"}),  # no package -> hard error
        behavior_kind="persona",
    )
    trace = run_episode(
        _a2a_spec(single_issue_scenario),
        agent_policy=seller,
        opponent_policy=buyer,
    )
    assert trace.errored is True
    assert trace.error_reason is not None


def test_a2a_matrix_maps_persona_to_opponent_and_seller_target(multi_issue_scenario) -> None:
    specs = list(
        build_a2a_matrix(
            [multi_issue_scenario],
            personas=["neutral", "hardball"],
            strategies=["control", "logrolling"],
            seeds=[0, 1],
            first_speakers=("buyer", "seller"),
        )
    )
    # 2 personas x 2 strategies x 2 first-speakers x 2 seeds = 16 (multi-issue keeps logrolling).
    assert len(specs) == 16
    assert all(s.target_role == "seller" for s in specs)
    assert all(s.mode == "a2a" for s in specs)
    assert {s.opponent for s in specs} == {"neutral", "hardball"}  # personas on opponent axis
    assert {s.strategy for s in specs} == {"control", "logrolling"}
    assert all(s.opponent == s.persona for s in specs)


def test_a2a_matrix_skips_logrolling_on_single_issue(single_issue_scenario) -> None:
    specs = list(
        build_a2a_matrix(
            [single_issue_scenario],
            personas=["neutral"],
            strategies=["control", "logrolling"],
            seeds=[0],
            first_speakers=("buyer",),
        )
    )
    assert {s.strategy for s in specs} == {"control"}  # logrolling skipped
