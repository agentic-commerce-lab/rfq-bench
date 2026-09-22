"""A2A scoring: both sides from the same traces (seller strategy + buyer persona)."""

from __future__ import annotations

from rfq_bench.agent.llm_client import LLMResponse
from rfq_bench.agent.negotiator import LLMNegotiator
from rfq_bench.core.scoring import score_trace
from rfq_bench.report import build_report
from rfq_bench.runners.offline import EpisodeSpec, run_episode


class _Client:
    def __init__(self, data: dict) -> None:
        self._data = data

    def complete(self, system: str, user: str, tool: dict) -> LLMResponse:
        return LLMResponse(data=dict(self._data), tokens=1, cost_usd=0.001)


def _a2a_trace(scenario, persona, strategy):
    spec = EpisodeSpec(
        scenario=scenario,
        strategy=strategy,
        opponent=persona,
        target_role="seller",
        first_speaker="buyer",
        seed=0,
        mode="a2a",
        persona=persona,
    )
    seller = LLMNegotiator(strategy, _Client({"action": "accept"}), behavior_kind="strategy")
    buyer = LLMNegotiator(
        persona,
        _Client({"action": "offer", "price": 100, "rationale": "mid"}),
        behavior_kind="persona",
    )
    return run_episode(spec, agent_policy=seller, opponent_policy=buyer)


def test_score_trace_role_scores_the_other_side(single_issue_scenario) -> None:
    trace = _a2a_trace(single_issue_scenario, "hardball", "control")
    scn = single_issue_scenario
    seller_q = score_trace(trace, scn).q  # default: target_role = seller
    buyer_q = score_trace(trace, scn, role="buyer").q
    # price 100 -> both sides utility 0.5, batna 0.25, ideal 1.0 -> q = 1/3 each.
    assert seller_q is not None and abs(seller_q - (0.25 / 0.75)) < 1e-6
    assert buyer_q is not None and abs(buyer_q - (0.25 / 0.75)) < 1e-6


def test_seller_report_groups_by_strategy(single_issue_scenario) -> None:
    traces = [
        _a2a_trace(single_issue_scenario, persona, strategy)
        for persona in ("neutral", "hardball")
        for strategy in ("control", "boulware")
    ]
    rep = build_report(traces, {single_issue_scenario.id: single_issue_scenario})
    assert rep.arm_label == "strategy"
    assert {e.strategy for e in rep.effects} == {"control", "boulware"}
    assert rep.control_present  # 'control' strategy is present -> Δ defined


def test_buyer_report_groups_by_persona(single_issue_scenario) -> None:
    traces = [
        _a2a_trace(single_issue_scenario, persona, strategy)
        for persona in ("neutral", "hardball")
        for strategy in ("control", "boulware")
    ]
    rep = build_report(
        traces,
        {single_issue_scenario.id: single_issue_scenario},
        control="neutral",
        role="buyer",
        group_key="opponent",
        arm_label="persona",
    )
    assert rep.arm_label == "persona"
    assert rep.scored_role == "buyer"
    # Personas are the arms now (grouped on the opponent axis).
    assert {e.strategy for e in rep.effects} == {"neutral", "hardball"}
    assert rep.control_present  # 'neutral' persona present -> Δ defined
    text = rep.render()
    assert "scoring the buyer" in text
    assert "persona" in text.splitlines()[3]  # header row uses the persona label
