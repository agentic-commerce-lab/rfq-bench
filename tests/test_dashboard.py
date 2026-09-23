"""The dashboard payload must agree with the scoring kernel, exactly.

The browser recomputes only cheap aggregates over the frozen per-episode ``q``
values in the payload, so the invariant we protect here is: the ``q`` the payload
carries is exactly ``score_trace``'s, and macro-averaging it per strategy
reproduces ``strategy_score`` / ``build_report``.
"""

from __future__ import annotations

import json

import pytest

from rfq_bench.agent.llm_client import LLMResponse
from rfq_bench.agent.negotiator import LLMNegotiator
from rfq_bench.core.contracts import Scenario
from rfq_bench.core.scoring import score_trace, strategy_score
from rfq_bench.datasets import load_scenarios
from rfq_bench.report import build_dashboard, build_payload
from rfq_bench.report.aggregate import build_report
from rfq_bench.runners.offline import EpisodeSpec, build_matrix, run_episode


@pytest.fixture(scope="module")
def scenarios() -> dict[str, Scenario]:
    return {s.id: s for s in load_scenarios("data/scenarios")}


@pytest.fixture(scope="module")
def traces(scenarios: dict[str, Scenario]) -> list:
    # A small but representative slice of the matrix, deterministic (scripted).
    specs = list(
        build_matrix(
            list(scenarios.values()),
            strategies=["control", "boulware", "conceder"],
            opponents=["hardliner", "fast_conceder"],
            seeds=[0, 1],
        )
    )
    return [run_episode(spec) for spec in specs]


def test_payload_q_matches_score_trace(traces: list, scenarios: dict[str, Scenario]) -> None:
    payload = build_payload(traces, scenarios)
    assert len(payload["episodes"]) == len(traces)
    for ep, trace in zip(payload["episodes"], traces, strict=True):
        scored = score_trace(trace, scenarios[trace.scenario_id])
        assert ep["q"] == scored.q
        assert ep["degenerate"] == scored.degenerate


def test_payload_macro_average_matches_strategy_score(
    traces: list, scenarios: dict[str, Scenario]
) -> None:
    payload = build_payload(traces, scenarios)
    episodes = [score_trace(t, scenarios[t.scenario_id]) for t in traces]
    for strat in payload["meta"]["strategies"]:
        expected = strategy_score([e for e in episodes if e.strategy == strat])
        qs = [
            e["q"]
            for e in payload["episodes"]
            if e["strategy"] == strat and e["q"] is not None and not e["degenerate"]
        ]
        got = 100.0 * (sum(qs) / len(qs))
        assert got == pytest.approx(expected)


def test_payload_meta_dimensions(traces: list, scenarios: dict[str, Scenario]) -> None:
    payload = build_payload(traces, scenarios)
    meta = payload["meta"]
    assert meta["strategies"] == ["boulware", "conceder", "control"]
    assert set(meta["opponents"]) == {"hardliner", "fast_conceder"}
    assert meta["n_episodes"] == len(traces)
    # Degenerate count is consistent with the per-episode flags.
    assert meta["n_degenerate"] == sum(1 for e in payload["episodes"] if e["degenerate"])
    report = build_report(traces, scenarios)
    assert meta["n_no_zopa"] == sum(1 for e in payload["episodes"] if e["no_zopa"])
    assert meta["n_no_zopa"] == report.n_no_zopa > 0  # the anchor set has a no-ZOPA case
    assert meta["n_episodes"] - meta["n_degenerate"] - meta["n_no_zopa"] == report.n_episodes


def test_scenario_metadata_has_zopa_and_bounds(
    traces: list, scenarios: dict[str, Scenario]
) -> None:
    payload = build_payload(traces, scenarios)
    for sid, sc in payload["scenarios"].items():
        assert sc["issues"], f"{sid} has no issues"
        for role in ("buyer", "seller"):
            p = sc["parties"][role]
            assert p["worst"] <= p["batna"] <= 1.0
            assert p["batna"] <= p["ideal"] <= 1.0
        assert isinstance(sc["zopa_exists"], bool)


def test_replay_steps_carry_both_party_utilities(
    traces: list, scenarios: dict[str, Scenario]
) -> None:
    payload = build_payload(traces, scenarios)
    ep = payload["episodes"][0]
    assert ep["steps"]
    for step in ep["steps"]:
        if step["outcome"] is not None:
            assert set(step["utilities"]) == {"buyer", "seller"}


class _RationaleClient:
    """Stub LLM client: always a safe, legal mid-price offer with a rationale."""

    def complete(self, system: str, user: str, tool: dict) -> LLMResponse:
        return LLMResponse(
            data={"action": "offer", "outcome": {"price": 100}, "rationale": "meet in the middle"},
            tokens=7,
        )


def _llm_trace(scenarios: dict[str, Scenario]):
    scenario = scenarios["single_price_a"]
    spec = EpisodeSpec(
        scenario=scenario,
        strategy="control",
        opponent="reciprocal",
        target_role="buyer",
        first_speaker="buyer",
        seed=0,
    )
    policy = LLMNegotiator("control", _RationaleClient())
    return run_episode(spec, agent_policy=policy)


def test_llm_rationales_reach_payload(scenarios: dict[str, Scenario]) -> None:
    trace = _llm_trace(scenarios)
    # The agent (buyer) side must carry the model's message on its steps.
    agent_steps = [s for s in trace.steps if s.party == "buyer"]
    assert agent_steps and all(s.rationale == "meet in the middle" for s in agent_steps)

    payload = build_payload([trace], scenarios)
    assert payload["meta"]["has_messages"] is True
    ep = payload["episodes"][0]
    buyer_steps = [s for s in ep["steps"] if s["party"] == "buyer"]
    assert buyer_steps and all(s["rationale"] == "meet in the middle" for s in buyer_steps)
    # Opponent side is scripted -> no message.
    seller_steps = [s for s in ep["steps"] if s["party"] == "seller"]
    assert seller_steps and all(s["rationale"] is None for s in seller_steps)


class _OffGridClient:
    """Stub: proposes an off-grid price that snaps to a legal tier (not an error)."""

    def complete(self, system: str, user: str, tool: dict) -> LLMResponse:
        return LLMResponse(
            data={"action": "offer", "outcome": {"price": 95}, "rationale": "just under 100"},
            tokens=5,
        )


def test_snap_adjustment_reaches_payload(scenarios: dict[str, Scenario]) -> None:
    scenario = scenarios["single_price_a"]
    spec = EpisodeSpec(
        scenario=scenario,
        strategy="linear",
        opponent="reciprocal",
        target_role="buyer",
        first_speaker="buyer",
        seed=0,
    )
    trace = run_episode(spec, agent_policy=LLMNegotiator("linear", _OffGridClient()))
    assert trace.errored is False

    payload = build_payload([trace], scenarios)
    assert payload["meta"]["has_adjustments"] is True
    adj = [s for s in payload["episodes"][0]["steps"] if s["adjusted"]]
    assert adj, "expected at least one adjusted step"
    s = adj[0]
    assert s["outcome"]["price"] == 90  # 95 -> nearest legal tier (ties to lower)
    assert "snap" in s["adjust_reason"].lower()
    assert s["intended_outcome"] == {"price": 95}
    assert s["rationale"] == "just under 100"  # the model's words survive
    assert s["error"] is False

    html = build_dashboard([trace], scenarios)
    assert "adjusted" in html


class _NoOutcomeClient:
    """Stub: returns no usable outcome -> a hard error that aborts the episode."""

    def complete(self, system: str, user: str, tool: dict) -> LLMResponse:
        return LLMResponse(data={"action": "offer", "rationale": "oops"}, tokens=3)


def test_hard_error_excludes_episode(scenarios: dict[str, Scenario]) -> None:
    from rfq_bench.core.scoring import score_trace

    scenario = scenarios["single_price_a"]
    spec = EpisodeSpec(
        scenario=scenario,
        strategy="linear",
        opponent="reciprocal",
        target_role="buyer",
        first_speaker="buyer",
        seed=0,
    )
    trace = run_episode(spec, agent_policy=LLMNegotiator("linear", _NoOutcomeClient()))
    assert trace.errored is True
    assert score_trace(trace, scenario).scorable is False  # excluded from scoring

    payload = build_payload([trace], scenarios)
    assert payload["meta"]["has_errors"] is True
    assert payload["meta"]["n_errored"] == 1
    assert payload["episodes"][0]["errored"] is True

    html = build_dashboard([trace], scenarios)
    assert "EXCLUDED" in html


class _CostClient:
    """Stub LLM client that reports real USD spend, like OpenRouter's usage.cost."""

    def complete(self, system: str, user: str, tool: dict) -> LLMResponse:
        return LLMResponse(
            data={"action": "offer", "outcome": {"price": 100}, "rationale": "ok"},
            tokens=1500,
            prompt_tokens=1200,
            completion_tokens=300,
            cost_usd=0.0042,
        )


def test_real_cost_flows_to_trace_and_payload(scenarios: dict[str, Scenario]) -> None:
    scenario = scenarios["single_price_a"]
    spec = EpisodeSpec(
        scenario=scenario,
        strategy="control",
        opponent="reciprocal",
        target_role="buyer",
        first_speaker="buyer",
        seed=0,
    )
    trace = run_episode(spec, agent_policy=LLMNegotiator("control", _CostClient()))
    # Per-turn costs summed across the agent's turns.
    n_agent_turns = sum(1 for s in trace.steps if s.party == "buyer")
    assert trace.cost_usd == pytest.approx(0.0042 * n_agent_turns)

    payload = build_payload([trace], scenarios)
    assert payload["meta"]["has_cost"] is True
    assert payload["meta"]["total_cost_usd"] == pytest.approx(trace.cost_usd)
    ep0 = payload["episodes"][0]
    assert ep0["cost_usd"] == pytest.approx(trace.cost_usd)
    # The episode carries its model, so the cost overview can break down by model.
    assert ep0["model"] == trace.llm_config.model

    # The dashboard ships the cost-overview panel and its renderer.
    html = build_dashboard([trace], scenarios)
    assert "Cost overview" in html and "function renderCost" in html


def test_scripted_run_reports_no_cost(traces: list, scenarios: dict[str, Scenario]) -> None:
    payload = build_payload(traces, scenarios)
    assert payload["meta"]["has_cost"] is False
    assert all(t.cost_usd is None for t in traces)


class _FullResponseClient:
    """Stub LLM client that returns a raw reply and a chain-of-thought."""

    def complete(self, system: str, user: str, tool: dict) -> LLMResponse:
        return LLMResponse(
            data={"action": "offer", "outcome": {"price": 100}, "rationale": "midpoint"},
            tokens=9,
            content='{"action": "offer", "outcome": {"price": 100}, "rationale": "midpoint"}',
            reasoning="Buyer wants low; I counter near the middle to keep momentum.",
        )


def test_full_response_and_reasoning_reach_payload_and_html(
    scenarios: dict[str, Scenario],
) -> None:
    spec = EpisodeSpec(
        scenario=scenarios["single_price_a"],
        strategy="anchoring",
        opponent="hardliner",
        target_role="seller",
        first_speaker="buyer",
        seed=0,
    )
    trace = run_episode(spec, agent_policy=LLMNegotiator("anchoring", _FullResponseClient()))
    agent_steps = [s for s in trace.steps if s.party == "seller"]
    assert agent_steps and all(s.raw_response and s.reasoning for s in agent_steps)

    payload = build_payload([trace], scenarios)
    assert payload["meta"]["has_reasoning"] is True
    step = next(s for s in payload["episodes"][0]["steps"] if s["party"] == "seller")
    assert step["raw_response"] and step["reasoning"]

    # The replay renders collapsible "thinking" and "full response" panels.
    html = build_dashboard([trace], scenarios)
    assert "full response" in html
    assert "thinking (" in html


def test_scripted_run_has_no_adjustments_or_errors(
    traces: list, scenarios: dict[str, Scenario]
) -> None:
    payload = build_payload(traces, scenarios)
    assert payload["meta"]["has_adjustments"] is False
    assert payload["meta"]["has_errors"] is False
    assert all(
        not s["adjusted"] and not s["error"] for e in payload["episodes"] for s in e["steps"]
    )


def test_scripted_run_has_no_messages(traces: list, scenarios: dict[str, Scenario]) -> None:
    payload = build_payload(traces, scenarios)
    assert payload["meta"]["has_messages"] is False
    assert all(s["rationale"] is None for e in payload["episodes"] for s in e["steps"])


def test_missing_control_does_not_crash(scenarios: dict[str, Scenario]) -> None:
    # A subset run that omits the control arm must degrade, not raise.
    specs = list(
        build_matrix(
            list(scenarios.values()),
            strategies=["boulware", "conceder"],
            opponents=["hardliner"],
            seeds=[0],
        )
    )
    ts = [run_episode(s) for s in specs]

    report = build_report(ts, scenarios)  # must not raise
    assert report.control_present is False
    assert "ABSENT" in report.render()
    assert all(e.strategy != "control" for e in report.effects)

    payload = build_payload(ts, scenarios)
    assert payload["meta"]["control_present"] is False
    # S_s is still present for every arm.
    assert payload["episodes"] and all(
        e["q"] is not None for e in payload["episodes"] if not e["no_zopa"]
    )


def test_render_is_self_contained(traces: list, scenarios: dict[str, Scenario]) -> None:
    html = build_dashboard(traces, scenarios, source="results/x.jsonl")
    # No external network loads: no remote scripts, styles, or imports. (A URL in
    # the vendored uPlot banner comment is inert and allowed.)
    assert 'src="http' not in html and "src='http" not in html
    assert 'href="http' not in html and "href='http" not in html
    assert "@import" not in html
    # uPlot is inlined, not linked.
    assert "var uPlot=function" in html
    # The payload placeholder was substituted with valid embedded JSON.
    assert "__RFQ_BENCH_PAYLOAD__" not in html
    start = html.index('id="payload"')
    blob = html[html.index(">", start) + 1 : html.index("</script>", start)]
    data = json.loads(blob.replace("<\\/", "</"))
    assert data["meta"]["n_episodes"] == len(traces)


def test_fidelity_reaches_payload_and_matches_report(
    traces: list, scenarios: dict[str, Scenario]
) -> None:
    from rfq_bench.report.fidelity import build_fidelity_report

    payload = build_payload(traces, scenarios)
    eps = payload["episodes"]
    assert all(e["fidelity"] is not None for e in eps)
    # Scripted agents are the strategy: every offer sits on the reference.
    offers = [o for e in eps for o in e["fidelity"]["offers"]]
    assert offers and all(o["gap"] == pytest.approx(0.0, abs=1e-12) for o in offers)
    # Pooling the payload per arm reproduces the text report's counts.
    rep = build_fidelity_report(traces, scenarios)
    for arm in rep.arms:
        mine = [e for e in eps if e["strategy"] == arm.strategy]
        assert len(mine) == arm.n_episodes
        assert sum(len(e["fidelity"]["offers"]) for e in mine) == arm.n_offers
    assert payload["meta"]["fidelity_tolerance"] == rep.tolerance
    assert 'id="fidPanel"' in build_dashboard(traces, scenarios)
