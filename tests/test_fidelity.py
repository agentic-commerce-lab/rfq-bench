"""Strategy fidelity: replaying the reference strategy over a persisted trace."""

from __future__ import annotations

from typing import Any

import pytest

from rfq_bench.core.contracts import LLMConfig, Scenario, Step, Trace
from rfq_bench.opponents import OPPONENTS
from rfq_bench.report.fidelity import _fit_beta, build_fidelity_report, episode_fidelity
from rfq_bench.runners.offline import EpisodeSpec, run_episode
from rfq_bench.strategies import STRATEGIES
from rfq_bench.strategies.registry import MULTI_ISSUE_ONLY


def _trace(
    scenario: Scenario,
    steps: list[tuple[int, str, str, Any]],
    *,
    strategy: str = "boulware",
    **kw: Any,
) -> Trace:
    return Trace(
        scenario_id=scenario.id,
        outcome_space_version="v0.1",
        strategy=strategy,
        opponent="hardliner",
        target_role="seller",
        first_speaker="buyer",
        seed=0,
        llm_config=LLMConfig(base_url="x", model="m", temperature=0, max_tokens=10),
        steps=[
            Step(round=r, party=p, action=a, outcome=None if o is None else {"price": o})  # type: ignore[arg-type]
            for r, p, a, o in steps
        ],
        outcome_kind=kw.get("outcome_kind", "agreement"),
        agreement=kw.get("agreement"),
        utilities={"buyer": 0.5, "seller": 0.5},
        rounds_to_close=len(steps),
        latency_s=0.0,
        token_cost=0,
        deadline=kw.get("deadline", scenario.deadline_rounds),
        errored=kw.get("errored", False),
    )


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("opponent", OPPONENTS)
@pytest.mark.parametrize("role", ["buyer", "seller"])
def test_scripted_agent_replays_with_zero_gap(
    strategy: str, opponent: str, role: str, single_issue_scenario, multi_issue_scenario
) -> None:
    # The correctness anchor: an agent that IS the strategy must score perfect fidelity.
    scenario = multi_issue_scenario if strategy in MULTI_ISSUE_ONLY else single_issue_scenario
    for first in ("buyer", "seller"):
        spec = EpisodeSpec(
            scenario=scenario,
            strategy=strategy,
            opponent=opponent,
            target_role=role,  # type: ignore[arg-type]
            first_speaker=first,  # type: ignore[arg-type]
            seed=0,
        )
        ep = episode_fidelity(run_episode(spec), scenario)
        assert ep is not None
        assert ep.offers, "the agent made at least one offer"
        assert all(o.gap == pytest.approx(0.0, abs=1e-12) for o in ep.offers)
        assert ep.early_accepts == 0 and ep.missed_accepts == 0


def test_capped_deadline_is_recorded_and_replayed(single_issue_scenario) -> None:
    spec = EpisodeSpec(
        scenario=single_issue_scenario,
        strategy="linear",
        opponent="hardliner",
        target_role="seller",
        first_speaker="buyer",
        seed=0,
    )
    trace = run_episode(spec, max_rounds=4)
    assert trace.deadline == 4
    ep = episode_fidelity(trace, single_issue_scenario)
    assert ep is not None and all(o.gap == pytest.approx(0.0, abs=1e-12) for o in ep.offers)


def test_early_concession_and_early_accept_are_measured(single_issue_scenario) -> None:
    # Boulware seller (opening 1.0, beta 0.05) should hold at 120 (utility 1.0) early.
    # Seller range: BATNA 0.25 -> ideal 1.0, span 0.75.
    trace = _trace(
        single_issue_scenario,
        [
            (0, "buyer", "offer", 80),
            (0, "seller", "offer", 120),  # on strategy
            (1, "buyer", "offer", 80),
            (1, "seller", "offer", 100),  # utility 0.5 vs reference 1.0
            (2, "buyer", "offer", 90),
            (2, "seller", "accept", 90),  # utility 0.25: Boulware would not accept
        ],
        agreement={"price": 90},
    )
    ep = episode_fidelity(trace, single_issue_scenario)
    assert ep is not None
    assert [o.gap for o in ep.offers] == pytest.approx([0.0, (0.5 - 1.0) / 0.75])
    assert ep.opening_gap == pytest.approx(0.0)
    assert ep.accept_decisions == 3
    assert ep.early_accepts == 1
    assert ep.missed_accepts == 0


def test_missed_accept_is_measured(single_issue_scenario) -> None:
    # The buyer offers the seller's ideal (120); every strategy accepts that.
    trace = _trace(
        single_issue_scenario,
        [(0, "buyer", "offer", 120), (0, "seller", "offer", 120), (1, "buyer", "accept", 120)],
        strategy="conceder",
        agreement={"price": 120},
    )
    ep = episode_fidelity(trace, single_issue_scenario)
    assert ep is not None and ep.missed_accepts == 1 and ep.early_accepts == 0


def test_old_trace_without_deadline_infers_it_from_a_timeout(single_issue_scenario) -> None:
    steps = [
        (r, p, "offer", 80 if p == "buyer" else 120) for r in range(3) for p in ("buyer", "seller")
    ]
    trace = _trace(single_issue_scenario, steps, outcome_kind="no_deal", deadline=None)
    ep = episode_fidelity(trace, single_issue_scenario)
    assert ep is not None and ep.deadline == 3


def test_unknown_or_errored_traces_are_skipped(single_issue_scenario) -> None:
    steps = [(0, "buyer", "offer", 80), (0, "seller", "offer", 120)]
    custom = _trace(single_issue_scenario, steps, strategy="my_custom_prompt")
    errored = _trace(single_issue_scenario, steps, errored=True)
    ok = _trace(single_issue_scenario, steps)
    rep = build_fidelity_report(
        [custom, errored, ok], {single_issue_scenario.id: single_issue_scenario}
    )
    assert rep.n_checked == 1 and rep.n_skipped == 2
    assert rep.skipped_strategies == ["my_custom_prompt"]
    assert [a.strategy for a in rep.arms] == ["boulware"]
    assert "Strategy fidelity" in rep.render()


@pytest.mark.parametrize("beta", [0.2, 1.0, 5.0])
def test_beta_fit_recovers_the_concession_shape(beta: float) -> None:
    pts = [(t / 9, 0.9 * (1 - (t / 9) ** (1 / beta))) for t in range(10)]
    fitted = _fit_beta(pts)
    assert fitted is not None and fitted == pytest.approx(beta, rel=0.05)


def test_beta_fit_needs_enough_offers() -> None:
    assert _fit_beta([(0.0, 0.9), (0.5, 0.5)]) is None
