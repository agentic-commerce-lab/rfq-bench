"""Exact-value tests for the scientific kernel."""

from __future__ import annotations

import math

import pytest

from rfq_bench.core.contracts import LLMConfig, Step, Trace
from rfq_bench.core.scoring import (
    ScoredEpisode,
    episode_score,
    score_trace,
    strategy_effects,
    strategy_score,
)


def test_episode_score_worked_example() -> None:
    # From the concept: U=80, d=50, I=100 -> q=0.60.
    assert episode_score(80, 50, 100) == pytest.approx(0.60)


def test_episode_score_clips_below_batna() -> None:
    assert episode_score(40, 50, 100) == 0.0


def test_episode_score_clips_above_ideal() -> None:
    assert episode_score(120, 50, 100) == 1.0


def test_episode_score_no_deal_at_batna_is_zero() -> None:
    assert episode_score(50, 50, 100) == 0.0


def test_episode_score_degenerate_returns_none() -> None:
    assert episode_score(50, 50, 50) is None


def _trace(strategy: str, scenario_id: str, u_buyer: float, **kw: object) -> Trace:
    return Trace(
        scenario_id=scenario_id,
        outcome_space_version="v0.1",
        strategy=strategy,
        opponent=str(kw.get("opponent", "hardliner")),
        target_role="buyer",
        first_speaker="buyer",
        seed=int(kw.get("seed", 0)),  # type: ignore[arg-type]
        llm_config=LLMConfig(base_url="x", model="m", temperature=0, max_tokens=10),
        steps=[Step(round=0, party="buyer", action="offer", outcome={"price": 90})],
        outcome_kind="agreement",
        agreement={"price": 90},
        utilities={"buyer": u_buyer, "seller": 1 - u_buyer},
        rounds_to_close=1,
        latency_s=0.1,
        token_cost=100,
    )


def test_score_trace_uses_scenario_private_values(single_issue_scenario) -> None:
    # Agreement at price 90 -> buyer utility 0.75, batna 0.25, ideal 1.0.
    tr = _trace("control", single_issue_scenario.id, u_buyer=0.75)
    ep = score_trace(tr, single_issue_scenario)
    # q = (0.75 - 0.25) / (1.0 - 0.25) = 0.6667
    assert ep.q == pytest.approx((0.75 - 0.25) / (1.0 - 0.25))
    assert ep.scorable


def test_strategy_score_macro_average() -> None:
    eps = [
        ScoredEpisode("s1", "boulware", "hardliner", "buyer", "buyer", 0.4, False),
        ScoredEpisode("s2", "boulware", "hardliner", "buyer", "buyer", 0.8, False),
    ]
    assert strategy_score(eps) == pytest.approx(60.0)


def test_strategy_score_excludes_degenerate() -> None:
    eps = [
        ScoredEpisode("s1", "boulware", "hardliner", "buyer", "buyer", 0.5, False),
        ScoredEpisode("s2", "boulware", "hardliner", "buyer", "buyer", None, True),
    ]
    assert strategy_score(eps) == pytest.approx(50.0)


def test_strategy_effects_delta_and_ordering() -> None:
    eps = [
        # control mean q = 0.5 -> S = 50
        ScoredEpisode("s1", "control", "hardliner", "buyer", "buyer", 0.4, False),
        ScoredEpisode("s2", "control", "hardliner", "buyer", "buyer", 0.6, False),
        # boulware mean q = 0.8 -> S = 80, delta = +30
        ScoredEpisode("s1", "boulware", "hardliner", "buyer", "buyer", 0.7, False),
        ScoredEpisode("s2", "boulware", "hardliner", "buyer", "buyer", 0.9, False),
    ]
    results = strategy_effects(eps, control="control", n_boot=200, seed=0)
    by_name = {r.strategy: r for r in results}
    assert by_name["control"].delta == pytest.approx(0.0)
    assert by_name["boulware"].delta == pytest.approx(30.0)
    # control sorts first; arms by descending delta.
    assert results[0].strategy == "control"
    # CI is finite for the arm.
    assert math.isfinite(by_name["boulware"].ci_low)
    assert math.isfinite(by_name["boulware"].ci_high)


def test_strategy_effects_requires_control() -> None:
    eps = [ScoredEpisode("s1", "boulware", "hardliner", "buyer", "buyer", 0.5, False)]
    with pytest.raises(ValueError, match="control"):
        strategy_effects(eps, control="control")


def test_strategy_effects_without_control_is_graceful() -> None:
    # A subset run that omits the control arm: S_s defined, effect undefined.
    eps = [
        ScoredEpisode("s1", "boulware", "hardliner", "buyer", "buyer", 0.7, False),
        ScoredEpisode("s2", "boulware", "hardliner", "buyer", "buyer", 0.9, False),
        ScoredEpisode("s1", "conceder", "hardliner", "buyer", "buyer", 0.2, False),
    ]
    results = strategy_effects(eps, control="control", require_control=False)
    by_name = {r.strategy: r for r in results}
    assert by_name["boulware"].score == pytest.approx(80.0)
    assert math.isnan(by_name["boulware"].delta)
    assert math.isnan(by_name["boulware"].ci_low)
    # With no control, arms are ranked by S_s descending.
    assert [r.strategy for r in results] == ["boulware", "conceder"]
