"""End-to-end offline runner behavior."""

from __future__ import annotations

import pytest

from rfq_bench.core.scoring import score_trace
from rfq_bench.datasets import load_scenarios
from rfq_bench.report import build_report
from rfq_bench.runners.offline import EpisodeSpec, build_matrix, run_episode


def test_max_rounds_caps_the_deadline(single_issue_scenario) -> None:
    spec = EpisodeSpec(
        scenario=single_issue_scenario,
        strategy="anchoring",
        opponent="hardliner",
        target_role="seller",
        first_speaker="buyer",
        seed=0,
    )
    full = run_episode(spec)
    capped = run_episode(spec, max_rounds=3)
    # The cap limits the highest round index reached (rounds are 0-based).
    assert max(s.round for s in capped.steps) <= 2
    assert len(capped.steps) < len(full.steps)
    # A cap above the scenario deadline is a no-op (never extends the negotiation).
    wide = run_episode(spec, max_rounds=single_issue_scenario.deadline_rounds + 50)
    assert len(wide.steps) == len(full.steps)


def test_agreement_reached_and_scored(single_issue_scenario) -> None:
    spec = EpisodeSpec(
        scenario=single_issue_scenario,
        strategy="linear",
        opponent="fast_conceder",
        target_role="buyer",
        first_speaker="buyer",
        seed=0,
    )
    trace = run_episode(spec)
    assert trace.outcome_kind == "agreement"
    assert trace.agreement is not None
    # Buyer never accepts below its BATNA.
    assert trace.utilities["buyer"] >= single_issue_scenario.buyer.batna
    ep = score_trace(trace, single_issue_scenario)
    assert ep.scorable
    assert 0.0 <= ep.q <= 1.0


def test_no_zopa_walk_away(no_zopa_scenario) -> None:
    spec = EpisodeSpec(
        scenario=no_zopa_scenario,
        strategy="control",
        opponent="hardliner",
        target_role="buyer",
        first_speaker="buyer",
        seed=0,
    )
    trace = run_episode(spec)
    assert trace.agreement is None
    assert trace.outcome_kind == "walk_away"
    assert trace.validity["correct_walk_away"] is True
    # No ZOPA: a correct walk-away is a safety result (walk✓), not a q = 0 in S_s.
    ep = score_trace(trace, no_zopa_scenario)
    assert ep.no_zopa and ep.q is None and not ep.scorable


def test_determinism_same_inputs_same_trace(single_issue_scenario) -> None:
    spec = EpisodeSpec(
        scenario=single_issue_scenario,
        strategy="boulware",
        opponent="reciprocal",
        target_role="seller",
        first_speaker="seller",
        seed=0,
    )
    a = run_episode(spec)
    b = run_episode(spec)
    assert a.agreement == b.agreement
    assert [s.outcome for s in a.steps] == [s.outcome for s in b.steps]


def test_full_matrix_report_runs() -> None:
    scenarios = load_scenarios("data/scenarios")
    index = {s.id: s for s in scenarios}
    specs = list(
        build_matrix(
            scenarios,
            strategies=["control", "boulware", "conceder"],
            opponents=["hardliner", "fast_conceder"],
            seeds=[0],
        )
    )
    traces = [run_episode(s) for s in specs]
    rep = build_report(traces, index, n_boot=200, seed=0)
    names = {e.strategy for e in rep.effects}
    assert "control" in names
    # control's delta is exactly zero by construction.
    control = next(e for e in rep.effects if e.strategy == "control")
    assert control.delta == pytest.approx(0.0)
    # logrolling is skipped on single-issue scenarios but valid on multi-issue ones.
    assert rep.render()
