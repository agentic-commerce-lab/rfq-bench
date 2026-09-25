"""Model duel: pairings, matrix, trace provenance, and the scoring maths (no network)."""

from __future__ import annotations

import json
import re

import pytest

from rfq_bench.core.contracts import LLMConfig, Step, Trace
from rfq_bench.report.duel import build_duel, duel_payload, render_duel_html
from rfq_bench.runners.offline import (
    EpisodeSpec,
    build_duel_matrix,
    duel_pairings,
    run_episode,
)

A, B = "vendor/a-model", "vendor/b-model"


def test_pairings_cross_and_full() -> None:
    assert duel_pairings((A, B), "cross") == [(A, B), (B, A)]
    assert duel_pairings((A, B), "full") == [(A, B), (B, A), (A, A), (B, B)]
    with pytest.raises(ValueError):
        duel_pairings((A, A), "cross")
    with pytest.raises(ValueError):
        duel_pairings((A, B), "round-robin")


def test_matrix_plays_every_cell_once_per_pairing(single_issue_scenario) -> None:
    specs = list(
        build_duel_matrix(
            [single_issue_scenario],
            models=(A, B),
            pairings="full",
            personas=["neutral"],
            strategies=["control"],
            seeds=[0],
            first_speakers=("buyer", "seller"),
        )
    )
    assert len(specs) == 2 * 4  # first speakers x pairings
    assert {(s.seller_model, s.buyer_model) for s in specs} == set(duel_pairings((A, B), "full"))
    assert all(s.mode == "duel" and s.target_role == "seller" for s in specs)


def test_trace_records_mode_and_models(single_issue_scenario) -> None:
    spec = EpisodeSpec(
        scenario=single_issue_scenario,
        strategy="control",
        opponent="hardliner",  # scripted stand-in: the provenance is what's tested
        target_role="seller",
        first_speaker="buyer",
        seed=0,
        mode="duel",
        seller_model=A,
        buyer_model=B,
    )
    t = run_episode(spec)
    assert t.mode == "duel"
    assert t.models_by_role == {"seller": A, "buyer": B}


def _u(q: float) -> float:
    # single_issue_scenario: both BATNAs 0.25, ideal 1.0.
    return 0.25 + 0.75 * q


def _trace(scenario_id: str, seller: str, buyer: str, q_s: float, q_b: float) -> Trace:
    return Trace(
        scenario_id=scenario_id,
        outcome_space_version="v0.1",
        strategy="control",
        opponent="neutral",
        target_role="seller",
        first_speaker="buyer",
        seed=0,
        llm_config=LLMConfig(base_url="x", model=seller, temperature=0, max_tokens=10),
        steps=[Step(round=0, party="buyer", action="offer", outcome={"price": 100})],
        outcome_kind="agreement",
        agreement={"price": 100},
        utilities={"seller": _u(q_s), "buyer": _u(q_b)},
        rounds_to_close=1,
        latency_s=1.0,
        token_cost=10,
        mode="duel",
        persona="neutral",
        models_by_role={"seller": seller, "buyer": buyer},
    )


@pytest.fixture
def scenarios(single_issue_scenario):
    copies = [single_issue_scenario.model_copy(update={"id": f"s{i}"}) for i in range(3)]
    return {s.id: s for s in copies}


# (seller, buyer) -> (seller q, buyer q), identical in every scenario.
_Q = {(A, B): (0.6, 0.2), (B, A): (0.4, 0.3), (A, A): (0.5, 0.5), (B, B): (0.5, 0.5)}


def _duel_traces(scenarios, pairs):
    return [_trace(sid, s, b, *_Q[(s, b)]) for sid in scenarios for (s, b) in pairs]


def test_model_score_averages_both_roles_against_the_other_model(scenarios) -> None:
    d = build_duel(_duel_traces(scenarios, [(A, B), (B, A)]), scenarios)
    o = d.overall
    # S_A = (s[A→B] + b[B→A]) / 2 = (0.6 + 0.3) / 2; S_B = (s[B→A] + b[A→B]) / 2 = (0.4 + 0.2) / 2
    assert o.score_a == pytest.approx(45.0) and o.score_b == pytest.approx(30.0)
    assert o.delta == pytest.approx(-15.0)
    assert o.ci_low == pytest.approx(-15.0) and o.ci_high == pytest.approx(-15.0)
    assert o.n_cells == 3
    assert d.seller_skill is None and d.buyer_skill is None  # cross only
    assert "A Model captures more value" in d.render()
    assert "S_A" not in d.render()


def test_full_pairings_separate_seller_and_buyer_skill(scenarios) -> None:
    d = build_duel(_duel_traces(scenarios, list(_Q)), scenarios)
    assert d.seller_skill is not None and d.buyer_skill is not None
    # A sells: mean(0.6, 0.5); B sells: mean(0.4, 0.5)
    assert d.seller_skill.delta == pytest.approx(-10.0)
    # A buys: mean(b[A→A]=0.5, b[B→A]=0.3); B buys: mean(b[A→B]=0.2, b[B→B]=0.5)
    assert d.buyer_skill.delta == pytest.approx(-5.0)
    assert d.pairing_scores[(A, B)] == pytest.approx(60.0)


def test_cells_missing_a_pairing_are_not_matched(scenarios) -> None:
    ts = _duel_traces(scenarios, [(A, B), (B, A)])
    ts = [t for t in ts if not (t.scenario_id == "s0" and t.models_by_role["seller"] == B)]
    d = build_duel(ts, scenarios)
    assert d.overall.n_cells == 2
    assert any("unequal episode counts" in w for w in d.warnings)


def test_duel_needs_both_cross_pairings(scenarios) -> None:
    with pytest.raises(ValueError, match="cross pairings"):
        build_duel(_duel_traces(scenarios, [(A, B), (A, A), (B, B)]), scenarios)


def test_duel_dashboard_payload(scenarios) -> None:
    ts = _duel_traces(scenarios, list(_Q))
    d = build_duel(ts, scenarios)
    html = render_duel_html(duel_payload(d, ts, scenarios, source="results/duel.jsonl"))
    assert "__RFQ_BENCH_DUEL__" not in html and "__REPLAY_JS__" not in html
    assert "function renderReplay" in html  # the shared replay is inlined
    blob = re.search(r'id="payload" type="application/json">(.*?)</script>', html, re.S)
    assert blob is not None
    data = json.loads(blob.group(1))
    assert [m["name"] for m in data["models"]] == ["A Model", "B Model"]
    assert data["models"][0]["provider"] == "Vendor"
    assert data["n_negotiations"] == 6
    assert [r["name"] for r in data["by_scenario"]] == ["Widget"] * 3
    assert data["winner"] == A  # A leads by 15 points with a CI clear of 0
    assert [r["name"] for r in data["roles"]] == ["When selling", "When buying"]
    assert data["side"]["rows"][1]["label"].startswith("Deals closed")
    assert len(data["matchups"]) == 4 and len(data["episodes"]) == len(ts)
    assert data["episodes"][0]["models_by_role"] == ts[0].models_by_role
    assert data["side"]["cols"][0] == {"seller": A, "buyer": B}


def test_model_usage_sums_each_models_sides(scenarios) -> None:
    from rfq_bench.report.duel import model_usage

    base = _trace("s0", A, B, 0.5, 0.5)
    t = base.model_copy(
        update={
            "steps": [
                Step(round=0, party="buyer", action="offer", outcome={"price": 100}),
                Step(round=0, party="seller", action="offer", outcome={"price": 110}),
                Step(round=1, party="buyer", action="accept"),
            ],
            "token_cost_by_role": {"seller": 300, "buyer": 500},
            "prompt_tokens_by_role": {"seller": 200, "buyer": 400},
            "cached_tokens_by_role": {"seller": 100, "buyer": 0},
            "cost_usd_by_role": {"seller": 0.03, "buyer": 0.01},
        }
    )
    selfplay = t.model_copy(update={"models_by_role": {"seller": A, "buyer": A}})
    u = model_usage([t, selfplay])
    # A: seller in both episodes + buyer in the self-play one.
    assert u[A].calls == 1 + 1 + 2 and u[A].tokens == 300 + 300 + 500
    assert u[A].cost_usd == pytest.approx(0.07) and u[A].completion_tokens == 1100 - 800
    assert u[A].cache_rate == pytest.approx(200 / 800)
    assert u[B].calls == 2 and u[B].cost_usd == pytest.approx(0.01)


def test_display_names_are_reader_friendly() -> None:
    from rfq_bench.report.duel import _provider, _short

    assert _short("anthropic/claude-opus-5.5") == "Claude Opus 5.5"
    assert _short("openai/gpt-6-astra-pro") == "GPT-6 Astra Pro"
    assert _short("openai/gpt-5.6-terra") == "GPT-5.6 Terra"
    assert _provider("openai/gpt-6-sol") == "OpenAI" and _provider("anthropic/x") == "Anthropic"
