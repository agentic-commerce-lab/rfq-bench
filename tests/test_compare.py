"""Cross-run comparison: matched cells, paired Δ, condition warnings (pure)."""

from __future__ import annotations

import pytest

from rfq_bench.core.contracts import LLMConfig, Step, Trace
from rfq_bench.report.compare import compare_runs


def _q_to_u(q: float) -> float:
    # single_issue_scenario: buyer batna 0.25, ideal 1.0.
    return 0.25 + q * 0.75


def _trace(
    scenario_id: str,
    q: float,
    *,
    model: str = "m-a",
    strategy: str = "control",
    seed: int = 0,
    tool_choice: str = "forced",
    errored: bool = False,
) -> Trace:
    u = _q_to_u(q)
    return Trace(
        scenario_id=scenario_id,
        outcome_space_version="v0.1",
        strategy=strategy,
        opponent="hardliner",
        target_role="buyer",
        first_speaker="buyer",
        seed=seed,
        llm_config=LLMConfig(
            base_url="x", model=model, temperature=0, max_tokens=10, tool_choice=tool_choice
        ),
        steps=[Step(round=0, party="buyer", action="offer", outcome={"price": 90})],
        outcome_kind="agreement",
        agreement={"price": 90},
        utilities={"buyer": u, "seller": 1 - u},
        rounds_to_close=1,
        latency_s=1.0,
        token_cost=10,
        errored=errored,
    )


@pytest.fixture
def scenarios(single_issue_scenario):
    s2 = single_issue_scenario.model_copy(update={"id": "s2"})
    s3 = single_issue_scenario.model_copy(update={"id": "s3"})
    return {s.id: s for s in (single_issue_scenario, s2, s3)}


def test_paired_delta_on_matched_cells_only(scenarios) -> None:
    ids = ["single_price_a", "s2", "s3"]
    a = [_trace(s, 0.4) for s in ids]
    # B: +0.2 on every shared cell, plus a cell A lacks (must be ignored).
    b = [_trace(s, 0.6, model="m-b") for s in ids] + [
        _trace("s2", 1.0, model="m-b", strategy="boulware")
    ]
    cmp = compare_runs(a, b, scenarios)
    o = cmp.overall
    assert o.score_a == pytest.approx(40.0) and o.score_b == pytest.approx(60.0)
    assert o.delta == pytest.approx(20.0)
    assert o.ci_low == pytest.approx(20.0) and o.ci_high == pytest.approx(20.0)
    assert o.n_cells == 3 and o.n_scenarios == 3
    assert cmp.cells_only_b == 1 and cmp.cells_only_a == 0
    assert "m-b captures" not in cmp.render()  # labels, not models, name the winner
    assert "B captures more value" in cmp.render()
    assert [r.name for r in cmp.by_strategy] == ["control"]


def test_seeds_are_averaged_within_a_cell(scenarios) -> None:
    a = [_trace("s2", 0.2, seed=0), _trace("s2", 0.6, seed=1)]  # cell mean 0.4
    b = [_trace("s2", 0.5, model="m-b")]
    o = compare_runs(a, b, scenarios).overall
    assert o.score_a == pytest.approx(40.0) and o.delta == pytest.approx(10.0)
    # A single scenario is a single cluster: no CI.
    assert o.ci_low != o.ci_low


def test_errored_cells_are_not_matched(scenarios) -> None:
    a = [_trace("s2", 0.4), _trace("s3", 0.4)]
    b = [_trace("s2", 0.9, model="m-b", errored=True), _trace("s3", 0.5, model="m-b")]
    cmp = compare_runs(a, b, scenarios)
    assert cmp.overall.n_cells == 1 and cmp.overall.delta == pytest.approx(10.0)
    # The errored episode still shows up in the separate-track error rate.
    assert cmp.side_b["error_rate"] == pytest.approx(0.5)


def test_condition_difference_other_than_model_warns(scenarios) -> None:
    a = [_trace("s2", 0.4)]
    b = [_trace("s2", 0.4, model="m-b", tool_choice="auto")]
    cmp = compare_runs(a, b, scenarios)
    assert any("tool_choice differs" in w for w in cmp.warnings)
    assert not any("model" in w for w in cmp.warnings)


def test_bootstrap_is_reproducible(scenarios) -> None:
    ids = ["single_price_a", "s2", "s3"]
    a = [_trace(s, 0.4) for s in ids]
    b = [_trace(s, q, model="m-b") for s, q in zip(ids, [0.1, 0.5, 0.9], strict=True)]
    r1 = compare_runs(a, b, scenarios, seed=3).overall
    r2 = compare_runs(a, b, scenarios, seed=3).overall
    assert (r1.ci_low, r1.ci_high) == (r2.ci_low, r2.ci_high)
    assert r1.ci_low < r1.delta < r1.ci_high


def test_html_dashboard_embeds_payload(scenarios) -> None:
    import json
    import re

    from rfq_bench.report.compare import comparison_payload, render_compare_html

    ids = ["single_price_a", "s2", "s3"]
    a = [_trace(s, 0.4) for s in ids]
    b = [_trace(s, 0.6, model="m-b", tool_choice="auto") for s in ids]
    cmp = compare_runs(a, b, scenarios, label_a="base", label_b="</script>x")
    payload = comparison_payload(cmp, source_a="a.jsonl", source_b="b.jsonl")
    html = render_compare_html(payload)
    assert "__RFQ_BENCH_COMPARE__" not in html
    # The label can't break out of the payload <script> tag.
    blob = re.search(r'id="payload" type="application/json">(.*?)</script>', html, re.S)
    assert blob is not None
    data = json.loads(blob.group(1))
    assert data["overall"]["delta"] == pytest.approx(20.0)
    assert data["label_b"] == "</script>x"
    assert any("tool_choice" in w for w in data["warnings"])
    assert {g["strategy"] for g in data["grid"]} == {"control"}
    assert [r["name"] for r in data["by_scenario"]] == sorted(ids)
