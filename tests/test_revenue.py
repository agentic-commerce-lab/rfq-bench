"""Revenue track: agreed price per episode, total per arm, share of the best price."""

from __future__ import annotations

from rfq_bench.datasets import load_scenarios
from rfq_bench.report import build_report
from rfq_bench.report.dashboard import build_payload
from rfq_bench.report.metrics import best_price, revenue
from rfq_bench.runners.offline import build_matrix, run_episode


def test_revenue_is_the_agreed_price_and_zero_without_a_deal(single_issue_scenario) -> None:
    assert revenue(single_issue_scenario, {"price": 100}) == 100.0
    assert revenue(single_issue_scenario, None) == 0.0
    assert best_price(single_issue_scenario) == 120.0  # seller's best option


def test_multi_issue_revenue_uses_the_price_issue(multi_issue_scenario) -> None:
    deal = {"price": 150, "delivery_days": 14, "warranty_months": 12}
    assert revenue(multi_issue_scenario, deal) == 150.0
    assert best_price(multi_issue_scenario) == 200.0


def _traces():
    scenarios = load_scenarios("data/scenarios")
    index = {s.id: s for s in scenarios}
    specs = build_matrix(
        scenarios, strategies=["control", "conceder"], opponents=["hardliner"], seeds=[0]
    )
    return [run_episode(s) for s in specs], index


def test_report_and_dashboard_agree_on_revenue() -> None:
    traces, index = _traces()
    rep = build_report(traces, index, n_boot=50, seed=0)
    for arm in ("control", "conceder"):
        arm_traces = [t for t in traces if t.strategy == arm]
        expected = sum(revenue(index[t.scenario_id], t.agreement) or 0.0 for t in arm_traces)
        assert rep.side[arm].total_revenue == expected
        shares = [
            (revenue(index[t.scenario_id], t.agreement) or 0.0) / best_price(index[t.scenario_id])
            for t in arm_traces
        ]
        assert abs(rep.side[arm].mean_revenue_share - sum(shares) / len(shares)) < 1e-12

    eps = build_payload(traces, index)["episodes"]
    dash_total = sum(e["revenue"] for e in eps if e["strategy"] == "control")
    assert dash_total == rep.side["control"].total_revenue
    assert all(0.0 <= e["revenue_share"] <= 1.0 for e in eps)


def test_report_shows_revenue_and_labels_it_spend_for_the_buyer() -> None:
    traces, index = _traces()
    text = build_report(traces, index, n_boot=50, seed=0).render()
    assert "revenue" in text and "%best" in text
    buyer = build_report(traces, index, n_boot=50, seed=0, role="buyer").render()
    assert "spend" in buyer
