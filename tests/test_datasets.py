"""Dataset loading and ZOPA sanity checks."""

from __future__ import annotations

from rfq_bench.core.zopa import compute_zopa
from rfq_bench.datasets import load_scenarios


def test_anchor_set_loads() -> None:
    scenarios = load_scenarios("data/scenarios")
    ids = {s.id for s in scenarios}
    assert {
        "single_price_a",
        "single_price_b",
        "multi_pdw_a",
        "multi_pdw_b",
        "no_zopa_price",
    } <= ids


def test_zopa_present_except_diagnostic() -> None:
    for s in load_scenarios("data/scenarios"):
        exists = compute_zopa(s).exists
        if s.kind.value == "no_zopa_diagnostic":
            assert not exists
        else:
            assert exists
