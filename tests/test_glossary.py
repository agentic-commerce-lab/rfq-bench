"""Plain-language tooltip descriptions for the dashboard."""

from __future__ import annotations

from rfq_bench.agent.personas import load_persona_guidance
from rfq_bench.datasets import load_scenarios
from rfq_bench.opponents import OPPONENTS
from rfq_bench.report.glossary import (
    OPPONENT_DESCRIPTIONS,
    STRATEGY_DESCRIPTIONS,
    build_glossary,
    scenario_description,
)
from rfq_bench.strategies import STRATEGIES


def test_every_builtin_name_has_a_description() -> None:
    # A new built-in strategy, scripted opponent or persona needs a tooltip too.
    assert set(STRATEGIES) <= set(STRATEGY_DESCRIPTIONS)
    assert set(OPPONENTS) <= set(OPPONENT_DESCRIPTIONS)
    assert set(load_persona_guidance()) <= set(OPPONENT_DESCRIPTIONS)
    for text in (*STRATEGY_DESCRIPTIONS.values(), *OPPONENT_DESCRIPTIONS.values()):
        assert text.strip().endswith(".")


def test_single_issue_description_states_both_limits_and_the_overlap() -> None:
    sc = {s.id: s for s in load_scenarios("data/scenarios_price")}["price_narrow"]
    text = scenario_description(sc)
    assert "pays at most 260" in text and "needs at least 245" in text
    assert "(4 options)" in text


def test_no_zopa_description_says_walking_away_is_right() -> None:
    sc = {s.id: s for s in load_scenarios("data/scenarios")}["no_zopa_price"]
    assert "walking away is the right outcome" in scenario_description(sc)


def test_multi_issue_description_names_the_real_trade(multi_issue_scenario) -> None:
    # multi_pdw_a: both put price first; next the buyer values warranty, the seller delivery.
    text = scenario_description(multi_issue_scenario)
    assert "Both care most about price" in text
    assert "buyer values warranty months" in text and "seller delivery days" in text


def test_custom_names_fall_back_to_their_guidance_text(multi_issue_scenario) -> None:
    g = build_glossary(
        strategies=["control", "my_strategy"],
        opponents=["my_persona", "unknown"],
        scenarios=[multi_issue_scenario],
        strategy_guidance={"my_strategy": "Open high. Then wait for the deadline."},
        persona_guidance={"my_persona": "Always ask for free shipping."},
    )
    assert g["strategy"]["control"] == STRATEGY_DESCRIPTIONS["control"]
    assert g["strategy"]["my_strategy"] == "Custom: Open high."
    assert g["opponent"]["my_persona"] == "Custom: Always ask for free shipping."
    assert g["opponent"]["unknown"] == ""
    assert multi_issue_scenario.id in g["scenario"]
