"""Strategies and personas definable/overridable via markdown files."""

from __future__ import annotations

from pathlib import Path

from rfq_bench.agent.personas import (
    PERSONA_GUIDANCE,
    load_persona_guidance,
    persona_instruction,
)
from rfq_bench.agent.prompts import (
    STRATEGY_GUIDANCE,
    load_strategy_guidance,
    read_guidance_dir,
    strategy_instruction,
)


def test_read_guidance_dir_missing_is_empty(tmp_path: Path) -> None:
    assert read_guidance_dir(tmp_path / "nope") == {}


def test_read_guidance_dir_reads_md_by_stem(tmp_path: Path) -> None:
    (tmp_path / "aggressive.md").write_text("Push hard, concede little.", encoding="utf-8")
    (tmp_path / "empty.md").write_text("  \n", encoding="utf-8")  # skipped
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")  # not .md
    got = read_guidance_dir(tmp_path)
    assert got == {"aggressive": "Push hard, concede little."}


def test_load_strategy_guidance_adds_and_overrides(tmp_path: Path) -> None:
    (tmp_path / "aggressive.md").write_text("Be aggressive.", encoding="utf-8")
    (tmp_path / "control.md").write_text("OVERRIDDEN control.", encoding="utf-8")
    g = load_strategy_guidance(extra_dir=tmp_path)
    assert g["aggressive"] == "Be aggressive."  # new strategy
    assert g["control"] == "OVERRIDDEN control."  # override built-in
    assert g["boulware"] == STRATEGY_GUIDANCE["boulware"]  # untouched built-in kept


def test_load_persona_guidance_adds_and_overrides(tmp_path: Path) -> None:
    (tmp_path / "deadline_panic.md").write_text("You must close today.", encoding="utf-8")
    (tmp_path / "neutral.md").write_text("OVERRIDDEN neutral.", encoding="utf-8")
    g = load_persona_guidance(extra_dir=tmp_path)
    assert g["deadline_panic"] == "You must close today."
    assert g["neutral"] == "OVERRIDDEN neutral."
    assert g["hardball"] == PERSONA_GUIDANCE["hardball"]


def test_instruction_uses_provided_guidance_map() -> None:
    g = {"aggressive": "Be aggressive.", "control": "c", "neutral": "n"}
    assert strategy_instruction("aggressive", g) == "Be aggressive."
    assert persona_instruction("deadline_panic", {"neutral": "n"}) == "n"  # falls back to neutral


def test_negotiator_uses_file_defined_strategy(tmp_path: Path) -> None:
    from rfq_bench.agent.llm_client import LLMResponse
    from rfq_bench.agent.negotiator import LLMNegotiator
    from rfq_bench.strategies.base import NegotiationState

    (tmp_path / "aggressive.md").write_text("Be relentlessly aggressive.", encoding="utf-8")
    guidance = load_strategy_guidance(extra_dir=tmp_path)

    captured = {}

    class _C:
        def complete(self, system, user, tool):  # noqa: ANN001
            captured["user"] = user
            return LLMResponse(data={"action": "terminate"}, tokens=0)

    neg = LLMNegotiator(
        "aggressive",
        _C(),
        behavior_kind="strategy",
        instruction=strategy_instruction("aggressive", guidance),
    )
    # Minimal state via a tiny single-issue scenario is overkill; reuse a fixture below.
    from rfq_bench.datasets import load_scenarios

    scn = {s.id: s for s in load_scenarios("data/scenarios")}["single_price_a"]
    state = NegotiationState(
        role="seller",
        prefs=scn.preferences("seller"),
        issues=list(scn.issues),
        round=0,
        deadline=scn.deadline_rounds,
    )
    neg.act(None, state)
    assert "Be relentlessly aggressive." in captured["user"]
