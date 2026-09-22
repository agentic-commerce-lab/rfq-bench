"""Configurable system prompts: markdown files, per-role, with a packaged default."""

from __future__ import annotations

from pathlib import Path

import pytest

from rfq_bench.agent.llm_client import LLMResponse
from rfq_bench.agent.negotiator import LLMNegotiator
from rfq_bench.agent.prompts import (
    _BUILTIN_SYSTEM_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    load_system_prompt,
    resolve_system_prompt,
)
from rfq_bench.strategies.base import NegotiationState


def test_packaged_default_matches_builtin_fallback() -> None:
    # The shipped markdown is the source of truth; the hardcoded fallback must not drift.
    assert DEFAULT_SYSTEM_PROMPT == _BUILTIN_SYSTEM_PROMPT


def test_resolve_defaults_to_packaged_when_no_files(tmp_path: Path) -> None:
    # Empty prompts dir -> the packaged default.
    assert resolve_system_prompt("buyer", prompts_dir=tmp_path) == DEFAULT_SYSTEM_PROMPT


def test_resolve_shared_file_in_prompts_dir(tmp_path: Path) -> None:
    (tmp_path / "system.md").write_text("SHARED PROMPT", encoding="utf-8")
    assert resolve_system_prompt("buyer", prompts_dir=tmp_path) == "SHARED PROMPT"
    assert resolve_system_prompt("seller", prompts_dir=tmp_path) == "SHARED PROMPT"


def test_resolve_per_role_file_wins_over_shared(tmp_path: Path) -> None:
    (tmp_path / "system.md").write_text("SHARED", encoding="utf-8")
    (tmp_path / "system_buyer.md").write_text("BUYER ONLY", encoding="utf-8")
    assert resolve_system_prompt("buyer", prompts_dir=tmp_path) == "BUYER ONLY"
    assert resolve_system_prompt("seller", prompts_dir=tmp_path) == "SHARED"  # falls through


def test_explicit_path_wins_and_missing_raises(tmp_path: Path) -> None:
    f = tmp_path / "custom.md"
    f.write_text("EXPLICIT", encoding="utf-8")
    (tmp_path / "system.md").write_text("SHARED", encoding="utf-8")
    assert resolve_system_prompt("seller", explicit=f, prompts_dir=tmp_path) == "EXPLICIT"
    with pytest.raises(FileNotFoundError):
        resolve_system_prompt("seller", explicit=tmp_path / "nope.md", prompts_dir=tmp_path)


def test_load_system_prompt_rejects_empty(tmp_path: Path) -> None:
    f = tmp_path / "empty.md"
    f.write_text("   \n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_system_prompt(f)


class _CaptureClient:
    def __init__(self) -> None:
        self.system: str | None = None

    def complete(self, system: str, user: str, tool: dict) -> LLMResponse:
        self.system = system
        return LLMResponse(data={"action": "terminate"}, tokens=0)


def test_negotiator_uses_injected_system_prompt(single_issue_scenario) -> None:
    client = _CaptureClient()
    neg = LLMNegotiator("control", client, system_prompt="CUSTOM SELLER PROMPT")
    state = NegotiationState(
        role="seller",
        prefs=single_issue_scenario.preferences("seller"),
        issues=list(single_issue_scenario.issues),
        round=0,
        deadline=single_issue_scenario.deadline_rounds,
    )
    neg.act(None, state)
    assert client.system == "CUSTOM SELLER PROMPT"


def test_negotiator_default_is_packaged_prompt(single_issue_scenario) -> None:
    client = _CaptureClient()
    neg = LLMNegotiator("control", client)  # no system_prompt -> default
    state = NegotiationState(
        role="seller",
        prefs=single_issue_scenario.preferences("seller"),
        issues=list(single_issue_scenario.issues),
        round=0,
        deadline=single_issue_scenario.deadline_rounds,
    )
    neg.act(None, state)
    assert client.system == DEFAULT_SYSTEM_PROMPT
