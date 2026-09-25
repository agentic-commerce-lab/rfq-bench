"""Reproducibility: [llm] config pins, trace provenance, provider capture (no network)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from rfq_bench.agent.llm_client import _extract_provider
from rfq_bench.config import load_config
from rfq_bench.provenance import (
    condition_differences,
    fingerprint,
    run_provenance,
    tool_fingerprint,
)
from rfq_bench.runners.offline import EpisodeSpec, run_episode


def test_llm_section_parses_and_rejects_bad_values(tmp_path: Path) -> None:
    ok = tmp_path / "ok.toml"
    ok.write_text('[llm]\ntool_choice = "auto"\nmax_tokens = 4096\ntemperature = 0.0\n')
    cfg = load_config(ok)
    assert cfg.llm.overrides() == {"tool_choice": "auto", "max_tokens": 4096, "temperature": 0.0}
    bad = tmp_path / "bad.toml"
    bad.write_text('[llm]\ntool_choice = "sometimes"\n')
    with pytest.raises(ValueError):
        load_config(bad)


def test_config_llm_overrides_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from rfq_bench.cli import _agent_settings

    monkeypatch.setenv("RFQ_BENCH_TOOL_CHOICE", "forced")
    monkeypatch.setenv("RFQ_BENCH_MAX_TOKENS", "512")
    f = tmp_path / "c.toml"
    f.write_text('[llm]\ntool_choice = "auto"\n')
    s = _agent_settings(load_config(f).llm)
    assert s.tool_choice == "auto"  # from the config
    assert s.max_tokens == 512  # not pinned: still from the environment
    assert s.to_llm_config().tool_choice == "auto"


def test_fingerprints_are_stable_and_scenario_independent() -> None:
    assert fingerprint("abc") == fingerprint("abc") and len(fingerprint("abc")) == 16
    assert fingerprint({"b": 1, "a": 2}) == fingerprint({"a": 2, "b": 1})
    t1 = tool_fingerprint(strict=False, message_channel=True)
    assert t1 == tool_fingerprint(strict=False, message_channel=True)
    assert t1 != tool_fingerprint(strict=False, message_channel=False)
    prov = run_provenance()
    assert set(prov) == {"code_version", "started_at"} and prov["code_version"]


def test_condition_differences_between_and_within_runs() -> None:
    a = [{"system_prompt_seller": "x", "instruction_seller.control": "i", "started_at": "t1"}]
    same = [{"system_prompt_seller": "x", "instruction_seller.control": "i", "started_at": "t2"}]
    changed = [{"system_prompt_seller": "y", "instruction_seller.control": "i"}]
    assert condition_differences(a, same) == []  # run-only keys never warn
    assert condition_differences(a, changed) == ["seller system prompt differs between the runs"]
    assert condition_differences(a + changed) == [
        "seller system prompt changed within the run (2 versions)"
    ]
    # A key only one side has (e.g. another strategy) is not a difference.
    other_arm = [{"system_prompt_seller": "x", "instruction_seller.boulware": "j"}]
    assert condition_differences(a, other_arm) == []


def test_provider_is_read_from_attr_or_model_extra() -> None:
    assert _extract_provider(SimpleNamespace(provider="Azure")) == "Azure"
    assert _extract_provider(SimpleNamespace(model_extra={"provider": "Anthropic"})) == "Anthropic"
    assert _extract_provider(SimpleNamespace(model_extra={})) is None


def test_run_episode_records_provenance(single_issue_scenario) -> None:
    spec = EpisodeSpec(
        scenario=single_issue_scenario,
        strategy="control",
        opponent="hardliner",
        target_role="seller",
        first_speaker="buyer",
        seed=0,
    )
    t = run_episode(spec, provenance={"code_version": "abc", "tool_schema": "f00"})
    assert t.provenance == {"code_version": "abc", "tool_schema": "f00"}
    assert all(s.provider is None for s in t.steps)  # scripted: no provider
