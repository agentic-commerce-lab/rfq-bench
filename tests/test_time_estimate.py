"""Pre-run wall-clock estimate from trace history (no network)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rfq_bench.agent.time_estimate import call_history, estimate_run_time

MODEL = "vendor/test-model"


def _trace(
    model: str,
    *,
    scenario: str = "s1",
    latency: float = 10.0,
    llm_steps: int = 5,
    scripted_steps: int = 5,
    errored: bool = False,
    base_url: str = "https://openrouter.ai/api/v1",
) -> str:
    steps = [{"party": "seller"}] * llm_steps + [{"party": "buyer"}] * scripted_steps
    return json.dumps(
        {
            "scenario_id": scenario,
            "target_role": "seller",
            "llm_config": {"model": model, "base_url": base_url},
            "steps": steps,
            "latency_s": latency,
            "token_cost_by_role": {"seller": 100, "buyer": 0},
            "errored": errored,
        }
    )


def _write(tmp_path: Path, *lines: str) -> Path:
    (tmp_path / "run.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path


def _specs(scenario, n: int = 1):
    return [SimpleNamespace(scenario=scenario) for _ in range(n)]


def test_history_counts_only_model_side_steps_and_skips_errors(tmp_path: Path) -> None:
    d = _write(
        tmp_path,
        _trace(MODEL, latency=10.0, llm_steps=5),
        _trace(MODEL, latency=99.0, errored=True),
        _trace("other/model", latency=99.0),
    )
    h = call_history(MODEL, d)
    assert h is not None
    assert h.calls == 5 and h.episodes == 1 and h.s_per_call == pytest.approx(2.0)


def test_legacy_trace_without_role_tokens_uses_target_role(tmp_path: Path) -> None:
    t = json.loads(_trace(MODEL, llm_steps=4, latency=8.0))
    del t["token_cost_by_role"]
    h = call_history(MODEL, _write(tmp_path, json.dumps(t)))
    assert h is not None and h.calls == 4 and h.s_per_call == pytest.approx(2.0)


def test_typical_uses_scenario_history_ceiling_uses_deadline(
    tmp_path: Path, multi_issue_scenario
) -> None:
    sc = multi_issue_scenario
    d = _write(tmp_path, _trace(MODEL, scenario=sc.id, latency=6.0, llm_steps=3))  # 2 s/call
    est = estimate_run_time(
        _specs(sc, 4),
        agent="llm",
        model=MODEL,
        max_rounds=None,
        concurrency=2,
        stagger_s=0.0,
        results_dir=d,
    )
    assert est is not None and est.same_model
    assert est.typical_s == pytest.approx(4 * min(3, sc.deadline_rounds) * 2.0 / 2)
    assert est.ceiling_s == pytest.approx(4 * sc.deadline_rounds * 2.0 / 2)


def test_stagger_ramp_is_a_floor(tmp_path: Path, multi_issue_scenario) -> None:
    d = _write(tmp_path, _trace(MODEL, latency=0.1, llm_steps=1))
    est = estimate_run_time(
        _specs(multi_issue_scenario, 11),
        agent="llm",
        model=MODEL,
        max_rounds=1,
        concurrency=10,
        stagger_s=5.0,
        results_dir=d,
    )
    assert est is not None and est.ceiling_s == pytest.approx(50.0)


def test_falls_back_to_other_remote_models_not_loopback(
    tmp_path: Path, multi_issue_scenario
) -> None:
    d = _write(
        tmp_path,
        _trace("other/model", latency=4.0, llm_steps=2),  # 2 s/call
        _trace("laya:english", latency=100.0, llm_steps=1, base_url="http://127.0.0.1:8770/api"),
    )
    est = estimate_run_time(
        _specs(multi_issue_scenario),
        agent="a2a",
        model=MODEL,
        max_rounds=None,
        concurrency=1,
        stagger_s=0.0,
        results_dir=d,
    )
    assert est is not None and not est.same_model
    assert est.s_per_call == pytest.approx(2.0)
    assert "other models" in est.basis


def test_none_without_history_or_for_scripted(tmp_path: Path, multi_issue_scenario) -> None:
    kw = dict(model=MODEL, max_rounds=None, concurrency=1, stagger_s=0.0, results_dir=tmp_path)
    assert estimate_run_time(_specs(multi_issue_scenario), agent="llm", **kw) is None
    _write(tmp_path, _trace(MODEL))
    assert estimate_run_time(_specs(multi_issue_scenario), agent="scripted", **kw) is None
