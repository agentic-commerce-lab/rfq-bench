"""Pre-run cost estimate: call ceiling, history rate, pricing fallback (no network)."""

from __future__ import annotations

import json
from types import SimpleNamespace

from rfq_bench.agent.cost_estimate import (
    ASSUMED_COMPLETION_TOKENS,
    calls_ceiling,
    estimate_run_cost,
    history_rate,
)
from rfq_bench.agent.settings import AgentSettings

MODEL = "vendor/test-model"


def _settings(base_url: str = "https://openrouter.ai/api/v1", **kw) -> AgentSettings:
    return AgentSettings(
        OPENAI_BASE_URL=base_url, RFQ_BENCH_MODEL=MODEL, OPENAI_API_KEY="test-key", **kw
    )


def _specs(scenario, n: int = 1):
    return [SimpleNamespace(scenario=scenario) for _ in range(n)]


def _trace(model: str, parties: list[str], cost_by_role: dict[str, float]) -> str:
    return json.dumps(
        {
            "llm_config": {"model": model},
            "steps": [{"party": p} for p in parties],
            "cost_usd_by_role": cost_by_role,
        }
    )


class _Fetch:
    """Fake OpenRouter: /models pricing and /key credit; records every URL."""

    def __init__(self, *, pricing=None, remaining=None, fail: bool = False) -> None:
        self.pricing, self.remaining, self.fail = pricing, remaining, fail
        self.urls: list[str] = []

    def __call__(self, url: str, headers: dict[str, str]):
        self.urls.append(url)
        if self.fail:
            raise TimeoutError("offline")
        if url.endswith("/key"):
            assert headers["Authorization"] == "Bearer test-key"
            return {"data": {"limit_remaining": self.remaining}}
        models = [{"id": MODEL, "pricing": self.pricing}] if self.pricing else []
        return {"data": models}


def test_ceiling_is_two_calls_per_round_in_a2a_one_in_llm(multi_issue_scenario) -> None:
    d = multi_issue_scenario.deadline_rounds
    specs = _specs(multi_issue_scenario, 3)
    assert calls_ceiling(specs, agent="a2a", max_rounds=None) == 3 * 2 * d
    assert calls_ceiling(specs, agent="llm", max_rounds=None) == 3 * d


def test_ceiling_respects_max_rounds_but_never_exceeds_deadline(multi_issue_scenario) -> None:
    d = multi_issue_scenario.deadline_rounds
    specs = _specs(multi_issue_scenario)
    assert calls_ceiling(specs, agent="a2a", max_rounds=4) == 2 * 4
    assert calls_ceiling(specs, agent="a2a", max_rounds=d + 50) == 2 * d


def test_history_rate_divides_role_cost_by_that_roles_calls(tmp_path) -> None:
    lines = [
        # A2A: both sides cost; 2 buyer + 2 seller calls -> $0.8 / 4 calls.
        _trace(MODEL, ["buyer", "seller", "buyer", "seller"], {"buyer": 0.4, "seller": 0.4}),
        # --agent llm: only the agent side reports cost; the scripted side is ignored.
        _trace(MODEL, ["buyer", "seller", "buyer", "seller"], {"seller": 0.2}),
        # Another model: excluded.
        _trace("other/model", ["buyer"], {"buyer": 99.0}),
    ]
    (tmp_path / "run.jsonl").write_text("\n".join(lines) + "\nnot json\n")
    usd, calls, files = history_rate(MODEL, tmp_path)  # type: ignore[misc]
    assert calls == 6
    assert abs(usd - 1.0 / 6) < 1e-12
    assert files == ("run.jsonl",)


def test_history_rate_none_without_matching_traces(tmp_path) -> None:
    (tmp_path / "run.jsonl").write_text(_trace("other/model", ["buyer"], {"buyer": 1.0}))
    assert history_rate(MODEL, tmp_path) is None


def test_estimate_prefers_history_and_reports_credit(tmp_path, multi_issue_scenario) -> None:
    (tmp_path / "run.jsonl").write_text(
        _trace(MODEL, ["buyer", "seller"], {"buyer": 0.01, "seller": 0.01})
    )
    fetch = _Fetch(pricing={"prompt": "0.001", "completion": "0.001"}, remaining=2.5)
    est = estimate_run_cost(
        _specs(multi_issue_scenario, 2),
        agent="a2a",
        max_rounds=None,
        settings=_settings(),
        results_dir=tmp_path,
        fetch=fetch,
    )
    assert est is not None
    assert est.usd_per_call == 0.01
    assert est.max_usd == est.calls_max * 0.01
    assert est.worst_usd is None  # history already reflects real completion sizes
    assert "measured over 2 past calls" in est.basis
    assert est.credit_remaining == 2.5


def test_estimate_falls_back_to_published_pricing(tmp_path, multi_issue_scenario) -> None:
    fetch = _Fetch(pricing={"prompt": "0.000001", "completion": "0.000002"})
    est = estimate_run_cost(
        _specs(multi_issue_scenario),
        agent="a2a",
        max_rounds=None,
        settings=_settings(RFQ_BENCH_MAX_TOKENS=4096),
        results_dir=tmp_path,
        fetch=fetch,
    )
    assert est is not None and est.usd_per_call is not None and est.worst_usd is not None
    assert "published pricing" in est.basis
    prompt_tokens = (est.usd_per_call - ASSUMED_COMPLETION_TOKENS * 2e-6) / 1e-6
    assert 200 < prompt_tokens < 5000  # measured from the real prompt pieces
    assert est.worst_usd > est.max_usd  # max_tokens completion on every call


def test_estimate_unavailable_without_history_or_pricing(tmp_path, multi_issue_scenario) -> None:
    est = estimate_run_cost(
        _specs(multi_issue_scenario),
        agent="llm",
        max_rounds=None,
        settings=_settings(),
        results_dir=tmp_path,
        fetch=_Fetch(),
    )
    assert est is not None and est.usd_per_call is None and est.max_usd is None


def test_network_failures_are_fail_soft(tmp_path, multi_issue_scenario) -> None:
    est = estimate_run_cost(
        _specs(multi_issue_scenario),
        agent="a2a",
        max_rounds=None,
        settings=_settings(),
        results_dir=tmp_path,
        fetch=_Fetch(fail=True),
    )
    assert est is not None and est.max_usd is None and est.credit_remaining is None


def test_non_openrouter_endpoint_makes_no_network_calls(tmp_path, multi_issue_scenario) -> None:
    fetch = _Fetch(pricing={"prompt": "1", "completion": "1"}, remaining=1.0)
    est = estimate_run_cost(
        _specs(multi_issue_scenario),
        agent="a2a",
        max_rounds=None,
        settings=_settings(base_url="http://localhost:8000/v1"),
        results_dir=tmp_path,
        fetch=fetch,
    )
    assert fetch.urls == []
    assert est is not None and est.credit_remaining is None


def test_no_estimate_for_agents_without_paid_calls(multi_issue_scenario) -> None:
    for agent in ("scripted", "laya"):
        assert (
            estimate_run_cost(
                _specs(multi_issue_scenario), agent=agent, max_rounds=None, settings=_settings()
            )
            is None
        )
