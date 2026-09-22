"""CLI wiring: `run` refreshes the dashboard so it never goes stale."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from click.testing import Result
from typer.testing import CliRunner

from rfq_bench.cli import _hms, app

runner = CliRunner()


def test_hms_formats_durations() -> None:
    assert _hms(0) == "0s"
    assert _hms(45) == "45s"
    assert _hms(543) == "9m03s"
    assert _hms(7620) == "2h07m"


def _embedded_n(html: str) -> int:
    m = re.search(r'"n_episodes":(\d+)', html)
    assert m, "no n_episodes in dashboard payload"
    return int(m.group(1))


def _run(args: list[str]) -> Result:
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    return result


def test_run_auto_refreshes_dashboard(tmp_path: Path) -> None:
    out = tmp_path / "traces.jsonl"
    dash = tmp_path / "dash.html"
    res = _run(
        [
            "run",
            "--strategies",
            "control",
            "--opponents",
            "hardliner",
            "--roles",
            "buyer",
            "--first-speakers",
            "buyer",
            "--seeds",
            "0",
            "--overwrite",
            "--yes",
            "--out",
            str(out),
            "--dashboard-out",
            str(dash),
        ]
    )
    n_traces = sum(1 for _ in out.open())
    assert dash.exists()
    assert _embedded_n(dash.read_text()) == n_traces
    assert f"refreshed dashboard for {n_traces} episodes" in res.output


def test_dashboard_reflects_appends(tmp_path: Path) -> None:
    out = tmp_path / "traces.jsonl"
    dash = tmp_path / "dash.html"
    base = [
        "run",
        "--opponents",
        "hardliner",
        "--roles",
        "buyer",
        "--first-speakers",
        "buyer",
        "--seeds",
        "0",
        "--yes",
        "--out",
        str(out),
        "--dashboard-out",
        str(dash),
    ]
    _run([*base, "--strategies", "control", "--overwrite"])
    first = _embedded_n(dash.read_text())
    _run([*base, "--strategies", "boulware"])  # append, no overwrite
    total = sum(1 for _ in out.open())
    assert total > first
    # The refreshed dashboard reflects the whole file, not just the last run.
    assert _embedded_n(dash.read_text()) == total


def test_scenarios_filter_limits_the_run(tmp_path: Path) -> None:
    out = tmp_path / "traces.jsonl"
    dash = tmp_path / "dash.html"
    res = _run(
        [
            "run",
            "--scenarios",
            "single_price_a",
            "--strategies",
            "control",
            "--opponents",
            "hardliner",
            "--roles",
            "buyer",
            "--first-speakers",
            "buyer",
            "--seeds",
            "0",
            "--overwrite",
            "--yes",
            "--out",
            str(out),
            "--dashboard-out",
            str(dash),
        ]
    )
    assert "scenarios     : 1  (single_price_a)" in res.output
    # Every trace is from the chosen scenario only.
    ids = {json.loads(line)["scenario_id"] for line in out.open()}
    assert ids == {"single_price_a"}


def test_unknown_scenario_is_rejected(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--scenarios",
            "does_not_exist",
            "--yes",
            "--out",
            str(tmp_path / "t.jsonl"),
            "--no-dashboard",
        ],
    )
    assert result.exit_code != 0
    assert "unknown scenario(s): does_not_exist" in result.output


def _trace_set(path: Path) -> set[str]:
    """Traces as a set of JSON strings, minus wall-clock latency (always varies)."""
    out = set()
    for line in path.open():
        d = json.loads(line)
        d.pop("latency_s", None)
        out.add(json.dumps(d, sort_keys=True))
    return out


def test_concurrency_matches_sequential(tmp_path: Path) -> None:
    seq = tmp_path / "seq.jsonl"
    par = tmp_path / "par.jsonl"
    base = [
        "run",
        "--strategies",
        "control,boulware",
        "--opponents",
        "hardliner",
        "--roles",
        "buyer",
        "--first-speakers",
        "buyer",
        "--seeds",
        "0",
        "--overwrite",
        "--yes",
        "--no-dashboard",
    ]
    _run([*base, "--out", str(seq)])  # j=1 default
    res = _run([*base, "--out", str(par), "-j", "4", "--stagger", "0"])  # 0 keeps the test fast
    assert "concurrency   : 4 episodes in parallel" in res.output
    # Deterministic scripted episodes: same trace content regardless of concurrency.
    assert _trace_set(seq) == _trace_set(par)


def test_stagger_defaults_on_for_concurrent_and_off_serial(tmp_path: Path) -> None:
    out = tmp_path / "t.jsonl"
    base = [
        "run",
        "--scenarios",
        "single_price_a",
        "--strategies",
        "control",
        "--opponents",
        "hardliner",
        "--roles",
        "buyer",
        "--first-speakers",
        "buyer",
        "--seeds",
        "0",
        "--overwrite",
        "--yes",
        "--no-dashboard",
        "--out",
        str(out),
    ]
    # Concurrent run: auto-on at the default ramp, no flag needed.
    assert "staggered 0.5s apart" in _run([*base, "-j", "3"]).output
    # Explicit opt-out.
    assert "staggered" not in _run([*base, "-j", "3", "--stagger", "0"]).output
    # Serial run: no concurrency/stagger line at all.
    assert "concurrency" not in _run(base).output


def test_stagger_shows_in_plan_and_keeps_content(tmp_path: Path) -> None:
    seq = tmp_path / "seq.jsonl"
    stg = tmp_path / "stg.jsonl"
    base = [
        "run",
        "--strategies",
        "control,boulware",
        "--opponents",
        "hardliner",
        "--roles",
        "buyer",
        "--first-speakers",
        "buyer",
        "--seeds",
        "0",
        "--overwrite",
        "--yes",
        "--no-dashboard",
    ]
    _run([*base, "--out", str(seq)])
    res = _run([*base, "--out", str(stg), "-j", "4", "--stagger", "0.01"])
    assert "staggered 0.01s apart" in res.output
    # Ramping start times must not change trace content.
    assert _trace_set(seq) == _trace_set(stg)


_ONE_EPISODE = [
    "run",
    "--scenarios",
    "single_price_a",
    "--strategies",
    "control",
    "--opponents",
    "hardliner",
    "--roles",
    "buyer",
    "--first-speakers",
    "buyer",
    "--seeds",
    "0",
    "--overwrite",
    "--yes",
    "--no-dashboard",
]


def test_retries_recover_transient_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import rfq_bench.cli as cli

    monkeypatch.setattr(cli, "_RETRY_BACKOFF_S", 0.0)  # no sleeping in tests
    real = cli.run_episode
    seen = {"n": 0}

    def flaky(spec, **kw):  # type: ignore[no-untyped-def]
        seen["n"] += 1
        if seen["n"] == 1:  # only the first attempt fails
            raise RuntimeError("transient boom")
        return real(spec, **kw)

    monkeypatch.setattr(cli, "run_episode", flaky)
    out = tmp_path / "t.jsonl"
    res = _run([*_ONE_EPISODE, "--retries", "2", "--out", str(out)])
    assert "1 episode(s) recovered" in res.output
    assert sum(1 for _ in out.open()) == 1  # the episode was ultimately written


def test_retries_exhausted_reports_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import rfq_bench.cli as cli

    monkeypatch.setattr(cli, "_RETRY_BACKOFF_S", 0.0)

    def always_fail(spec, **kw):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "run_episode", always_fail)
    out = tmp_path / "t.jsonl"
    res = _run([*_ONE_EPISODE, "--retries", "1", "--out", str(out)])
    assert "1 still failed" in res.output
    assert "failed after 1 retry" in res.output


def test_no_dashboard_flag_skips_build(tmp_path: Path) -> None:
    out = tmp_path / "traces.jsonl"
    dash = tmp_path / "dash.html"
    res = _run(
        [
            "run",
            "--strategies",
            "control",
            "--opponents",
            "hardliner",
            "--roles",
            "buyer",
            "--first-speakers",
            "buyer",
            "--seeds",
            "0",
            "--overwrite",
            "--yes",
            "--no-dashboard",
            "--out",
            str(out),
            "--dashboard-out",
            str(dash),
        ]
    )
    assert not dash.exists()
    assert "dashboard NOT refreshed" in res.output


def test_bad_agent_is_rejected() -> None:
    result = runner.invoke(app, ["run", "--agent", "nope", "--yes"])
    assert result.exit_code != 0
    assert "agent must be one of" in result.output


def test_a2a_run_plan_shows_personas_and_defaults_out() -> None:
    # Decline the confirm so no live API call is made; we only check the plan.
    result = runner.invoke(
        app,
        [
            "run",
            "--agent",
            "a2a",
            "--scenarios",
            "single_price_a",
            "--personas",
            "neutral,hardball",
            "--strategies",
            "control,boulware",
            "--seeds",
            "0",
        ],
        input="n\n",
    )
    assert result.exit_code != 0  # aborted at the confirm
    out = result.output
    assert "strategies(sell): 2" in out
    assert "personas(opp) : 2" in out
    assert "results/a2a_v0.jsonl" in out  # a2a default output file
    assert "TWO live API calls" in out  # cost warning
    # single-issue: no role mirror, 2 personas x 2 strategies x 2 first-speakers = 8
    assert "=> 8 episodes" in out
