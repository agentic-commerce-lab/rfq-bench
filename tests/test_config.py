"""Reusable config: parsing/validation and CLI precedence (flags > file > defaults)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rfq_bench.cli import app
from rfq_bench.config import BenchConfig, example_config_toml, load_config

runner = CliRunner()


def _write(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_load_config_parses_run_and_dashboard(tmp_path: Path) -> None:
    cfg = load_config(
        _write(
            tmp_path / "c.toml",
            '[run]\nstrategies = ["control", "boulware"]\nconcurrency = 4\n'
            '[dashboard]\ncontrol = "control"\nn_boot = 500\n',
        )
    )
    assert cfg.run.strategies == ["control", "boulware"]
    assert cfg.run.concurrency == 4
    assert cfg.run.opponents is None  # unspecified stays None
    assert cfg.dashboard.n_boot == 500


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="strategyz|extra"):
        load_config(_write(tmp_path / "c.toml", '[run]\nstrategyz = ["x"]\n'))


def test_invalid_toml_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid TOML"):
        load_config(_write(tmp_path / "c.toml", "[run]\nstrategies = [oops\n"))


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "absent.toml")


def test_example_config_is_valid() -> None:
    # The scaffold we hand users must itself parse and validate.
    import tomllib

    data = tomllib.loads(example_config_toml())
    BenchConfig.model_validate(data)


def _run(args: list[str]) -> str:
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    return result.output


def test_run_uses_config_file(tmp_path: Path) -> None:
    out = tmp_path / "traces.jsonl"
    cfg = tmp_path / "rfq-bench.toml"
    cfg.write_text(
        '[run]\nscenarios = ["single_price_a"]\nstrategies = ["control", "boulware"]\n'
        'opponents = ["hardliner"]\nseeds = [0]\nroles = ["buyer"]\n'
        f'first_speakers = ["buyer"]\nout = "{out}"\noverwrite = true\ndashboard = false\n',
        encoding="utf-8",
    )
    output = _run(["run", "--config", str(cfg), "--yes"])
    assert "using config" in output
    assert "strategies    : 2  (control, boulware)" in output
    ids = {json.loads(line)["scenario_id"] for line in out.open()}
    assert ids == {"single_price_a"}


def test_cli_flag_overrides_config(tmp_path: Path) -> None:
    out = tmp_path / "traces.jsonl"
    cfg = tmp_path / "rfq-bench.toml"
    cfg.write_text(
        '[run]\nscenarios = ["single_price_a"]\nstrategies = ["control", "boulware"]\n'
        'opponents = ["hardliner"]\nroles = ["buyer"]\nfirst_speakers = ["buyer"]\n'
        f'seeds = [0]\nout = "{out}"\noverwrite = true\ndashboard = false\n',
        encoding="utf-8",
    )
    # CLI --strategies must win over the file's two strategies.
    output = _run(["run", "--config", str(cfg), "--strategies", "control", "--yes"])
    assert "strategies    : 1  (control)" in output
    strategies = {json.loads(line)["strategy"] for line in out.open()}
    assert strategies == {"control"}


def test_auto_discovery_in_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "traces.jsonl"
    data_dir = Path("data/scenarios").resolve()  # absolute: the test chdirs away from the repo
    (tmp_path / "rfq-bench.toml").write_text(
        f'[run]\ndata = "{data_dir}"\nscenarios = ["single_price_a"]\nstrategies = ["control"]\n'
        'opponents = ["hardliner"]\nroles = ["buyer"]\nfirst_speakers = ["buyer"]\n'
        f'seeds = [0]\nout = "{out}"\noverwrite = true\ndashboard = false\n',
        encoding="utf-8",
    )
    # Restore the real auto-discovery that the hermetic conftest fixture disables.
    import rfq_bench.cli as cli
    from rfq_bench.config import discover_config as real_discover

    monkeypatch.setattr(cli, "discover_config", real_discover)
    monkeypatch.chdir(tmp_path)
    output = _run(["run", "--yes"])  # no --config: should auto-discover
    assert "auto-discovered config" in output
    assert out.exists()


def test_config_init_writes_and_refuses_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "rfq-bench.toml"
    _run(["config", "init", "--out", str(target)])
    assert target.exists() and "[run]" in target.read_text()
    # Second time without --force fails.
    result = runner.invoke(app, ["config", "init", "--out", str(target)])
    assert result.exit_code == 1
    assert "already exists" in result.output
    # --force overwrites.
    _run(["config", "init", "--out", str(target), "--force"])


def test_config_show_reports_settings(tmp_path: Path) -> None:
    cfg = tmp_path / "c.toml"
    cfg.write_text("[run]\nconcurrency = 8\n", encoding="utf-8")
    output = _run(["config", "show", "--config", str(cfg)])
    assert "concurrency = 8" in output


def test_max_rounds_from_config_and_cli_override(tmp_path: Path) -> None:
    out = tmp_path / "traces.jsonl"
    cfg = tmp_path / "c.toml"
    cfg.write_text("[run]\nmax_rounds = 5\n", encoding="utf-8")
    base = [
        "run",
        "--config",
        str(cfg),
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
    # From config.
    assert "max-rounds    : 5" in _run(base)
    max_round_5 = max(json.loads(line)["steps"][-1]["round"] for line in out.open())
    assert max_round_5 <= 4
    # CLI overrides config.
    assert "max-rounds    : 2" in _run([*base, "--max-rounds", "2"])
    max_round_2 = max(json.loads(line)["steps"][-1]["round"] for line in out.open())
    assert max_round_2 <= 1
