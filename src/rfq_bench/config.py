"""Reusable benchmark configuration loaded from a TOML file.

A config file captures the matrix and run knobs you would otherwise retype on
every ``rfq-bench run`` invocation, so a benchmark is reproducible from a single
committed file. Precedence at run time is: **explicit CLI flags > config file >
built-in defaults** — the CLI always wins, the config only fills in what you did
not pass.

Reading uses the standard-library ``tomllib`` (Python 3.11+); no new dependency.
Every field is optional: a value left out of the file simply falls through to the
CLI default. Unknown keys are rejected so a typo fails loudly instead of being
silently ignored.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_CONFIG_NAME = "rfq-bench.toml"


class RunConfig(BaseModel):
    """The ``[run]`` table: what matrix to run and how to execute it."""

    model_config = ConfigDict(extra="forbid")

    data: str | None = None
    out: str | None = None
    scenarios: list[str] | None = None
    strategies: list[str] | None = None
    opponents: list[str] | None = None
    personas: list[str] | None = None
    seeds: list[int] | None = None
    agent: str | None = None
    roles: list[str] | None = None
    first_speakers: list[str] | None = None
    max_rounds: int | None = Field(default=None, ge=1)
    retries: int | None = Field(default=None, ge=0)
    overwrite: bool | None = None
    concurrency: int | None = Field(default=None, ge=1)
    stagger: float | None = Field(default=None, ge=0.0)
    dashboard: bool | None = None
    dashboard_out: str | None = None


class DashboardConfig(BaseModel):
    """The ``[dashboard]`` table: scoring knobs for the refreshed dashboard."""

    model_config = ConfigDict(extra="forbid")

    control: str | None = None
    buyer_control: str | None = None  # A2A buyer-side Δ baseline persona
    n_boot: int | None = Field(default=None, ge=1)
    seed: int | None = None


class BenchConfig(BaseModel):
    """Top-level config: ``[run]`` plus ``[dashboard]`` tables."""

    model_config = ConfigDict(extra="forbid")

    run: RunConfig = Field(default_factory=RunConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)


def load_config(path: str | Path) -> BenchConfig:
    """Parse and validate a TOML config file into a :class:`BenchConfig`.

    Raises ``FileNotFoundError`` if the path does not exist and ``ValueError``
    (with the offending detail) on malformed TOML or an invalid/unknown key.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    try:
        raw = tomllib.loads(p.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"invalid TOML in {p}: {exc}") from exc
    return BenchConfig.model_validate(raw)


def discover_config(explicit: str | Path | None) -> Path | None:
    """Resolve which config file to use.

    An explicit path is returned as-is (existence is checked by ``load_config``).
    Otherwise, an ``rfq-bench.toml`` in the current directory is auto-discovered;
    if none exists, ``None`` means "no config, use defaults".
    """
    if explicit is not None:
        return Path(explicit)
    default = Path(DEFAULT_CONFIG_NAME)
    return default if default.exists() else None


def example_config_toml() -> str:
    """A commented example config covering every supported key."""
    return """\
# rfq-bench configuration — reusable benchmark setup.
# Precedence at run time: explicit CLI flags > this file > built-in defaults.
# Every key is optional; omit a key to fall back to the CLI default.
# Auto-discovered as ./rfq-bench.toml, or pass --config <path>.

[run]
# Which matrix to run. Lists may be shortened to a subset; omit for "all".
# scenarios     = ["single_price_a", "multi_pdw_a"]
# strategies    = ["control", "boulware", "anchoring"]
# opponents     = ["hardliner", "reciprocal"]
# personas      = ["neutral", "bargain_hunter"]  # buyer side, --agent a2a only
# seeds         = [0, 1]
# roles         = ["buyer"]            # which side the tested agent plays
# first_speakers = ["buyer"]           # who opens

# Execution.
agent          = "scripted"    # "scripted", "llm", or "a2a" (LLM buyer persona vs LLM seller)
# max_rounds   = 6             # cap rounds/episode (fewer LLM calls; changes outcomes)
retries        = 2             # re-attempts for an episode that errors (transient API failures)
concurrency    = 1             # episodes in parallel (>1 speeds up --agent llm)
# stagger      = 0.5           # seconds between episode starts (auto 0.5s when concurrency > 1)
overwrite      = false         # truncate the output file before writing

# Output.
out            = "results/offline_v0.jsonl"
dashboard      = true                  # rebuild the dashboard after the run
dashboard_out  = "results/dashboard.html"

[dashboard]
# Scoring knobs used when the dashboard is refreshed after a run.
control        = "control"             # control condition for Δ vs control
n_boot         = 2000                  # bootstrap resamples for CIs
seed           = 0                     # bootstrap RNG seed
"""
