"""rfq-bench command line: run episodes, score traces, print reports."""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

import click
import typer

from rfq_bench.agent.personas import PERSONAS, load_persona_guidance, persona_instruction
from rfq_bench.agent.prompts import load_strategy_guidance, strategy_instruction
from rfq_bench.config import (
    DEFAULT_CONFIG_NAME,
    BenchConfig,
    discover_config,
    example_config_toml,
    load_config,
)
from rfq_bench.core.contracts import Role, Scenario, Trace
from rfq_bench.datasets import load_scenarios
from rfq_bench.opponents import OPPONENTS
from rfq_bench.report import build_dashboard, build_report
from rfq_bench.runners.offline import (
    EpisodeSpec,
    build_a2a_matrix,
    build_duel_matrix,
    build_matrix,
    run_episode,
)
from rfq_bench.store import TraceWriter, read_traces
from rfq_bench.strategies import STRATEGIES

app = typer.Typer(add_completion=False, help="Within-agent negotiation benchmark.")

_DEFAULT_DATA = "data/scenarios"
_DEFAULT_OUT = "results/offline_v0.jsonl"
# A2A self-play writes to its own file so the immutable offline suite is never mixed in.
_DEFAULT_A2A_OUT = "results/a2a_v0.jsonl"
# Laya (local decision model) writes to its own file too.
_DEFAULT_LAYA_OUT = "results/laya_v0.jsonl"
_DEFAULT_DUEL_OUT = "results/duel_v0.jsonl"
_DEFAULT_DUEL_DASHBOARD = "results/duel_dashboard.html"
_AGENT_KINDS = ("scripted", "llm", "a2a", "laya", "duel")
# Default ramp between episode starts for concurrent runs, so a large --concurrency
# doesn't hit the provider with N simultaneous first-requests. Auto-applied only when
# concurrency > 1; override with --stagger, disable with --stagger 0.
_DEFAULT_STAGGER_S = 0.5
# Base backoff between episode retries; multiplied by the attempt number.
_RETRY_BACKOFF_S = 1.0


def _scenario_index(scenarios: list[Scenario]) -> dict[str, Scenario]:
    return {s.id: s for s in scenarios}


def _write_dashboard_file(
    *,
    traces: str,
    data: str,
    out: str,
    control: str = "control",
    buyer_control: str = "neutral",
    n_boot: int = 2000,
    seed: int = 0,
) -> int:
    """Build the self-contained dashboard from ``traces`` and write it to ``out``.

    Returns the number of episodes embedded. The dashboard inlines the data at
    build time, so it is a point-in-time snapshot: rebuild it whenever the trace
    file changes (``run`` does this automatically).
    """
    scenarios = _scenario_index(load_scenarios(data))
    all_traces = list(read_traces(traces))
    if not all_traces:
        typer.echo("no traces found", err=True)
        raise typer.Exit(1)
    if any(t.mode == "duel" for t in all_traces):
        return _write_duel_dashboard(
            all_traces, scenarios, traces=traces, out=out, n_boot=n_boot, seed=seed
        )
    html = build_dashboard(
        all_traces,
        scenarios,
        control=control,
        buyer_control=buyer_control,
        n_boot=n_boot,
        seed=seed,
        source=traces,
    )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return len(all_traces)


def _write_duel_dashboard(
    all_traces: list[Trace],
    scenarios: dict[str, Scenario],
    *,
    traces: str,
    out: str,
    n_boot: int,
    seed: int,
) -> int:
    """Write the model-duel dashboard (the compare template with duel sections)."""
    from datetime import UTC, datetime

    from rfq_bench.report.duel import build_duel, duel_payload, duel_traces, render_duel_html

    duel = duel_traces(all_traces)
    try:
        result = build_duel(duel, scenarios, n_boot=n_boot, seed=seed)
    except ValueError as exc:
        typer.echo(f"duel dashboard not written: {exc}", err=True)
        return 0
    payload = duel_payload(result, duel, scenarios, source=traces)
    payload["generated"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_duel_html(payload), encoding="utf-8")
    return len(duel)


def _csv(value: str | None, default: tuple[str, ...]) -> list[str]:
    if not value:
        return list(default)
    return [v.strip() for v in value.split(",") if v.strip()]


def _select_scenarios(value: str | None, scenarios: list[Scenario]) -> list[Scenario]:
    """Filter loaded scenarios to a comma-separated id subset, preserving order.

    Unknown ids are rejected with the list of available ones, so a typo fails
    loudly instead of silently running an empty or wrong matrix.
    """
    if not value:
        return scenarios
    wanted = [v.strip() for v in value.split(",") if v.strip()]
    by_id = {s.id: s for s in scenarios}
    bad = [w for w in wanted if w not in by_id]
    if bad:
        available = ", ".join(by_id) or "<none>"
        raise typer.BadParameter(f"unknown scenario(s): {', '.join(bad)}. Available: {available}")
    # De-duplicate while preserving the user's order.
    seen: dict[str, None] = {}
    for w in wanted:
        seen.setdefault(w, None)
    return [by_id[w] for w in seen]


def _roles(value: str | None, default: tuple[Role, ...], what: str) -> tuple[Role, ...]:
    if not value:
        return default
    picked = [v.strip().lower() for v in value.split(",") if v.strip()]
    bad = [p for p in picked if p not in ("buyer", "seller")]
    if bad:
        raise typer.BadParameter(f"{what} must be 'buyer' and/or 'seller', got: {', '.join(bad)}")
    # De-duplicate while preserving order; picked is validated to buyer/seller above.
    seen: dict[Role, None] = {}
    for p in picked:
        seen.setdefault(cast(Role, p), None)
    return tuple(seen)


def _fmt(items: list[str], cap: int = 10) -> str:
    shown = items[:cap]
    suffix = "" if len(items) <= cap else f", … (+{len(items) - cap})"
    return ", ".join(shown) + suffix


def _hms(seconds: float) -> str:
    """Compact duration: '45s', '9m03s', '2h07m'."""
    total = int(round(max(0.0, seconds)))
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _print_cost_estimate(
    specs: list[EpisodeSpec],
    *,
    agent: str,
    max_rounds: int | None,
    models: tuple[str, str] | None = None,
) -> None:
    """Print the model-spend ceiling for LLM runs (see agent.cost_estimate).

    Uses the model set in .env / the environment. Never blocks the run: any
    failure prints "unavailable" and the run proceeds to the confirmation.
    """
    if agent not in ("llm", "a2a", "duel"):
        return
    from rfq_bench.agent.cost_estimate import estimate_run_cost, split_estimate
    from rfq_bench.agent.settings import AgentSettings

    try:
        if agent == "duel" and models is not None:
            base = AgentSettings()
            # Each model makes half the calls (one side per episode, both pairings).
            est = split_estimate(
                [
                    estimate_run_cost(
                        specs,
                        agent="a2a",
                        max_rounds=max_rounds,
                        settings=base.model_copy(update={"model": m}),
                    )
                    for m in models
                ]
            )
        else:
            est = estimate_run_cost(
                specs, agent=agent, max_rounds=max_rounds, settings=AgentSettings()
            )
    except Exception as exc:  # an estimate must never block a run
        typer.echo(f"  cost estimate : unavailable ({type(exc).__name__}: {exc})")
        return
    if est is None:
        return
    pad = " " * 18
    if est.max_usd is None:
        typer.echo(f"  cost estimate : unavailable for {est.model} ({est.basis})")
    else:
        typer.secho(
            f"  cost estimate : up to ${est.max_usd:,.2f}  ({est.model}; {est.calls_max:,} "
            "calls if every episode runs to its deadline)",
            bold=True,
        )
        typer.echo(f"{pad}basis: {est.basis}")
        if est.worst_usd is not None:
            typer.echo(
                f"{pad}if it is a reasoning model using all max_tokens on every call: "
                f"up to ${est.worst_usd:,.2f}"
            )
        typer.echo(
            f"{pad}agreements end episodes early, so actual spend is usually lower; "
            "re-asks and transient retries come on top"
        )
    if est.credit_remaining is not None:
        typer.echo(
            f"  credit        : ${est.credit_remaining:,.2f} remaining on this OpenRouter key"
        )
        if est.max_usd is not None and est.max_usd > est.credit_remaining:
            typer.secho(
                "  WARNING: the cost ceiling exceeds the remaining credit; "
                "the run may stop partway.",
                fg="yellow",
                bold=True,
            )


def _print_time_estimate(
    specs: list[EpisodeSpec],
    *,
    agent: str,
    max_rounds: int | None,
    concurrency: int,
    stagger: float,
    models: tuple[str, str] | None = None,
) -> None:
    """Print the wall-clock estimate for model-calling runs (see agent.time_estimate).

    History-based only; never blocks the run.
    """
    if agent not in ("llm", "a2a", "laya", "duel"):
        return
    from rfq_bench.agent.time_estimate import estimate_run_time, split_time_estimate

    try:
        if agent == "duel" and models is not None:
            est = split_time_estimate(
                [
                    estimate_run_time(
                        specs,
                        agent="a2a",
                        model=m,
                        max_rounds=max_rounds,
                        concurrency=concurrency,
                        stagger_s=stagger,
                    )
                    for m in models
                ]
            )
        else:
            if agent == "laya":
                from rfq_bench.agent.laya_settings import LayaSettings

                model = LayaSettings().to_llm_config().model
            else:
                from rfq_bench.agent.settings import AgentSettings

                model = AgentSettings().model
            est = estimate_run_time(
                specs,
                agent=agent,
                model=model,
                max_rounds=max_rounds,
                concurrency=concurrency,
                stagger_s=stagger,
            )
    except Exception as exc:  # an estimate must never block a run
        typer.echo(f"  time estimate : unavailable ({type(exc).__name__}: {exc})")
        return
    if est is None:
        typer.echo("  time estimate : unavailable (no past runs in results/ to measure from)")
        return
    pad = " " * 18
    if _hms(est.typical_s) == _hms(est.ceiling_s):
        headline = (
            f"up to {_hms(est.ceiling_s)} if every episode runs to its deadline "
            f"(-j {concurrency}; no history for these scenarios to shorten it)"
        )
    else:
        headline = (
            f"~{_hms(est.typical_s)} typical, up to {_hms(est.ceiling_s)} "
            f"if every episode runs to its deadline (-j {concurrency})"
        )
    typer.secho(f"  time estimate : {headline}", bold=True)
    typer.echo(f"{pad}basis: {est.basis}")
    if not est.same_model:
        typer.echo(f"{pad}model speed varies widely — treat this as an order of magnitude")
    typer.echo(f"{pad}re-asks, retries and rate limits come on top; live ETA shows once running")


def _print_run_plan(
    specs: list[EpisodeSpec],
    *,
    scenarios: list[Scenario],
    strategies: list[str],
    opponents: list[str],
    seeds: list[int],
    roles: tuple[Role, ...],
    first_speakers: tuple[Role, ...],
    agent: str,
    out_path: Path,
    overwrite: bool,
    concurrency: int,
    stagger: float = 0.0,
    max_rounds: int | None = None,
    personas: list[str] | None = None,
    models: tuple[str, str] | None = None,
    pairings: str = "cross",
) -> None:
    duel = agent == "duel"
    a2a = agent in ("a2a", "duel")
    typer.echo("Run plan")
    typer.echo(
        f"  agent         : {agent}"
        + (
            "  (model duel: the MODEL is the treatment; each cell played once per pairing)"
            if duel
            else "  (scored: LLM seller strategy; opponent: LLM buyer persona)"
            if a2a
            else ""
        )
    )
    if duel and models is not None:
        a, b = models
        typer.echo(f"  models        : A = {a}  ·  B = {b}")
        pairs = "A→B, B→A" + (", A→A, B→B (self-play)" if pairings == "full" else "")
        typer.echo(f"  pairings      : {pairings}  (seller→buyer: {pairs})")
    if max_rounds is not None:
        typer.echo(f"  max-rounds    : {max_rounds} (capped; changes outcomes)")
    if concurrency > 1:
        ramp = f", staggered {stagger:g}s apart" if stagger > 0 else ""
        typer.echo(f"  concurrency   : {concurrency} episodes in parallel{ramp}")
    typer.echo(f"  scenarios     : {len(scenarios)}  ({_fmt([s.id for s in scenarios])})")
    if a2a:
        ps = personas or []
        typer.echo(f"  strategies(sell): {len(strategies)}  ({_fmt(strategies)})")
        typer.echo(f"  personas(opp) : {len(ps)}  ({_fmt(ps)})")
    else:
        typer.echo(f"  strategies    : {len(strategies)}  ({_fmt(strategies)})")
        typer.echo(f"  opponents     : {len(opponents)}  ({_fmt(opponents)})")
        typer.echo(f"  roles         : {_fmt(list(roles))}")
    typer.echo(f"  first-speakers: {_fmt(list(first_speakers))}")
    typer.echo(f"  seeds         : {len(seeds)}  ({_fmt([str(s) for s in seeds])})")
    typer.echo(f"  output        : {out_path}  ({'overwrite' if overwrite else 'append'})")

    if a2a:
        naive = (
            len(scenarios)
            * len(personas or [])
            * len(strategies)
            * len(first_speakers)
            * len(seeds)
            * ((4 if pairings == "full" else 2) if duel else 1)
        )
    else:
        naive = (
            len(scenarios)
            * len(strategies)
            * len(opponents)
            * len(roles)
            * len(first_speakers)
            * len(seeds)
        )
    skipped = naive - len(specs)
    note = f"   ({skipped} cells skipped: logrolling is multi-issue only)" if skipped else ""
    typer.secho(f"  => {len(specs)} episodes{note}", bold=True)
    if agent == "llm":
        typer.echo("  note: --agent llm makes a live API call per agent turn.")
    if agent == "laya":
        typer.echo(
            "  note: --agent laya calls the local Laya server once per turn "
            "(deterministic, no API cost; requires the server running)."
        )
    if a2a:
        typer.secho(
            f"  note: --agent {agent} makes TWO live API calls per round (buyer + seller); "
            "cost roughly doubles vs --agent llm.",
            fg="yellow",
        )


@app.command()
def run(
    ctx: typer.Context,
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="TOML config file with reusable run settings "
        f"(auto-discovers ./{DEFAULT_CONFIG_NAME}). "
        "Explicit CLI flags override the file; the file overrides built-in defaults.",
    ),
    data: str = typer.Option(_DEFAULT_DATA, help="Scenario dataset directory."),
    out: str = typer.Option(_DEFAULT_OUT, help="Output JSONL trace file (append-only)."),
    scenarios: str | None = typer.Option(
        None, help="Comma-separated scenario id subset (default: all in the dataset)."
    ),
    strategies: str | None = typer.Option(
        None, help="Comma-separated strategy subset (seller side in --agent a2a)."
    ),
    opponents: str | None = typer.Option(None, help="Comma-separated opponent subset."),
    personas: str | None = typer.Option(
        None,
        help="Comma-separated buyer persona subset (--agent a2a: default all; "
        "--agent duel: default neutral).",
    ),
    models: str | None = typer.Option(
        None,
        help="--agent duel only: the two models, 'A,B' (OpenRouter-style ids on the "
        "configured endpoint). Everything else (decoding, prompts, tool_choice) is shared.",
    ),
    pairings: str | None = typer.Option(
        None,
        help="--agent duel only: 'cross' (A sells to B, B sells to A; default) or 'full' "
        "(plus both self-play pairs — separates seller from buyer skill, 2x cost).",
    ),
    seeds: str = typer.Option("0", help="Comma-separated integer seeds."),
    agent: str = typer.Option(
        "scripted",
        help="Agent kind: 'scripted', 'llm' (LLM vs scripted), 'a2a' "
        "(LLM buyer persona vs LLM seller strategy), 'laya', or 'duel' "
        "(two models against each other in both roles; needs --models).",
    ),
    roles: str | None = typer.Option(
        None,
        help="Which role(s) the tested agent plays: 'buyer', 'seller', or 'buyer,seller' "
        "(default: both). Use 'seller' to pin an --agent llm to the seller side.",
    ),
    first_speakers: str | None = typer.Option(
        None, help="Who opens: 'buyer', 'seller', or both (default: both)."
    ),
    max_rounds: int | None = typer.Option(
        None,
        "--max-rounds",
        min=1,
        help="Cap rounds per episode (default: each scenario's own deadline). Fewer rounds "
        "means fewer LLM calls, but it changes outcomes — not comparable to full-deadline runs.",
    ),
    retries: int = typer.Option(
        2,
        "--retries",
        min=0,
        help="Re-attempts for an episode that errors (e.g. a transient API/network failure), "
        "with a short backoff. 0 disables. Recovered and still-failed counts are reported.",
    ),
    overwrite: bool = typer.Option(False, help="Truncate the output file before writing."),
    concurrency: int = typer.Option(
        1,
        "--concurrency",
        "-j",
        min=1,
        help="Episodes to run in parallel. Episodes are independent, so >1 speeds up "
        "--agent llm (I/O-bound) roughly linearly until the provider rate-limits.",
    ),
    stagger: float | None = typer.Option(
        None,
        "--stagger",
        min=0.0,
        help="Minimum seconds between episode starts, ramping up parallel workers so a "
        "large --concurrency doesn't fire N requests at the same instant (rate limits). "
        f"Default: {_DEFAULT_STAGGER_S}s when --concurrency > 1, off otherwise. "
        "Pass --stagger 0 to start all workers immediately.",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the confirmation prompt (for non-interactive runs)."
    ),
    dashboard: bool = typer.Option(
        True,
        "--dashboard/--no-dashboard",
        help="After writing traces, (re)build results/dashboard.html so it matches this run.",
    ),
    dashboard_out: str = typer.Option(
        "results/dashboard.html", help="Where to write the refreshed dashboard."
    ),
) -> None:
    """Run the offline SAO matrix and append immutable traces."""
    # Merge a reusable config file underneath the CLI: a flag the user actually typed
    # wins; anything left at its default is filled from the file (if any).
    cfg_path = discover_config(config)
    if cfg_path is not None:
        try:
            cfg = load_config(cfg_path)
        except (FileNotFoundError, ValueError) as exc:
            typer.echo(f"config error: {exc}", err=True)
            raise typer.Exit(1) from exc
    else:
        cfg = BenchConfig()
    if cfg_path is not None and config is not None:
        typer.echo(f"using config: {cfg_path}")
    elif cfg_path is not None:
        typer.echo(f"using auto-discovered config: {cfg_path}")

    def _typed(name: str) -> bool:
        # True when the user passed this option on the command line (not a default).
        # Compare by enum name to stay robust across typer's/click's ParameterSource.
        source = ctx.get_parameter_source(name)
        return source is not None and source.name == "COMMANDLINE"

    def _csv_of(values: list[str] | list[int] | None) -> str | None:
        return ",".join(str(v) for v in values) if values is not None else None

    rc = cfg.run
    # String/CSV options: fill from config only when the user did not type them.
    data = data if _typed("data") else (rc.data or data)
    out = out if _typed("out") else (rc.out or out)
    if not _typed("scenarios") and rc.scenarios is not None:
        scenarios = _csv_of(rc.scenarios)
    if not _typed("strategies") and rc.strategies is not None:
        strategies = _csv_of(rc.strategies)
    if not _typed("opponents") and rc.opponents is not None:
        opponents = _csv_of(rc.opponents)
    if not _typed("personas") and rc.personas is not None:
        personas = _csv_of(rc.personas)
    if not _typed("models") and rc.models is not None:
        models = _csv_of(rc.models)
    if not _typed("pairings") and rc.pairings is not None:
        pairings = rc.pairings
    if not _typed("seeds") and rc.seeds is not None:
        seeds = _csv_of(rc.seeds) or seeds
    if not _typed("roles") and rc.roles is not None:
        roles = _csv_of(rc.roles)
    if not _typed("first_speakers") and rc.first_speakers is not None:
        first_speakers = _csv_of(rc.first_speakers)
    if not _typed("max_rounds") and rc.max_rounds is not None:
        max_rounds = rc.max_rounds
    if not _typed("retries") and rc.retries is not None:
        retries = rc.retries
    agent = agent if _typed("agent") else (rc.agent or agent)
    # Scalar options with concrete defaults: use config only if the flag was untyped.
    if not _typed("overwrite") and rc.overwrite is not None:
        overwrite = rc.overwrite
    if not _typed("concurrency") and rc.concurrency is not None:
        concurrency = rc.concurrency
    if not _typed("stagger") and rc.stagger is not None:
        stagger = rc.stagger
    if not _typed("dashboard") and rc.dashboard is not None:
        dashboard = rc.dashboard
    if not _typed("dashboard_out") and rc.dashboard_out is not None:
        dashboard_out = rc.dashboard_out

    # Resolve the auto stagger: on by default for concurrent runs, off for serial ones.
    stagger_s = (_DEFAULT_STAGGER_S if concurrency > 1 else 0.0) if stagger is None else stagger
    if agent not in _AGENT_KINDS:
        raise typer.BadParameter(f"agent must be one of: {', '.join(_AGENT_KINDS)}")
    # A2A self-play / Laya default to their own output files so they never mix with
    # the immutable offline suite (unless the user pins --out explicitly).
    if agent == "a2a" and not _typed("out") and rc.out is None:
        out = _DEFAULT_A2A_OUT
    if agent == "laya" and not _typed("out") and rc.out is None:
        out = _DEFAULT_LAYA_OUT
    if agent == "duel":
        if not _typed("out") and rc.out is None:
            out = _DEFAULT_DUEL_OUT
        if not _typed("dashboard_out") and rc.dashboard_out is None:
            dashboard_out = _DEFAULT_DUEL_DASHBOARD
        # The model is the treatment: hold strategy and persona at their baselines
        # unless the user asks for more.
        strategies = strategies or "control"
        personas = personas or "neutral"
    duel_models: tuple[str, str] | None = None
    pairing_mode = pairings or "cross"
    if agent == "duel":
        names = [m.strip() for m in (models or "").split(",") if m.strip()]
        if len(names) != 2 or names[0] == names[1]:
            raise typer.BadParameter("--agent duel needs --models A,B with two different models")
        if pairing_mode not in ("cross", "full"):
            raise typer.BadParameter("--pairings must be 'cross' or 'full'")
        duel_models = (names[0], names[1])
    elif models or pairings:
        raise typer.BadParameter("--models/--pairings only apply to --agent duel")
    selected_scenarios = _select_scenarios(scenarios, load_scenarios(data))
    strat = _csv(strategies, STRATEGIES)
    opps = _csv(opponents, OPPONENTS)
    persona_list = _csv(personas, PERSONAS)
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    role_tuple = _roles(roles, ("buyer", "seller"), "roles")
    first_tuple = _roles(first_speakers, ("buyer", "seller"), "first-speakers")

    if agent == "duel":
        assert duel_models is not None
        specs = list(
            build_duel_matrix(
                selected_scenarios,
                models=duel_models,
                pairings=pairing_mode,
                personas=persona_list,
                strategies=strat,
                seeds=seed_list,
                first_speakers=first_tuple,
            )
        )
    elif agent == "a2a":
        specs = list(
            build_a2a_matrix(
                selected_scenarios,
                personas=persona_list,
                strategies=strat,
                seeds=seed_list,
                first_speakers=first_tuple,
            )
        )
    else:
        specs = list(
            build_matrix(
                selected_scenarios,
                strategies=strat,
                opponents=opps,
                seeds=seed_list,
                roles=role_tuple,
                first_speakers=first_tuple,
            )
        )

    out_path = Path(out)
    _print_run_plan(
        specs,
        scenarios=selected_scenarios,
        strategies=strat,
        opponents=opps,
        personas=persona_list,
        seeds=seed_list,
        roles=role_tuple,
        first_speakers=first_tuple,
        agent=agent,
        out_path=out_path,
        overwrite=overwrite,
        concurrency=concurrency,
        stagger=stagger_s,
        max_rounds=max_rounds,
        models=duel_models,
        pairings=pairing_mode,
    )
    if not specs:
        typer.echo("Nothing to run with the current settings.", err=True)
        raise typer.Exit(1)
    _print_cost_estimate(specs, agent=agent, max_rounds=max_rounds, models=duel_models)
    _print_time_estimate(
        specs,
        agent=agent,
        max_rounds=max_rounds,
        concurrency=concurrency,
        stagger=stagger_s,
        models=duel_models,
    )
    if not yes:
        typer.confirm(f"Run {len(specs)} episodes?", abort=True)

    if overwrite and out_path.exists():
        out_path.unlink()
    writer = TraceWriter(out_path)

    agent_factory: Callable[[EpisodeSpec], Any] | None = None
    opponent_factory: Callable[[EpisodeSpec], Any] | None = None
    llm_config = None
    llm_config_for: Callable[[EpisodeSpec], Any] | None = None
    if agent in ("llm", "a2a", "duel"):
        from rfq_bench.agent.llm_client import LLMClient
        from rfq_bench.agent.negotiator import LLMNegotiator
        from rfq_bench.agent.settings import AgentSettings

        settings = AgentSettings()
        client = LLMClient(settings)  # httpx-based; safe to share across worker threads
        llm_config = settings.to_llm_config()
        strict = settings.strict_tool
        seller_prompt = settings.system_prompt_for("seller")
        buyer_prompt = settings.system_prompt_for("buyer")
        # Effective guidance = built-ins + any prompts/{strategies,personas}/*.md.
        strat_guidance = load_strategy_guidance()
        persona_guidance = load_persona_guidance()
        if agent == "duel":
            # One client per model; decoding, prompts and tool_choice are shared, so
            # the model is the only thing that differs between pairings.
            assert duel_models is not None
            model_settings = {m: settings.model_copy(update={"model": m}) for m in duel_models}
            clients = {m: LLMClient(ms) for m, ms in model_settings.items()}
            configs = {m: ms.to_llm_config() for m, ms in model_settings.items()}
            llm_config_for = lambda spec: configs[cast(str, spec.seller_model)]  # noqa: E731
            agent_factory = lambda spec: LLMNegotiator(  # noqa: E731
                spec.strategy,
                clients[cast(str, spec.seller_model)],
                behavior_kind="strategy",
                strict=strict,
                system_prompt=seller_prompt,
                instruction=strategy_instruction(spec.strategy, strat_guidance),
                max_reasks=settings.max_reasks,
                message_channel=True,
            )
            opponent_factory = lambda spec: LLMNegotiator(  # noqa: E731
                spec.persona or "neutral",
                clients[cast(str, spec.buyer_model)],
                behavior_kind="persona",
                strict=strict,
                system_prompt=buyer_prompt,
                instruction=persona_instruction(spec.persona or "neutral", persona_guidance),
                max_reasks=settings.max_reasks,
                message_channel=True,
            )
        elif agent == "a2a":
            # Seller is the scored agent (its strategy is the treatment); the buyer
            # persona is the opponent/environment, on the standard opponent axis.
            agent_factory = lambda spec: LLMNegotiator(  # noqa: E731
                spec.strategy,
                client,
                behavior_kind="strategy",
                strict=strict,
                system_prompt=seller_prompt,
                instruction=strategy_instruction(spec.strategy, strat_guidance),
                max_reasks=settings.max_reasks,
                message_channel=True,  # both LLM sides can talk to each other
            )
            opponent_factory = lambda spec: LLMNegotiator(  # noqa: E731
                spec.persona or "neutral",
                client,
                behavior_kind="persona",
                strict=strict,
                system_prompt=buyer_prompt,
                instruction=persona_instruction(spec.persona or "neutral", persona_guidance),
                max_reasks=settings.max_reasks,
                message_channel=True,
            )
        else:
            agent_factory = lambda spec: LLMNegotiator(  # noqa: E731
                spec.strategy,
                client,
                behavior_kind="strategy",
                strict=strict,
                system_prompt=settings.system_prompt_for(spec.target_role),
                instruction=strategy_instruction(spec.strategy, strat_guidance),
                max_reasks=settings.max_reasks,
            )
    elif agent == "laya":
        from rfq_bench.agent.laya_client import LayaClient
        from rfq_bench.agent.laya_negotiator import LayaNegotiator
        from rfq_bench.agent.laya_settings import LayaSettings

        laya_settings = LayaSettings()
        laya_client = LayaClient(laya_settings)  # loopback HTTP; lock-guarded internally
        llm_config = laya_settings.to_llm_config()
        accept_thr = laya_settings.accept_threshold
        walk_thr = laya_settings.walk_threshold
        laya_guidance = load_strategy_guidance()  # built-ins + prompts/strategies/*.md
        # Laya plays the tested (target) side and picks its own offer; the strategy is
        # injected as guidance text it reads. The opponent stays scripted.
        agent_factory = lambda spec: LayaNegotiator(  # noqa: E731
            laya_client,
            strategy=spec.strategy,
            instruction=strategy_instruction(spec.strategy, laya_guidance),
            accept_threshold=accept_thr,
            walk_threshold=walk_thr,
        )

    start_lock = threading.Lock()
    next_start = 0.0

    def throttle_start() -> None:
        # Space out episode *starts* by at least `stagger` seconds so a large -j does
        # not fire N requests at the same instant (a burst that trips rate limits).
        nonlocal next_start
        if stagger_s <= 0:
            return
        with start_lock:
            now = time.perf_counter()
            wait = max(0.0, next_start - now)
            next_start = max(now, next_start) + stagger_s
        if wait > 0:
            time.sleep(wait)

    def run_one(spec: EpisodeSpec) -> Trace:
        # A fresh policy per episode keeps workers independent; the LLM client is
        # shared and thread-safe. Episodes are deterministic given their spec for
        # scripted policies; with LLM sides, seeds/temperature govern reproducibility.
        throttle_start()
        policy = agent_factory(spec) if agent_factory else None
        opp = opponent_factory(spec) if opponent_factory else None
        return run_episode(
            spec,
            agent_policy=policy,
            opponent_policy=opp,
            llm_config=llm_config_for(spec) if llm_config_for else llm_config,
            max_rounds=max_rounds,
        )

    stats_lock = threading.Lock()
    retry_count = 0  # total re-attempts made
    recovered = 0  # episodes that errored at least once but ultimately succeeded

    def run_with_retries(spec: EpisodeSpec) -> Trace:
        nonlocal retry_count, recovered
        for attempt in range(retries + 1):
            try:
                trace = run_one(spec)
            except Exception:
                if attempt >= retries:
                    raise  # exhausted; the caller records a final failure
                with stats_lock:
                    retry_count += 1
                time.sleep(_RETRY_BACKOFF_S * (attempt + 1))  # linear backoff
                continue
            if attempt > 0:
                with stats_lock:
                    recovered += 1
            return trace
        raise RuntimeError("unreachable")  # pragma: no cover

    n = 0
    failures: list[tuple[EpisodeSpec, Exception]] = []
    label = f"running {len(specs)} episodes"
    if concurrency > 1:
        label += f", {concurrency} in parallel"

    t0 = time.perf_counter()

    def progress_suffix(done: int) -> str:
        elapsed = time.perf_counter() - t0
        if done <= 0 or elapsed <= 0:
            return f"0/{len(specs)}"
        rate_per_min = done / elapsed * 60.0
        eta = (len(specs) - done) * (elapsed / done)
        return f"{done}/{len(specs)} · {rate_per_min:.1f} ep/min · ETA {_hms(eta)}"

    show = lambda s: s or ""  # noqa: E731  (progress suffix; None before first update)
    if concurrency == 1:
        with click.progressbar(length=len(specs), label=label, item_show_func=show) as bar:
            for spec in specs:
                try:
                    trace = run_with_retries(spec)
                except Exception as exc:
                    failures.append((spec, exc))
                else:
                    writer.append(trace)
                    n += 1
                bar.update(1, current_item=progress_suffix(n + len(failures)))
    else:
        # Independent episodes fan out across a thread pool (blocking network I/O).
        # Submitting all specs only queues them — the pool keeps at most `concurrency`
        # requests in flight at once; `--stagger` additionally ramps their start times.
        # A lock serializes the append so JSONL lines never interleave; trace order does
        # not affect scoring.
        write_lock = threading.Lock()
        done = 0
        with (
            click.progressbar(length=len(specs), label=label, item_show_func=show) as pbar,
            ThreadPoolExecutor(max_workers=concurrency) as pool,
        ):
            futures = {pool.submit(run_with_retries, spec): spec for spec in specs}
            for fut in as_completed(futures):
                spec = futures[fut]
                try:
                    trace = fut.result()
                except Exception as exc:
                    failures.append((spec, exc))
                else:
                    with write_lock:
                        writer.append(trace)
                        n += 1
                done += 1
                pbar.update(1, current_item=progress_suffix(done))

    typer.echo(f"wrote {n} traces to {out_path}")
    if retry_count:
        typer.echo(
            f"retries: {retry_count} re-attempt(s), {recovered} episode(s) recovered, "
            f"{len(failures)} still failed"
        )
    if failures:
        spec0, exc0 = failures[0]
        typer.secho(
            f"WARNING: {len(failures)}/{len(specs)} episode(s) failed after {retries} "
            f"retr{'y' if retries == 1 else 'ies'} "
            f"(e.g. {spec0.scenario.id}/{spec0.strategy}: {type(exc0).__name__}: {exc0})",
            fg="yellow",
            err=True,
        )

    if dashboard and out_path.exists() and out_path.stat().st_size > 0:
        # The dashboard embeds data at build time, so refresh it from the full
        # trace file now — this reflects appends, not just this run's episodes.
        dc = cfg.dashboard
        total = _write_dashboard_file(
            traces=str(out_path),
            data=data,
            out=dashboard_out,
            control=dc.control or "control",
            buyer_control=dc.buyer_control or "neutral",
            n_boot=dc.n_boot or 2000,
            seed=dc.seed or 0,
        )
        typer.echo(f"refreshed dashboard for {total} episodes -> {dashboard_out}")
    elif dashboard:
        typer.echo("dashboard NOT refreshed: no traces were written")
    else:
        typer.echo(
            f"dashboard NOT refreshed; run `rfq-bench dashboard --traces {out_path}` to update it"
        )


_SCORING_CONFIG_HELP = (
    "Read paths and scoring settings from a run config (the same TOML as `run --config`): "
    "traces from [run].out (or the agent's default output), data from [run].data, the "
    "dashboard file from [run].dashboard_out, control/buyer_control/n_boot/seed from "
    "[dashboard]. Explicit flags still win. Not auto-discovered."
)


def _scoring_options_from_config(
    ctx: typer.Context, config: str | None, **current: Any
) -> dict[str, Any]:
    """For ``report`` / ``dashboard``: fill untyped options from a run config.

    Mirrors ``run``'s precedence (typed flag > config > built-in default), so the
    same config that produced a trace file can re-score or re-render it. Only keys
    present in ``current`` are touched (``report`` has no ``out``).
    """
    if config is None:
        return current
    try:
        cfg = load_config(config)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"config error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"using config: {config}", err=True)

    def typed(name: str) -> bool:
        source = ctx.get_parameter_source(name)
        return source is not None and source.name == "COMMANDLINE"

    rc, dc = cfg.run, cfg.dashboard
    # Where `run` would have written the traces for this config.
    agent_default = {
        "a2a": _DEFAULT_A2A_OUT,
        "laya": _DEFAULT_LAYA_OUT,
        "duel": _DEFAULT_DUEL_OUT,
    }.get(rc.agent or "")
    from_config: dict[str, Any] = {
        "traces": rc.out or agent_default,
        "data": rc.data,
        "out": rc.dashboard_out,
        "control": dc.control,
        "buyer_control": dc.buyer_control,
        "n_boot": dc.n_boot,
        "seed": dc.seed,
    }
    merged = dict(current)
    for key, value in from_config.items():
        if key in merged and value is not None and not typed(key):
            merged[key] = value
    return merged


@app.command()
def report(
    ctx: typer.Context,
    config: str | None = typer.Option(None, "--config", "-c", help=_SCORING_CONFIG_HELP),
    traces: str = typer.Option(_DEFAULT_OUT, help="JSONL trace file to score."),
    data: str = typer.Option(_DEFAULT_DATA, help="Scenario dataset directory."),
    control: str = typer.Option("control", help="Control condition (seller strategy) name."),
    buyer_control: str = typer.Option(
        "neutral", help="A2A buyer-side control persona (Δ baseline for the buyer table)."
    ),
    n_boot: int = typer.Option(2000, help="Bootstrap resamples for CIs."),
    seed: int = typer.Option(0, help="Bootstrap RNG seed (for reproducible CIs)."),
    require_control: bool = typer.Option(
        False,
        "--require-control",
        help="Fail if the control arm is absent (guards a full benchmark) instead of "
        "reporting S_s with an undefined Δ.",
    ),
) -> None:
    """Score an immutable trace file and print the strategy-effects report.

    A2A self-play traces score the seller strategy as the treatment and the buyer
    persona as the opponent, so they flow through this same report unchanged.
    """
    opts = _scoring_options_from_config(
        ctx,
        config,
        traces=traces,
        data=data,
        control=control,
        buyer_control=buyer_control,
        n_boot=n_boot,
        seed=seed,
    )
    traces, data, control = opts["traces"], opts["data"], opts["control"]
    buyer_control, n_boot, seed = opts["buyer_control"], opts["n_boot"], opts["seed"]
    scenarios = _scenario_index(load_scenarios(data))
    all_traces = list(read_traces(traces))
    if not all_traces:
        typer.echo("no traces found", err=True)
        raise typer.Exit(1)
    duel_ts = [t for t in all_traces if t.mode == "duel"]
    if duel_ts:
        # A duel's treatment is the model, so the strategy table (which would pool
        # the pairings) is replaced by the duel readout.
        from rfq_bench.report.duel import build_duel

        try:
            typer.echo(build_duel(duel_ts, scenarios, n_boot=n_boot, seed=seed).render())
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from exc
        return
    try:
        rep = build_report(
            all_traces,
            scenarios,
            control=control,
            n_boot=n_boot,
            seed=seed,
            require_control=require_control,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(rep.render())
    # A2A: also score the buyer, grouped by persona (the opponent axis), so both
    # sides of the self-play are surfaced from the same traces.
    a2a_traces = [t for t in all_traces if t.mode == "a2a"]
    if a2a_traces:
        buyer_rep = build_report(
            a2a_traces,
            scenarios,
            control=buyer_control,
            n_boot=n_boot,
            seed=seed,
            role="buyer",
            group_key="opponent",
            arm_label="persona",
        )
        typer.echo("")
        typer.echo(buyer_rep.render())
    # Manipulation check: did the tested agent actually play its strategy?
    from rfq_bench.report.fidelity import build_fidelity_report

    typer.echo("")
    typer.echo(build_fidelity_report(all_traces, scenarios).render())
    # Prompt-cache hit rate and A2A message activity (only when the traces have them).
    from rfq_bench.report.usage import usage_summary

    for line in usage_summary(all_traces):
        typer.echo(line)


@app.command()
def compare(
    traces_a: str = typer.Argument(..., help="JSONL trace file for run A (the baseline)."),
    traces_b: str = typer.Argument(..., help="JSONL trace file for run B."),
    data: str = typer.Option(_DEFAULT_DATA, help="Scenario dataset directory."),
    label_a: str | None = typer.Option(None, help="Name for run A (default: its file name)."),
    label_b: str | None = typer.Option(None, help="Name for run B (default: its file name)."),
    n_boot: int = typer.Option(2000, help="Bootstrap resamples for CIs."),
    seed: int = typer.Option(0, help="Bootstrap RNG seed (for reproducible CIs)."),
    html: bool = typer.Option(
        True, "--html/--no-html", help="Also write the self-contained compare dashboard."
    ),
    html_out: str | None = typer.Option(
        None,
        "--html-out",
        help="Where to write the compare dashboard "
        "(default: results/compare_<label-a>_vs_<label-b>.html).",
    ),
) -> None:
    """Compare two runs of the same condition (e.g. two models) on matched cells.

    Reports B − A in S_s points with a scenario-clustered CI, per strategy and per
    opponent, plus the separate tracks side by side. Warns when anything other than
    the model differs between the runs (mode, deadline, decoding config).
    """
    from rfq_bench.report.compare import compare_runs

    scenarios = _scenario_index(load_scenarios(data))
    a, b = list(read_traces(traces_a)), list(read_traces(traces_b))
    for path, ts in ((traces_a, a), (traces_b, b)):
        if not ts:
            typer.echo(f"no traces found in {path}", err=True)
            raise typer.Exit(1)
    missing = sorted({t.scenario_id for t in a + b} - set(scenarios))
    if missing:
        typer.echo(
            f"scenario(s) not in {data}: {', '.join(missing)} — pass the dataset with --data",
            err=True,
        )
        raise typer.Exit(1)
    result = compare_runs(
        a,
        b,
        scenarios,
        label_a=label_a or Path(traces_a).stem,
        label_b=label_b or Path(traces_b).stem,
        n_boot=n_boot,
        seed=seed,
    )
    typer.echo(result.render())
    if html:
        from datetime import UTC, datetime

        from rfq_bench.report.compare import comparison_payload, render_compare_html

        payload = comparison_payload(result, source_a=traces_a, source_b=traces_b)
        payload["generated"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        name_a, name_b = (
            re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-") or "run"
            for label in (result.label_a, result.label_b)
        )
        dest = Path(html_out or f"results/compare_{name_a}_vs_{name_b}.html")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(render_compare_html(payload), encoding="utf-8")
        typer.echo("")
        typer.echo(f"wrote compare dashboard to {dest}")


@app.command()
def dashboard(
    ctx: typer.Context,
    config: str | None = typer.Option(None, "--config", "-c", help=_SCORING_CONFIG_HELP),
    traces: str = typer.Option(_DEFAULT_OUT, help="JSONL trace file to visualize."),
    data: str = typer.Option(_DEFAULT_DATA, help="Scenario dataset directory."),
    out: str = typer.Option(
        "results/dashboard.html", help="Output HTML file (self-contained, offline)."
    ),
    control: str = typer.Option("control", help="Control condition (seller strategy) name."),
    buyer_control: str = typer.Option(
        "neutral", help="A2A buyer-side control persona (Δ baseline in the buyer view)."
    ),
    n_boot: int = typer.Option(2000, help="Bootstrap resamples for in-browser CIs."),
    seed: int = typer.Option(0, help="Bootstrap RNG seed (for reproducible CIs)."),
) -> None:
    """Build a self-contained interactive HTML dashboard from a trace file."""
    opts = _scoring_options_from_config(
        ctx,
        config,
        traces=traces,
        data=data,
        out=out,
        control=control,
        buyer_control=buyer_control,
        n_boot=n_boot,
        seed=seed,
    )
    n = _write_dashboard_file(**opts)
    if n == 0:
        raise typer.Exit(1)
    typer.echo(f"wrote dashboard for {n} episodes to {opts['out']}")


@app.command("list")
def list_cmd(data: str = typer.Option(_DEFAULT_DATA, help="Scenario dataset directory.")) -> None:
    """List strategies, opponents, personas, and scenarios.

    Strategies and personas include any defined by markdown files in
    ``prompts/strategies/`` and ``prompts/personas/`` (a new name is marked ``*``).
    """

    def _mark(effective: dict[str, str], builtin: tuple[str, ...]) -> str:
        builtin_set = set(builtin)
        names = sorted(effective, key=lambda n: (n not in builtin_set, n))
        return ", ".join(f"{n}*" if n not in builtin_set else n for n in names)

    typer.echo("strategies: " + _mark(load_strategy_guidance(), STRATEGIES))
    typer.echo("opponents:  " + ", ".join(OPPONENTS))
    typer.echo("personas:   " + _mark(load_persona_guidance(), PERSONAS))
    typer.echo("(* = added by a markdown file in prompts/strategies|personas/)")
    try:
        scenarios = load_scenarios(data)
        typer.echo("scenarios:  " + ", ".join(f"{s.id}({s.kind.value})" for s in scenarios))
    except FileNotFoundError:
        typer.echo("scenarios:  <none found>")


@app.command()
def doctor(
    data: str = typer.Option(_DEFAULT_DATA, help="Scenario dataset directory."),
    scenario: str | None = typer.Option(None, help="Scenario id (default: first multi-issue)."),
    strategy: str = typer.Option("anchoring", help="Strategy instruction to include."),
    role: str = typer.Option("seller", help="Which role the agent plays ('buyer'/'seller')."),
) -> None:
    """Make ONE real LLM tool call and print the raw response — diagnose function calling."""
    from rfq_bench.agent.llm_client import LLMClient
    from rfq_bench.agent.prompts import build_move_tool, build_user_payload
    from rfq_bench.agent.settings import AgentSettings
    from rfq_bench.strategies.base import NegotiationState

    if role not in ("buyer", "seller"):
        raise typer.BadParameter("role must be 'buyer' or 'seller'")
    scenarios = load_scenarios(data)
    scn = (
        next((s for s in scenarios if s.id == scenario), None)
        if scenario
        else next((s for s in scenarios if len(s.issues) > 1), scenarios[0])
    )
    if scn is None:
        raise typer.BadParameter(f"scenario {scenario!r} not found")

    settings = AgentSettings()
    client = LLMClient(settings)
    state = NegotiationState(
        role=cast(Role, role),
        prefs=scn.preferences(cast(Role, role)),
        issues=list(scn.issues),
        round=0,
        deadline=scn.deadline_rounds,
    )
    tool = build_move_tool(state.issues, strict=settings.strict_tool)
    system_prompt = settings.system_prompt_for(cast(Role, role))
    typer.echo(f"scenario      : {scn.id}  (agent = {role})")
    typer.echo(f"model         : {settings.model}")
    typer.echo(f"base_url      : {settings.base_url}")
    typer.echo(f"strict_tool   : {settings.strict_tool}")
    typer.echo(f"tool_choice   : {settings.tool_choice}")
    first_line = system_prompt.splitlines()[0][:70]
    typer.echo(f"system prompt : {len(system_prompt)} chars (first line: {first_line}…)")
    typer.echo("calling the endpoint with the submit_move tool …")
    instruction = strategy_instruction(strategy, load_strategy_guidance())
    resp = client.complete(system_prompt, build_user_payload(state, instruction, None), tool)
    typer.echo(f"finish_reason : {resp.finish_reason}")
    typer.echo(f"tokens        : {resp.tokens}   cost_usd: {resp.cost_usd}")
    typer.secho(f"raw tool args : {resp.content!r}", bold=True)
    typer.echo(f"parsed        : {resp.data}")
    if resp.reasoning:
        typer.echo(f"reasoning[:400]: {resp.reasoning[:400]}")
    if not resp.data:
        typer.secho(
            "\nThe model returned EMPTY tool arguments. If finish_reason is 'tool_calls', the "
            "model called the function but filled nothing — try RFQ_BENCH_STRICT_TOOL=false, or "
            "share this output and the model id.",
            fg="yellow",
        )


config_app = typer.Typer(
    add_completion=False, help="Create and inspect reusable benchmark config files."
)
app.add_typer(config_app, name="config")


@config_app.command("init")
def config_init(
    out: str = typer.Option(
        DEFAULT_CONFIG_NAME, "--out", "-o", help="Where to write the example config."
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite an existing file."),
) -> None:
    """Write a commented example config you can edit and reuse with `run --config`."""
    path = Path(out)
    if path.exists() and not force:
        typer.echo(f"{path} already exists (use --force to overwrite)", err=True)
        raise typer.Exit(1)
    path.write_text(example_config_toml(), encoding="utf-8")
    typer.echo(f"wrote example config to {path}")
    typer.echo(f"edit it, then run: rfq-bench run --config {path}")


@config_app.command("show")
def config_show(
    config: str | None = typer.Option(
        None, "--config", "-c", help=f"Config file to inspect (default: ./{DEFAULT_CONFIG_NAME})."
    ),
) -> None:
    """Validate a config file and print the settings it specifies."""
    cfg_path = discover_config(config)
    if cfg_path is None:
        typer.echo(f"no config file (looked for ./{DEFAULT_CONFIG_NAME})", err=True)
        raise typer.Exit(1)
    try:
        cfg = load_config(cfg_path)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"config: {cfg_path}")
    run_set = cfg.run.model_dump(exclude_none=True)
    dash_set = cfg.dashboard.model_dump(exclude_none=True)
    if run_set:
        typer.echo("[run]")
        for k, v in run_set.items():
            typer.echo(f"  {k} = {v!r}")
    if dash_set:
        typer.echo("[dashboard]")
        for k, v in dash_set.items():
            typer.echo(f"  {k} = {v!r}")
    if not run_set and not dash_set:
        typer.echo("(empty — all defaults)")
    typer.echo("note: explicit CLI flags on `run` override these values.")


if __name__ == "__main__":
    app()
