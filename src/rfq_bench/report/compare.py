"""Cross-run comparison: the same benchmark condition, two trace files (e.g. two models).

This is a **between-run** comparison, distinct from the benchmark's within-agent
strategy effect Δ_s. It is only meaningful when everything except the thing being
compared (normally the model) is held fixed, so it:

- **matches cells**: a cell is (scenario, strategy, opponent, role, first speaker).
  Only cells scorable in *both* runs are compared; seeds are averaged within a cell
  per run, so the runs may have different repetition counts.
- **pairs** the difference per cell (B − A) and reports it macro-averaged over cells,
  exactly like S_s, with a 95% CI from a bootstrap that resamples **scenarios**.
- **checks the condition**: any difference in mode, outcome-space version,
  deadline, or frozen decoding config other than the model is reported as a
  warning, because it confounds the comparison. (The system prompt is not on the
  trace, so it cannot be checked here — keep it identical.)

Pure: a function of the two trace lists and the scenarios, like ``core.scoring``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TypeVar

import numpy as np

from rfq_bench.core.contracts import Scenario, Trace
from rfq_bench.core.scoring import score_trace
from rfq_bench.report.metrics import side_metrics

Cell = tuple[str, str, str, str, str]
K = TypeVar("K", bound=tuple[str, ...])  # scenario, strategy, opponent, role, first speaker

# Frozen LLMConfig fields that must match across the two runs (model may differ).
_CONFIG_FIELDS = ("temperature", "max_tokens", "seed", "tool_choice")


def _cell(t: Trace) -> Cell:
    return (t.scenario_id, t.strategy, t.opponent, t.target_role, t.first_speaker)


def _cell_means(traces: Iterable[Trace], scenarios: dict[str, Scenario]) -> dict[Cell, float]:
    """Mean q per cell over its scorable repetitions (errored/no-ZOPA/degenerate dropped)."""
    qs: dict[Cell, list[float]] = defaultdict(list)
    for t in traces:
        ep = score_trace(t, scenarios[t.scenario_id])
        if ep.scorable:
            assert ep.q is not None
            qs[_cell(t)].append(ep.q)
    return {c: float(np.mean(v)) for c, v in qs.items()}


@dataclass(frozen=True)
class PairedDiff:
    """B − A over matched cells, in S_s points (0–100)."""

    name: str
    score_a: float
    score_b: float
    delta: float
    ci_low: float
    ci_high: float
    n_cells: int
    n_scenarios: int


def paired_diff(
    name: str,
    a: Mapping[K, float],
    b: Mapping[K, float],
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> PairedDiff:
    """Macro-average over matched cells, with a scenario-clustered bootstrap CI.

    Cells are tuples whose first element is the scenario id (the cluster).
    """
    matched = sorted(set(a) & set(b))
    nan = float("nan")
    if not matched:
        return PairedDiff(name, nan, nan, nan, nan, nan, 0, 0)
    by_scn: dict[str, list[float]] = defaultdict(list)  # scenario -> per-cell B − A
    for c in matched:
        by_scn[c[0]].append(b[c] - a[c])
    score_a = 100.0 * float(np.mean([a[c] for c in matched]))
    score_b = 100.0 * float(np.mean([b[c] for c in matched]))
    delta = score_b - score_a
    scns = sorted(by_scn)
    if len(scns) < 2:
        # One cluster: resampling it only ever returns the point estimate.
        ci_low = ci_high = nan
    else:
        boot = np.empty(n_boot)
        for i in range(n_boot):
            picked = rng.choice(len(scns), size=len(scns), replace=True)
            diffs = [d for j in picked for d in by_scn[scns[j]]]
            boot[i] = 100.0 * float(np.mean(diffs))
        ci_low, ci_high = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    return PairedDiff(name, score_a, score_b, delta, ci_low, ci_high, len(matched), len(scns))


def _describe(traces: list[Trace]) -> str:
    models = sorted({t.llm_config.model for t in traces})
    return ", ".join(models)


def _config_value(t: Trace, *, field_name: str) -> object:
    return getattr(t.llm_config, field_name)


def condition_warnings(a: list[Trace], b: list[Trace]) -> list[str]:
    """Differences other than the model that confound an A-vs-B comparison."""
    warns: list[str] = []
    checks: dict[str, Callable[[Trace], object]] = {
        "mode": lambda t: t.mode,
        "outcome_space_version": lambda t: t.outcome_space_version,
    }
    for f in _CONFIG_FIELDS:
        checks[f] = partial(_config_value, field_name=f)
    for label, get in checks.items():
        va, vb = {str(get(t)) for t in a}, {str(get(t)) for t in b}
        if va != vb:
            warns.append(f"{label} differs: A={sorted(va)} B={sorted(vb)}")
    # Deadline per matched cell (a --max-rounds cap changes outcomes).
    dl_a = {_cell(t): t.deadline for t in a}
    dl_b = {_cell(t): t.deadline for t in b}
    diff = [c for c in set(dl_a) & set(dl_b) if dl_a[c] != dl_b[c]]
    if diff:
        warns.append(f"deadline differs on {len(diff)} matched cell(s) (a --max-rounds cap?)")
    if len({t.llm_config.model for t in a + b}) > 2:
        warns.append("a file mixes several models; results pool them per file")
    return warns


@dataclass(frozen=True)
class Comparison:
    label_a: str
    label_b: str
    models_a: str
    models_b: str
    overall: PairedDiff
    by_strategy: list[PairedDiff]
    by_opponent: list[PairedDiff]
    # Per scenario: no CI (a scenario is one cluster), so ci_low/ci_high are NaN.
    by_scenario: list[PairedDiff]
    # (strategy, opponent) -> B − A in points, averaged over matched scenarios.
    grid: dict[tuple[str, str], float]
    side_a: dict[str, float | None]
    side_b: dict[str, float | None]
    cells_only_a: int
    cells_only_b: int
    warnings: list[str] = field(default_factory=list)

    def render(self) -> str:
        a, b = self.label_a, self.label_b
        lines = [
            f"rfq-bench compare — A = {a} ({self.models_a})",
            f"                    B = {b} ({self.models_b})",
            "Cross-run comparison on matched cells (scenario × strategy × opponent × role × "
            "first speaker).",
            "Δ = B − A in S_s points; 95% CI clustered by scenario. This is NOT the "
            "within-agent strategy effect.",
        ]
        if self.cells_only_a or self.cells_only_b:
            lines.append(
                f"unmatched cells skipped: {self.cells_only_a} only in A, "
                f"{self.cells_only_b} only in B (or unscorable in the other run)"
            )
        for warning in self.warnings:
            lines.append(f"WARNING: {warning}")
        lines.append("")
        o = self.overall
        lines.append(
            f"Overall  S_A = {o.score_a:.1f}  S_B = {o.score_b:.1f}  "
            f"Δ = {_signed(o.delta)}  {_ci(o)}  ({o.n_cells} cells, {o.n_scenarios} scenarios)"
        )
        lines.append(f"  → {_verdict(o, a, b)}")
        for title, rows in (("strategy", self.by_strategy), ("opponent", self.by_opponent)):
            lines.append("")
            w = max([12, len(title) + 2, *(len(r.name) + 2 for r in rows)])
            header = f"{title:<{w}}{'S_A':>8}{'S_B':>8}{'Δ':>9}{'95% CI':>20}{'cells':>7}"
            lines.append(header)
            lines.append("-" * len(header))
            for r in rows:
                lines.append(
                    f"{r.name:<{w}}{r.score_a:>8.1f}{r.score_b:>8.1f}{_signed(r.delta):>9}"
                    f"{_ci(r):>20}{r.n_cells:>7}"
                )
        lines.append("")
        lines.append(
            "Separate tracks (episodes in cells both runs cover, incl. no-ZOPA and errored)"
        )
        header = f"{'metric':<22}{'A':>12}{'B':>12}"
        lines.append(header)
        lines.append("-" * len(header))
        for key, (label, fmt) in _SIDE_ROWS.items():
            lines.append(
                f"{label:<22}{_fmt(self.side_a.get(key), fmt):>12}"
                f"{_fmt(self.side_b.get(key), fmt):>12}"
            )
        return "\n".join(lines)


_SIDE_ROWS: dict[str, tuple[str, str]] = {
    "episodes": ("episodes", "int"),
    "error_rate": ("error rate", "pct"),
    "agreement_rate": ("agreement rate", "pct"),
    "walk_away_accuracy": ("no-ZOPA walk✓", "pct"),
    "pareto_rate": ("Pareto-efficient", "pct"),
    "mean_rounds": ("rounds to close", "f1"),
    "mean_latency_s": ("latency / episode", "sec"),
    "mean_cost_usd": ("$ / episode", "usd"),
    "mean_tokens": ("tokens / episode", "int"),
    "total_cost_usd": ("total $", "usd"),
}


def _fmt(v: float | None, kind: str) -> str:
    if v is None or v != v:
        return "—"
    if kind == "int":
        return f"{int(v):,}"
    if kind == "pct":
        return f"{v * 100:.0f}%"
    if kind == "sec":
        return f"{v:.1f}s"
    if kind == "usd":
        return f"${v:.4f}"
    return f"{v:.1f}"


def _signed(x: float) -> str:
    return "—" if x != x else f"{x:+.1f}"


def _ci(r: PairedDiff) -> str:
    return "—" if r.ci_low != r.ci_low else f"[{r.ci_low:+.1f}, {r.ci_high:+.1f}]"


def _verdict(o: PairedDiff, a: str, b: str) -> str:
    if o.n_cells == 0:
        return "no matched cells — the runs share no comparable condition"
    if o.ci_low != o.ci_low:
        return "CI undefined (fewer than 2 scenarios) — no reliable winner"
    if o.ci_low > 0:
        return f"{b} captures more value above BATNA (CI excludes 0)"
    if o.ci_high < 0:
        return f"{a} captures more value above BATNA (CI excludes 0)"
    return "no clear difference (CI includes 0)"


def _side(traces: list[Trace], scenarios: dict[str, Scenario]) -> dict[str, float | None]:
    m = side_metrics("all", traces, scenarios)
    costs = [t.cost_usd for t in traces if t.cost_usd is not None]
    return {
        "episodes": float(len(traces)),
        "error_rate": sum(t.errored for t in traces) / len(traces) if traces else None,
        "agreement_rate": m.agreement_rate,
        "walk_away_accuracy": m.walk_away_accuracy,
        "pareto_rate": m.pareto_rate,
        "mean_rounds": m.mean_rounds_to_close,
        "mean_latency_s": float(np.mean([t.latency_s for t in traces])) if traces else None,
        "mean_cost_usd": float(np.mean(costs)) if costs else None,
        "mean_tokens": float(np.mean([t.token_cost for t in traces])) if traces else None,
        "total_cost_usd": float(sum(costs)) if costs else None,
    }


def compare_runs(
    a: list[Trace],
    b: list[Trace],
    scenarios: dict[str, Scenario],
    *,
    label_a: str = "A",
    label_b: str = "B",
    n_boot: int = 2000,
    seed: int = 0,
) -> Comparison:
    """Compare two runs of the same condition on their matched cells."""
    rng = np.random.default_rng(seed)
    ca, cb = _cell_means(a, scenarios), _cell_means(b, scenarios)

    def subset(cells: dict[Cell, float], idx: int, value: str) -> dict[Cell, float]:
        return {c: q for c, q in cells.items() if c[idx] == value}

    matched = set(ca) & set(cb)
    strategies = sorted({c[1] for c in matched})
    opponents = sorted({c[2] for c in matched})
    by_strategy = [
        paired_diff(s, subset(ca, 1, s), subset(cb, 1, s), n_boot=n_boot, rng=rng)
        for s in strategies
    ]
    by_opponent = [
        paired_diff(o, subset(ca, 2, o), subset(cb, 2, o), n_boot=n_boot, rng=rng)
        for o in opponents
    ]
    # Separate tracks on the shared design only, so unequal coverage (one run with
    # more strategies/scenarios) doesn't masquerade as a model difference. Cells are
    # matched by presence here, not scorability: errors and no-ZOPA walk-aways count.
    shared = {_cell(t) for t in a} & {_cell(t) for t in b}
    a_shared = [t for t in a if _cell(t) in shared]
    b_shared = [t for t in b if _cell(t) in shared]
    by_scenario = [
        paired_diff(sc, subset(ca, 0, sc), subset(cb, 0, sc), n_boot=n_boot, rng=rng)
        for sc in sorted({c[0] for c in matched})
    ]
    grid_diffs: dict[tuple[str, str], list[float]] = defaultdict(list)
    for c in matched:
        grid_diffs[(c[1], c[2])].append(100.0 * (cb[c] - ca[c]))
    grid = {k: float(np.mean(v)) for k, v in grid_diffs.items()}
    return Comparison(
        label_a=label_a,
        label_b=label_b,
        models_a=_describe(a),
        models_b=_describe(b),
        overall=paired_diff("overall", ca, cb, n_boot=n_boot, rng=rng),
        by_strategy=sorted(by_strategy, key=lambda r: -r.delta if r.delta == r.delta else 0),
        by_opponent=by_opponent,
        by_scenario=by_scenario,
        grid=grid,
        side_a=_side(a_shared, scenarios),
        side_b=_side(b_shared, scenarios),
        cells_only_a=len(set(ca) - set(cb)),
        cells_only_b=len(set(cb) - set(ca)),
        warnings=condition_warnings(a, b),
    )


def _json_num(x: float | None) -> float | None:
    return None if x is None or x != x else round(float(x), 4)


def payload_row(r: PairedDiff) -> dict[str, object]:
    """One PairedDiff as a JSON-safe dashboard row (NaN → None)."""
    return {
        "name": r.name,
        "a": _json_num(r.score_a),
        "b": _json_num(r.score_b),
        "delta": _json_num(r.delta),
        "lo": _json_num(r.ci_low),
        "hi": _json_num(r.ci_high),
        "cells": r.n_cells,
        "scenarios": r.n_scenarios,
    }


def comparison_payload(c: Comparison, *, source_a: str, source_b: str) -> dict[str, object]:
    """JSON-safe payload for the compare dashboard."""
    return {
        "kind": "compare",
        "title": "Run comparison",
        "label_a": c.label_a,
        "label_b": c.label_b,
        "models_a": c.models_a,
        "models_b": c.models_b,
        "source_a": source_a,
        "source_b": source_b,
        "overall": payload_row(c.overall),
        "verdict": _verdict(c.overall, c.label_a, c.label_b),
        "by_strategy": [payload_row(r) for r in c.by_strategy],
        "by_opponent": [payload_row(r) for r in c.by_opponent],
        "by_scenario": [payload_row(r) for r in c.by_scenario],
        "grid": [
            {"strategy": s, "opponent": op, "delta": _json_num(d)}
            for (s, op), d in sorted(c.grid.items())
        ],
        "side": {
            "cols": [c.label_a, c.label_b],
            "rows": [
                {
                    "label": label,
                    "kind": kind,
                    "values": [_json_num(c.side_a.get(key)), _json_num(c.side_b.get(key))],
                }
                for key, (label, kind) in _SIDE_ROWS.items()
            ],
        },
        "cells_only_a": c.cells_only_a,
        "cells_only_b": c.cells_only_b,
        "warnings": list(c.warnings),
    }


_TEMPLATE = Path(__file__).parent / "templates" / "compare.html"
_PLACEHOLDER = "__RFQ_BENCH_COMPARE__"


def render_compare_html(payload: dict[str, object]) -> str:
    """Inline the payload into the self-contained compare dashboard template."""
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return _TEMPLATE.read_text(encoding="utf-8").replace(_PLACEHOLDER, blob)
