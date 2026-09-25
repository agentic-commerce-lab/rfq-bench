"""Model duel scoring: two models negotiating against each other in both roles.

A duel run plays every A2A cell (scenario × seller strategy × buyer persona ×
first speaker) once per pairing of (seller model, buyer model). The model is the
treatment; strategy and persona are held fixed within a cell.

**Model score.** For cell ``c``, write ``s[X→Y]`` for the seller's q when model X
sells to model Y, and ``b[X→Y]`` for the buyer's q in the same episodes. Each
model's score averages its two roles against the other model::

    S_A(c) = ( s[A→B](c) + b[B→A](c) ) / 2
    S_B(c) = ( s[B→A](c) + b[A→B](c) ) / 2

Each pairing has exactly one seller and one buyer, so any advantage the scenarios
give one role cancels, and the opponent is always the other model. ``Δ = S_B − S_A``
is macro-averaged over cells with a scenario-clustered bootstrap CI (the same
machinery as ``rfq-bench compare``).

**Role skills** (``full`` pairings only, i.e. with both self-play pairs) hold the
counterpart fixed and swap only one side::

    seller Δ = mean over buyers Y  of  s[B→Y] − s[A→Y]
    buyer  Δ = mean over sellers X of  b[X→B] − b[X→A]

Pure: a function of the persisted traces and the scenarios. Cells need a scorable
episode in every pairing they use; errored, no-ZOPA and degenerate episodes are
dropped (errors and walk-aways are reported separately per pairing).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np

from rfq_bench.core.contracts import Role, Scenario, Trace
from rfq_bench.core.scoring import score_trace
from rfq_bench.report.compare import (
    _SIDE_ROWS,
    PairedDiff,
    _ci,
    _side,
    _signed,
    _verdict,
    paired_diff,
)

DuelCell = tuple[str, str, str, str]  # scenario, strategy, persona, first speaker
Pairing = tuple[str, str]  # (seller model, buyer model)


def _dcell(t: Trace) -> DuelCell:
    return (t.scenario_id, t.strategy, t.opponent, t.first_speaker)


def pairing_of(t: Trace) -> Pairing | None:
    m = t.models_by_role
    if "seller" in m and "buyer" in m:
        return (m["seller"], m["buyer"])
    return None


def duel_traces(traces: Iterable[Trace]) -> list[Trace]:
    return [t for t in traces if t.mode == "duel" and pairing_of(t) is not None]


def duel_models(traces: Iterable[Trace]) -> tuple[str, str]:
    """The two models in a duel file, in a stable (alphabetical) order."""
    models = sorted({m for t in traces for m in t.models_by_role.values()})
    if len(models) != 2:
        raise ValueError(f"a duel file must contain exactly two models, found {models}")
    return (models[0], models[1])


def _role_q(
    traces: Iterable[Trace], scenarios: dict[str, Scenario], role: Role
) -> dict[Pairing, dict[DuelCell, float]]:
    """Mean q of ``role`` per (pairing, cell), seeds averaged; unscorable dropped."""
    acc: dict[Pairing, dict[DuelCell, list[float]]] = defaultdict(lambda: defaultdict(list))
    for t in traces:
        pair = pairing_of(t)
        if pair is None:
            continue
        ep = score_trace(t, scenarios[t.scenario_id], role=role)
        if ep.scorable:
            assert ep.q is not None
            acc[pair][_dcell(t)].append(ep.q)
    return {p: {c: float(np.mean(v)) for c, v in cells.items()} for p, cells in acc.items()}


def _combine(*parts: dict[DuelCell, float]) -> dict[DuelCell, float]:
    """Per-cell mean of several cell→q maps, over the cells present in all of them."""
    if not parts:
        return {}
    shared = set(parts[0]).intersection(*parts[1:])
    return {c: float(np.mean([p[c] for p in parts])) for c in shared}


@dataclass(frozen=True)
class ModelUsage:
    """What one model spent across all the sides it played in a duel file."""

    calls: int  # moves made (one act each; re-asks are billed inside it)
    tokens: int
    prompt_tokens: int
    cached_tokens: int
    cost_usd: float | None  # provider-reported; None when never reported

    @property
    def completion_tokens(self) -> int:
        return max(0, self.tokens - self.prompt_tokens)

    @property
    def cost_per_call(self) -> float | None:
        return None if self.cost_usd is None or not self.calls else self.cost_usd / self.calls

    @property
    def tokens_per_call(self) -> float | None:
        return self.tokens / self.calls if self.calls else None

    @property
    def cache_rate(self) -> float | None:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else None


def model_providers(traces: Iterable[Trace]) -> dict[str, dict[str, int]]:
    """Which upstream providers served each model's moves, with call counts."""
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for t in traces:
        for s in t.steps:
            model = t.models_by_role.get(s.party)
            if model is not None and s.provider:
                out[model][s.provider] += 1
    return {m: dict(sorted(c.items(), key=lambda kv: -kv[1])) for m, c in out.items()}


def model_usage(traces: Iterable[Trace]) -> dict[str, ModelUsage]:
    """Sum each model's per-role tokens and spend over every episode it played in."""
    acc: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    seen_cost: set[str] = set()
    for t in traces:
        parties = [s.party for s in t.steps]
        for role, model in t.models_by_role.items():
            a = acc[model]
            a["calls"] += parties.count(role)
            a["tokens"] += t.token_cost_by_role.get(role, 0)
            a["prompt"] += t.prompt_tokens_by_role.get(role, 0)
            a["cached"] += t.cached_tokens_by_role.get(role, 0)
            cost = t.cost_usd_by_role.get(role)
            if cost is not None:
                a["cost"] += cost
                seen_cost.add(model)
    return {
        m: ModelUsage(
            calls=int(a["calls"]),
            tokens=int(a["tokens"]),
            prompt_tokens=int(a["prompt"]),
            cached_tokens=int(a["cached"]),
            cost_usd=a["cost"] if m in seen_cost else None,
        )
        for m, a in acc.items()
    }


@dataclass(frozen=True)
class Duel:
    model_a: str
    model_b: str
    pairings: list[Pairing]
    overall: PairedDiff
    by_strategy: list[PairedDiff]
    by_persona: list[PairedDiff]
    by_scenario: list[PairedDiff]
    grid: dict[tuple[str, str], float]
    # Role skills; None unless both self-play pairings are present.
    seller_skill: PairedDiff | None
    buyer_skill: PairedDiff | None
    # Seller S per pairing (points), as a plain readout of each matchup.
    pairing_scores: dict[Pairing, float]
    side: dict[Pairing, dict[str, float | None]]
    # Per-model token and USD spend over every side it played (see model_usage).
    usage: dict[str, ModelUsage] = field(default_factory=dict)
    # Upstream providers that served each model's calls: model -> {provider: calls}.
    providers: dict[str, dict[str, int]] = field(default_factory=dict)
    # When and from which code version the traces were produced (provenance).
    started: tuple[str, ...] = ()
    code_versions: tuple[str, ...] = ()
    warnings: list[str] = field(default_factory=list)

    def label(self, p: Pairing) -> str:
        """Compact 'seller → buyer' label by model name."""
        return f"{_short(p[0])} → {_short(p[1])}"

    def render(self) -> str:
        na, nb = _short(self.model_a), _short(self.model_b)
        lines = [
            f"rfq-bench duel — {na} ({self.model_a}) vs {nb} ({self.model_b})",
            "Score = mean of a model's seller q and buyer q against the OTHER model, per matched "
            "cell (scenario × strategy × persona × first speaker).",
            f"Δ = {nb} − {na} in points; 95% CI clustered by scenario.",
        ]
        for warning in self.warnings:
            lines.append(f"WARNING: {warning}")
        o = self.overall
        lines += [
            "",
            f"Overall  {na} = {o.score_a:.1f}  {nb} = {o.score_b:.1f}  Δ = {_signed(o.delta)}  "
            f"{_ci(o)}  ({o.n_cells} cells, {o.n_scenarios} scenarios)",
            f"  → {_verdict(o, na, nb)}",
        ]
        if self.seller_skill is not None and self.buyer_skill is not None:
            lines.append("")
            lines.append("Skill by role (counterpart held fixed)")
            for name, r in (("as seller", self.seller_skill), ("as buyer", self.buyer_skill)):
                lines.append(
                    f"  {name:<10} {na} = {r.score_a:.1f}  {nb} = {r.score_b:.1f}  "
                    f"Δ = {_signed(r.delta)}  {_ci(r)}"
                )
        wn = max(8, len(na) + 2, len(nb) + 2)
        for title, rows in (
            ("strategy", self.by_strategy),
            ("persona", self.by_persona),
            ("scenario", self.by_scenario),
        ):
            if len(rows) < 2:
                continue  # a single row just repeats the overall line
            lines.append("")
            w = max([12, len(title) + 2, *(len(r.name) + 2 for r in rows)])
            header = f"{title:<{w}}{na:>{wn}}{nb:>{wn}}{'Δ':>9}{'95% CI':>20}{'cells':>7}"
            lines += [header, "-" * len(header)]
            for r in rows:
                lines.append(
                    f"{r.name:<{w}}{r.score_a:>{wn}.1f}{r.score_b:>{wn}.1f}{_signed(r.delta):>9}"
                    f"{_ci(r):>20}{r.n_cells:>7}"
                )
        lines.append("")
        lines.append(
            "Per matchup, seller → buyer (seller score; other metrics are not part of the score)"
        )
        from rfq_bench.report.compare import _fmt

        for p in self.pairings:
            side = self.side[p]
            parts = [f"seller score {_num(self.pairing_scores.get(p))}"] + [
                f"{label} {_fmt(side.get(key), kind)}" for key, (label, kind) in _SIDE_ROWS.items()
            ]
            lines.append(f"  {self.label(p)}: " + " · ".join(parts))
        if self.usage:
            lines.append("")
            lines.append("Token cost by model (every side it played)")
            for m in (self.model_a, self.model_b):
                u = self.usage.get(m)
                if u is None:
                    continue
                cost = "—" if u.cost_usd is None else f"${u.cost_usd:.4f}"
                per = "—" if u.cost_per_call is None else f"${u.cost_per_call:.5f}"
                tpc = "—" if u.tokens_per_call is None else f"{u.tokens_per_call:,.0f}"
                lines.append(
                    f"  {_short(m)}: {cost} total · {u.calls:,} calls · {u.tokens:,} tokens "
                    f"({u.prompt_tokens:,} prompt / {u.completion_tokens:,} completion) · "
                    f"{tpc} tokens/call · {per}/call"
                )
        if self.providers:
            lines.append("")
            lines.append("Served by (upstream provider: calls)")
            for m in (self.model_a, self.model_b):
                served = self.providers.get(m)
                if served:
                    lines.append(
                        f"  {_short(m)}: " + ", ".join(f"{p} {n}" for p, n in served.items())
                    )
        if self.started or self.code_versions:
            lines.append("")
            when = f"{self.started[0]} … {self.started[-1]}" if self.started else "unknown"
            lines.append(f"Run: {when} · code {', '.join(self.code_versions) or 'unknown'}")
        return "\n".join(lines)


def _num(x: float | None) -> str:
    return "—" if x is None or x != x else f"{x:.1f}"


def _short(model: str) -> str:
    """'anthropic/claude-opus-5.5' -> 'Claude Opus 5.5'; 'openai/gpt-6-sol' -> 'GPT-6 Sol'."""
    parts = model.rsplit("/", 1)[-1].split(":")[0].split("-")
    if parts[0].lower() == "gpt" and len(parts) > 1:
        head, rest = f"GPT-{parts[1]}", parts[2:]
    else:
        head, rest = parts[0].capitalize(), parts[1:]
    return " ".join([head, *(p if p[:1].isdigit() else p.capitalize() for p in rest)])


_PROVIDERS = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "google": "Google",
    "deepseek": "DeepSeek",
}


def _provider(model: str) -> str:
    prefix = model.split("/", 1)[0] if "/" in model else ""
    return _PROVIDERS.get(prefix.lower(), prefix.capitalize())


# Plain-language per-matchup metrics for the duel page (not part of the score).
_DUEL_ROWS: list[tuple[str, str, str]] = [
    ("episodes", "Negotiations", "int"),
    ("deal_rate", "Deals closed, where a deal was possible", "pct"),
    ("walk_rate", "Walked away, where no deal was possible", "pct"),
    ("error_rate", "Failed (technical errors)", "pct"),
    ("moves", "Moves per negotiation", "f1"),
    ("latency", "Time per negotiation", "sec"),
    ("cost", "Cost per negotiation", "usd"),
    ("total_cost", "Total cost", "usd"),
]


def _duel_side(traces: list[Trace], scenarios: dict[str, Scenario]) -> dict[str, float | None]:
    from rfq_bench.core.zopa import compute_zopa

    def rate(xs: list[bool]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    possible = [t for t in traces if compute_zopa(scenarios[t.scenario_id]).exists]
    impossible = [t for t in traces if not compute_zopa(scenarios[t.scenario_id]).exists]
    costs = [t.cost_usd for t in traces if t.cost_usd is not None]
    return {
        "episodes": float(len(traces)),
        "deal_rate": rate([t.agreement is not None for t in possible]),
        "walk_rate": rate([t.agreement is None for t in impossible]),
        "error_rate": rate([t.errored for t in traces]),
        "moves": float(np.mean([len(t.steps) for t in traces])) if traces else None,
        "latency": float(np.mean([t.latency_s for t in traces])) if traces else None,
        "cost": float(np.mean(costs)) if costs else None,
        "total_cost": float(sum(costs)) if costs else None,
    }


def duel_warnings(traces: list[Trace]) -> list[str]:
    warns: list[str] = []
    checks: dict[str, Callable[[Trace], object]] = {
        "temperature": lambda t: t.llm_config.temperature,
        "max_tokens": lambda t: t.llm_config.max_tokens,
        "seed": lambda t: t.llm_config.seed,
        "tool_choice": lambda t: t.llm_config.tool_choice,
        "outcome_space_version": lambda t: t.outcome_space_version,
    }
    for label, get in checks.items():
        vals = sorted({str(get(t)) for t in traces})
        if len(vals) > 1:
            warns.append(f"{label} varies within the duel file: {vals}")
    from rfq_bench.provenance import condition_differences

    warns += condition_differences([t.provenance for t in traces if t.provenance])
    by_pair: dict[Pairing, int] = defaultdict(int)
    for t in traces:
        p = pairing_of(t)
        if p is not None:
            by_pair[p] += 1
    if len(set(by_pair.values())) > 1:
        counts = ", ".join(f"{_short(s)}→{_short(b)}: {n}" for (s, b), n in sorted(by_pair.items()))
        warns.append(f"pairings have unequal episode counts ({counts}); only matched cells count")
    return warns


def build_duel(
    traces: list[Trace],
    scenarios: dict[str, Scenario],
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> Duel:
    """Score a duel trace file (see the module docstring for the definitions)."""
    traces = duel_traces(traces)
    a, b = duel_models(traces)
    rng = np.random.default_rng(seed)
    sq = _role_q(traces, scenarios, "seller")
    bq = _role_q(traces, scenarios, "buyer")
    ab, ba, aa, bb = (a, b), (b, a), (a, a), (b, b)
    if ab not in sq or ba not in sq:
        raise ValueError("a duel needs both cross pairings (A sells to B and B sells to A)")

    score_a = _combine(sq[ab], bq.get(ba, {}))
    score_b = _combine(sq[ba], bq.get(ab, {}))

    def subset(cells: dict[DuelCell, float], idx: int, value: str) -> dict[DuelCell, float]:
        return {c: q for c, q in cells.items() if c[idx] == value}

    matched = set(score_a) & set(score_b)

    def breakdown(idx: int) -> list[PairedDiff]:
        return [
            paired_diff(v, subset(score_a, idx, v), subset(score_b, idx, v), n_boot=n_boot, rng=rng)
            for v in sorted({c[idx] for c in matched})
        ]

    overall = paired_diff("overall", score_a, score_b, n_boot=n_boot, rng=rng)
    grid_d: dict[tuple[str, str], list[float]] = defaultdict(list)
    for c in matched:
        grid_d[(c[1], c[2])].append(100.0 * (score_b[c] - score_a[c]))

    seller_skill = buyer_skill = None
    if aa in sq and bb in sq:
        seller_skill = paired_diff(
            "as seller",
            _combine(sq[ab], sq[aa]),  # A sells, buyer ∈ {B, A}
            _combine(sq[bb], sq[ba]),  # B sells, buyer ∈ {B, A}
            n_boot=n_boot,
            rng=rng,
        )
        buyer_skill = paired_diff(
            "as buyer",
            _combine(bq.get(aa, {}), bq.get(ba, {})),  # A buys, seller ∈ {A, B}
            _combine(bq.get(ab, {}), bq.get(bb, {})),  # B buys, seller ∈ {A, B}
            n_boot=n_boot,
            rng=rng,
        )

    pairings = [p for p in (ab, ba, aa, bb) if p in sq]
    pairing_scores = {p: 100.0 * float(np.mean(list(sq[p].values()))) for p in pairings}
    side = {p: _side([t for t in traces if pairing_of(t) == p], scenarios) for p in pairings}
    return Duel(
        model_a=a,
        model_b=b,
        pairings=pairings,
        overall=overall,
        by_strategy=breakdown(1),
        by_persona=breakdown(2),
        by_scenario=breakdown(0),
        grid={k: float(np.mean(v)) for k, v in grid_d.items()},
        seller_skill=seller_skill,
        buyer_skill=buyer_skill,
        pairing_scores=pairing_scores,
        side=side,
        usage=model_usage(traces),
        providers=model_providers(traces),
        started=tuple(
            sorted({t.provenance["started_at"] for t in traces if "started_at" in t.provenance})
        ),
        code_versions=tuple(
            sorted({t.provenance["code_version"] for t in traces if "code_version" in t.provenance})
        ),
        warnings=duel_warnings(traces),
    )


def duel_payload(
    d: Duel,
    traces: list[Trace],
    scenarios: dict[str, Scenario],
    *,
    source: str,
) -> dict[str, object]:
    """Payload for the duel dashboard: results by model NAME, plus replayable episodes."""
    from rfq_bench.report.compare import payload_row
    from rfq_bench.report.dashboard import build_payload

    name = {d.model_a: _short(d.model_a), d.model_b: _short(d.model_b)}
    o = d.overall
    winner = None
    if o.ci_low == o.ci_low and o.ci_low > 0:
        winner = d.model_b
    elif o.ci_high == o.ci_high and o.ci_high < 0:
        winner = d.model_a
    duel = duel_traces(traces)
    base = build_payload(duel, scenarios, source=source)
    meta = base["meta"]
    assert isinstance(meta, dict)
    return {
        "source": source,
        "models": [
            {"id": d.model_a, "name": name[d.model_a], "provider": _provider(d.model_a)},
            {"id": d.model_b, "name": name[d.model_b], "provider": _provider(d.model_b)},
        ],
        # Scored negotiations between the two models (both cross matchups).
        "n_negotiations": 2 * o.n_cells,
        "overall": payload_row(o),
        "verdict": _verdict(o, name[d.model_a], name[d.model_b]),
        "winner": winner,
        "roles": [
            {**payload_row(r), "name": label}
            for r, label in ((d.seller_skill, "When selling"), (d.buyer_skill, "When buying"))
            if r is not None
        ],
        "matchups": [
            {
                "seller": s,
                "buyer": b,
                "seller_s": _json_num(d.pairing_scores.get((s, b))),
                "agreement_rate": _json_num(d.side[(s, b)].get("agreement_rate")),
                "episodes": _json_num(d.side[(s, b)].get("episodes")),
            }
            for (s, b) in d.pairings
        ],
        "by_scenario": [
            {**payload_row(r), "name": _scenario_label(scenarios.get(r.name), r.name)}
            for r in d.by_scenario
        ],
        "by_strategy": [payload_row(r) for r in d.by_strategy],
        "by_persona": [payload_row(r) for r in d.by_persona],
        "grid": [
            {"strategy": st, "persona": op, "delta": _json_num(v)}
            for (st, op), v in sorted(d.grid.items())
        ],
        "side": {
            "cols": [{"seller": s, "buyer": b} for (s, b) in d.pairings],
            "rows": [
                {
                    "label": label,
                    "kind": kind,
                    "values": [
                        _json_num(
                            _duel_side([t for t in duel if pairing_of(t) == p], scenarios).get(key)
                        )
                        for p in d.pairings
                    ],
                }
                for key, label, kind in _DUEL_ROWS
            ],
        },
        "usage": [
            {
                "id": m,
                "calls": u.calls,
                "tokens": u.tokens,
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "cached_tokens": u.cached_tokens,
                "cost_usd": _json_num(u.cost_usd) if u.cost_usd is None else round(u.cost_usd, 6),
                "cost_per_call": None if u.cost_per_call is None else round(u.cost_per_call, 7),
                "tokens_per_call": _json_num(u.tokens_per_call),
                "cache_rate": _json_num(u.cache_rate),
            }
            for m in (d.model_a, d.model_b)
            if (u := d.usage.get(m)) is not None
        ],
        "providers": d.providers,
        "run": {"started": list(d.started), "code_versions": list(d.code_versions)},
        "warnings": list(d.warnings),
        # Replay: the same per-episode payload the main dashboard renders.
        "meta": {"has_messages": meta.get("has_messages", False)},
        "scenarios": base["scenarios"],
        "episodes": base["episodes"],
        "glossary": base["glossary"],
        "scenario_labels": {
            sid: _scenario_label(scenarios.get(sid), sid) for sid in {t.scenario_id for t in duel}
        },
    }


def _scenario_label(scenario: Scenario | None, fallback: str) -> str:
    """A reader-facing scenario name: its product, e.g. 'Office chairs (200 units)'."""
    product = scenario.product if scenario is not None else ""
    return product[:1].upper() + product[1:] if product else fallback


def render_duel_html(payload: dict[str, object]) -> str:
    """Inline payload, uPlot and the shared replay module into the duel template."""
    import json
    from pathlib import Path

    from rfq_bench.report.dashboard import inline_replay, uplot_assets

    template = (Path(__file__).parent / "templates" / "duel.html").read_text(encoding="utf-8")
    css, js = uplot_assets()
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = template.replace("/* __UPLOT_CSS__ */", css).replace("/* __UPLOT_JS__ */", js)
    return inline_replay(html).replace("__RFQ_BENCH_DUEL__", blob)


def _json_num(x: float | None) -> float | None:
    return None if x is None or x != x else round(float(x), 4)
