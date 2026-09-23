"""Build the self-contained interactive dashboard.

This module turns immutable traces plus their scenarios into a single, compact
JSON payload and injects it into a static, dependency-free HTML template. The
result is one offline HTML file: open it with a double-click, no server.

Scoring is **not** re-derived here. Every per-episode ``q`` comes from
:func:`rfq_bench.core.scoring.score_trace`, so the dashboard and
``rfq-bench report`` always agree on the numbers. The browser only recomputes
cheap aggregates (means and a scenario-clustered bootstrap) over those frozen
``q`` values, so filters stay live without touching the scoring kernel.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rfq_bench.core.contracts import Scenario, Trace
from rfq_bench.core.outcome_space import enumerate_outcomes
from rfq_bench.core.scoring import score_trace
from rfq_bench.core.utility import (
    ideal_utility,
    outcome_utility,
    reservation_utility,
    worst_utility,
)
from rfq_bench.core.zopa import compute_zopa
from rfq_bench.report.fidelity import DEFAULT_TOLERANCE, episode_fidelity
from rfq_bench.report.metrics import best_price, revenue

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_DATA_PLACEHOLDER = "__RFQ_BENCH_PAYLOAD__"
_UPLOT_JS_PLACEHOLDER = "/* __UPLOT_JS__ */"
_UPLOT_CSS_PLACEHOLDER = "/* __UPLOT_CSS__ */"


def _scenario_payload(scenario: Scenario) -> dict[str, Any]:
    """Public scenario metadata the replay view needs.

    This is an offline analysis artifact, never a model-visible payload, so it
    may expose the BATNA/ideal points that the replay chart draws. It stays out
    of every model path by construction (the dashboard reads persisted traces).
    """
    zopa = compute_zopa(scenario)
    parties: dict[str, dict[str, float]] = {}
    for role in ("buyer", "seller"):
        prefs = scenario.preferences(role)
        parties[role] = {
            "batna": reservation_utility(prefs),
            "ideal": ideal_utility(prefs, scenario.issues),
            "worst": worst_utility(prefs, scenario.issues),
        }
    # The outcome space is tiny (<=27), so ship every outcome's utilities. This
    # lets the browser compute joint surplus and Pareto efficiency exactly,
    # matching report/metrics.py, without re-deriving any private tables in JS.
    outcomes = [
        {
            "outcome": o,
            "buyer": outcome_utility(scenario.buyer, scenario.issues, o),
            "seller": outcome_utility(scenario.seller, scenario.issues, o),
        }
        for o in enumerate_outcomes(scenario.issues)
    ]
    return {
        "id": scenario.id,
        "kind": scenario.kind.value,
        "product": scenario.product,
        "deadline_rounds": scenario.deadline_rounds,
        "issues": [
            {"name": i.name, "unit": i.unit, "values": list(i.values)} for i in scenario.issues
        ],
        "parties": parties,
        "zopa_exists": zopa.exists,
        "zopa_outcomes": zopa.outcomes,
        "outcomes": outcomes,
    }


def _step_utilities(scenario: Scenario, outcome: dict[str, Any] | None) -> dict[str, float] | None:
    """Both parties' utility for a step's outcome, for the utility trajectory."""
    if outcome is None:
        return None
    return {
        "buyer": outcome_utility(scenario.buyer, scenario.issues, outcome),
        "seller": outcome_utility(scenario.seller, scenario.issues, outcome),
    }


def _revenue_fields(scenario: Scenario, agreement: dict[str, Any] | None) -> dict[str, Any]:
    r, best = revenue(scenario, agreement), best_price(scenario)
    return {"revenue": r, "revenue_share": (r / best) if (r is not None and best) else None}


def _fidelity_payload(trace: Trace, scenario: Scenario) -> dict[str, Any] | None:
    """Per-episode strategy-fidelity data (see ``report/fidelity.py``).

    Raw per-offer checks plus counts, so the browser can pool them over any
    filtered subset exactly as ``rfq-bench report`` pools them over an arm.
    None when the episode is not checkable (unknown strategy, errored, degenerate).
    """
    ep = episode_fidelity(trace, scenario)
    if ep is None:
        return None
    return {
        "offers": [
            {
                "round": o.round,
                "progress": o.progress,
                "gap": o.gap,
                "level": o.agent_level,
                "ref_level": o.reference_level,
                "ref_outcome": o.reference_outcome,
            }
            for o in ep.offers
        ],
        "early": ep.early_accepts,
        "missed": ep.missed_accepts,
        "decisions": ep.accept_decisions,
        "beta": ep.fitted_beta,
        "ref_beta": ep.reference_beta,
    }


def _episode_payload(trace: Trace, scenario: Scenario) -> dict[str, Any]:
    scored = score_trace(trace, scenario)
    # Also score the OTHER party (buyer, in A2A the persona/opponent side) so the
    # dashboard can toggle which side's captured value it shows — same traces.
    scored_buyer = score_trace(trace, scenario, role="buyer")
    steps = [
        {
            "round": s.round,
            "party": s.party,
            "action": s.action,
            "outcome": s.outcome,
            "utilities": _step_utilities(scenario, s.outcome),
            "rationale": s.rationale,
            "message": s.message,
            "message_truncated": s.message_truncated,
            "message_withheld": s.message_withheld,
            "adjusted": s.adjusted,
            "adjust_reason": s.adjust_reason,
            "adjust_kind": s.adjust_kind,
            "intended_action": s.intended_action,
            "intended_outcome": s.intended_outcome,
            "error": s.error,
            "error_reason": s.error_reason,
            "raw_response": s.raw_response,
            "reasoning": s.reasoning,
        }
        for s in trace.steps
    ]
    return {
        "scenario_id": trace.scenario_id,
        "scenario_kind": scenario.kind.value,
        "strategy": trace.strategy,
        "opponent": trace.opponent,
        "target_role": trace.target_role,
        "first_speaker": trace.first_speaker,
        "seed": trace.seed,
        "model": trace.llm_config.model,
        "mode": trace.mode,
        "q": scored.q,
        "degenerate": scored.degenerate,
        # No ZOPA: q is None and the episode is excluded from S_s (walk✓ instead).
        "no_zopa": scored.no_zopa,
        "q_buyer": scored_buyer.q,
        "q_buyer_degenerate": scored_buyer.degenerate,
        "errored": trace.errored,
        "error_reason": trace.error_reason,
        "outcome_kind": trace.outcome_kind,
        "agreement": trace.agreement,
        "utilities": dict(trace.utilities),
        "rounds_to_close": trace.rounds_to_close,
        "latency_s": trace.latency_s,
        "token_cost": trace.token_cost,
        "cost_usd": trace.cost_usd,
        "validity": dict(trace.validity),
        # Agreed price (seller revenue / buyer spend; 0 without a deal) and its share
        # of the seller's best price option — same functions as `rfq-bench report`.
        **_revenue_fields(scenario, trace.agreement),
        # Policy constraints that fired, per party (invalid moves the harness corrected).
        "policy_fires": {
            role: sum(1 for s in trace.steps if s.adjusted and s.party == role)
            for role in ("buyer", "seller")
        },
        "fidelity": _fidelity_payload(trace, scenario),
        "steps": steps,
    }


def build_payload(
    traces: list[Trace],
    scenarios: dict[str, Scenario],
    *,
    control: str = "control",
    buyer_control: str = "neutral",
    n_boot: int = 2000,
    seed: int = 0,
    source: str | None = None,
) -> dict[str, Any]:
    """Assemble the full JSON payload the dashboard renders from.

    ``control``/``n_boot``/``seed`` are echoed into the payload so the in-browser
    bootstrap uses the same control condition and settings as ``rfq-bench report``.
    """
    used_ids = {t.scenario_id for t in traces}
    episodes = [_episode_payload(t, scenarios[t.scenario_id]) for t in traces]
    n_degenerate = sum(1 for e in episodes if e["degenerate"])
    n_no_zopa = sum(1 for e in episodes if e["no_zopa"])
    has_messages = any(s["rationale"] or s["message"] for e in episodes for s in e["steps"])
    has_adjustments = any(s["adjusted"] for e in episodes for s in e["steps"])
    has_errors = any(e["errored"] for e in episodes)
    n_errored = sum(1 for e in episodes if e["errored"])
    has_reasoning = any(s["reasoning"] for e in episodes for s in e["steps"])
    has_cost = any(e["cost_usd"] is not None for e in episodes)
    total_cost = sum(e["cost_usd"] or 0.0 for e in episodes) if has_cost else None
    control_present = any(e["strategy"] == control for e in episodes)
    has_a2a = any(e["mode"] == "a2a" for e in episodes)
    return {
        "meta": {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "source": source,
            "control": control,
            "buyer_control": buyer_control,
            "has_a2a": has_a2a,
            "fidelity_tolerance": DEFAULT_TOLERANCE,
            "n_boot": n_boot,
            "seed": seed,
            "n_episodes": len(episodes),
            "n_degenerate": n_degenerate,
            "n_no_zopa": n_no_zopa,
            "has_messages": has_messages,
            "has_adjustments": has_adjustments,
            "has_errors": has_errors,
            "n_errored": n_errored,
            "has_reasoning": has_reasoning,
            "has_cost": has_cost,
            "total_cost_usd": total_cost,
            "control_present": control_present,
            "strategies": sorted({e["strategy"] for e in episodes}),
            "opponents": sorted({e["opponent"] for e in episodes}),
            "scenarios": sorted(used_ids),
            "roles": sorted({e["target_role"] for e in episodes}),
            "first_speakers": sorted({e["first_speaker"] for e in episodes}),
            "seeds": sorted({e["seed"] for e in episodes}),
            "models": sorted({t.llm_config.model for t in traces}),
        },
        "scenarios": {sid: _scenario_payload(scenarios[sid]) for sid in sorted(used_ids)},
        "episodes": episodes,
        # Plain-language tooltip texts for every strategy, opponent and scenario shown.
        "glossary": _glossary(episodes, [scenarios[sid] for sid in sorted(used_ids)]),
    }


def _glossary(episodes: list[dict[str, Any]], used: list[Scenario]) -> dict[str, dict[str, str]]:
    # Custom strategies/personas (prompts/*/*.md in the working dir) fall back to
    # the first sentence of their own guidance text.
    from rfq_bench.agent.personas import load_persona_guidance
    from rfq_bench.agent.prompts import load_strategy_guidance
    from rfq_bench.report.glossary import build_glossary

    return build_glossary(
        strategies=sorted({e["strategy"] for e in episodes}),
        opponents=sorted({e["opponent"] for e in episodes}),
        scenarios=used,
        strategy_guidance=load_strategy_guidance(),
        persona_guidance=load_persona_guidance(),
    )


def render_html(payload: dict[str, Any]) -> str:
    """Inline the payload and the vendored uPlot assets into the template."""
    template = (_TEMPLATE_DIR / "dashboard.html").read_text(encoding="utf-8")
    uplot_js = (_TEMPLATE_DIR / "vendor" / "uPlot.iife.min.js").read_text(encoding="utf-8")
    uplot_css = (_TEMPLATE_DIR / "vendor" / "uPlot.min.css").read_text(encoding="utf-8")
    # Compact JSON; </script guarded so the blob can't break out of its tag.
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = template.replace(_UPLOT_CSS_PLACEHOLDER, uplot_css)
    html = html.replace(_UPLOT_JS_PLACEHOLDER, uplot_js)
    html = html.replace(_DATA_PLACEHOLDER, blob)
    return html


def build_dashboard(
    traces: list[Trace],
    scenarios: dict[str, Scenario],
    *,
    control: str = "control",
    buyer_control: str = "neutral",
    n_boot: int = 2000,
    seed: int = 0,
    source: str | None = None,
) -> str:
    """Score traces and return the full self-contained dashboard HTML."""
    payload = build_payload(
        traces,
        scenarios,
        control=control,
        buyer_control=buyer_control,
        n_boot=n_boot,
        seed=seed,
        source=source,
    )
    return render_html(payload)
