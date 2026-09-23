"""Plain-language descriptions of strategies, opponents/personas and scenarios.

Shown as tooltips in the dashboard (hover a name). Written for readers who are not
negotiation researchers: what the thing *does*, in one or two short sentences.
Strategy and opponent texts are hand-written to match how each is implemented
(``strategies/registry.py``, ``opponents/registry.py``, the persona guidance);
scenario texts are generated from the scenario data, so new scenarios get one too.

The dashboard is an offline analysis artifact, never shown to a model, so a
scenario description may state both sides' walk-aways.
"""

from __future__ import annotations

from typing import Any

from rfq_bench.agent.framing import walk_away_view
from rfq_bench.core.contracts import Role, Scenario
from rfq_bench.core.outcome_space import enumerate_outcomes
from rfq_bench.core.zopa import compute_zopa

STRATEGY_DESCRIPTIONS: dict[str, str] = {
    "control": (
        "The baseline. Negotiates sensibly toward a fair deal, with no special tactic. "
        "Every other strategy is measured against this one (the Δ column)."
    ),
    "boulware": (
        "Plays tough: stays close to its best terms for most of the negotiation and "
        "only gives ground near the deadline."
    ),
    "linear": (
        "Gives ground at a steady pace, a little each round, from its opening offer "
        "toward its walk-away."
    ),
    "conceder": "Gives ground fast and early, to reach a deal quickly.",
    "tit_for_tat": (
        "Mirrors the other side: starts friendly, then moves when they move and holds "
        "when they hold."
    ),
    "anchoring": (
        "Opens with a bold but realistic first offer to set the frame, then gives "
        "ground steadily in small steps."
    ),
    "logrolling": (
        "Trades between issues: gives way on what matters less to it to win on what "
        "matters more. Only used when there are several issues."
    ),
}

OPPONENT_DESCRIPTIONS: dict[str, str] = {
    # Scripted opponents (--agent scripted / llm): fixed rules, no language model.
    "hardliner": ("Scripted, very tough: barely moves from its best terms until the very end."),
    "fast_conceder": "Scripted, soft: gives ground quickly toward its walk-away.",
    "reciprocal": ("Scripted, mirrors you: concedes about as much as you just conceded."),
    "integrative": ("Scripted, cooperative: looks for package deals that are good for both sides."),
    # LLM buyer personas (--agent a2a).
    "neutral": (
        "Baseline buyer: businesslike, aims for a fair deal, no special attitude. The "
        "other buyer personas are compared against it."
    ),
    "cost_focused": (
        "Cost-focused buyer: price comes first. Opens ambitiously, concedes slowly, "
        "keeps its limit to itself, and would rather walk away than overpay."
    ),
    "total_value": (
        "Total-value buyer: judges the whole package (price, delivery, warranty "
        "together) and looks for trades that help both sides."
    ),
    "reliability_first": (
        "Reliability-first buyer: dependable delivery and a strong warranty matter more "
        "than the lowest price; it will pay more to avoid risk."
    ),
    "relationship_focused": (
        "Relationship buyer: wants a reliable long-term supplier. Stays fair and open, "
        "avoids aggressive tactics, and values sustainable deals over maximum savings."
    ),
    "bargain_hunter": "Generic buyer fixated on the lowest price; pushes hard for discounts.",
    "time_pressured": (
        "Time-pressured buyer: delay is costly, so it prefers a quick acceptable deal over "
        "the best price, and pays more the closer the deadline gets."
    ),
    "supply_security": (
        "Supply-security buyer (the price-only risk persona): ending without a deal is the "
        "loss to avoid, so it pays a premium for a certain agreement, within its limit."
    ),
    "relationship_builder": (
        "Generic cooperative buyer: wants a good working relationship and fair trade-offs."
    ),
    "hardball": (
        "Generic aggressive buyer: ambitious opening, small concessions, ready to "
        "threaten walking away."
    ),
    "risk_averse": (
        "Generic cautious buyer: prefers safe, certain terms over pushing for a better deal."
    ),
}


def _first_sentence(text: str, limit: int = 220) -> str:
    """A short fallback description from a custom guidance file's text."""
    flat = " ".join(text.split())
    end = flat.find(". ")
    first = flat[: end + 1] if 0 <= end < limit else flat[:limit]
    return first if len(first) == len(flat) or first.endswith(".") else first + "…"


def _limit(scenario: Scenario, role: Role) -> Any:
    return walk_away_view(scenario.preferences(role), list(scenario.issues)).get("break_even")


def scenario_description(scenario: Scenario) -> str:
    """What the scenario is and what makes it easy or hard, from its own data."""
    issues = list(scenario.issues)
    zopa = compute_zopa(scenario)
    head = f"{scenario.product[:1].upper()}{scenario.product[1:]}."
    if len(issues) == 1:
        issue = issues[0]
        unit = f" {issue.unit}" if issue.unit else ""
        values = list(issue.values)
        b, s = _limit(scenario, "buyer"), _limit(scenario, "seller")
        span = f"{issue.name.replace('_', ' ')} only, options {values[0]}–{values[-1]}{unit}"
        utilities = scenario.buyer.preference_for(issue.name).value_utilities
        buyer_wants_low = utilities[0] > utilities[-1]  # values are listed low → high
        limits = (
            f"The buyer pays at most {b}, the seller needs at least {s}"
            if buyer_wants_low
            else f"The buyer accepts no worse than {b}, the seller no worse than {s}"
        )
        if zopa.exists:
            n = len(zopa.outcomes)
            fit = f"so any {issue.name} from {s} to {b} works for both ({n} option{'s' * (n > 1)})."
        else:
            fit = f"so no {issue.name} works for both: walking away is the right outcome."
        return f"{head} {span[:1].upper()}{span[1:]}. {limits}, {fit}"

    names = ", ".join(i.name.replace("_", " ") for i in issues)

    def ranking(role: Role) -> list[str]:
        prefs = sorted(scenario.preferences(role).issues, key=lambda ip: -ip.weight)
        return [ip.issue.replace("_", " ") for ip in prefs]

    buyer, seller = ranking("buyer"), ranking("seller")
    if buyer[0] != seller[0]:
        trade = (
            f"The buyer cares most about {buyer[0]}, the seller about {seller[0]}, "
            "which leaves room to trade."
        )
    elif buyer != seller:
        trade = (
            f"Both care most about {buyer[0]}; next, the buyer values {buyer[1]} and the "
            f"seller {seller[1]}, which leaves room to trade."
        )
    else:
        trade = "Both rank the issues the same way, so there is little room to trade."

    total = len(enumerate_outcomes(issues))
    if zopa.exists:
        n = len(zopa.outcomes)
        fit = f"{n} of the {total} possible packages work for both sides."
    else:
        fit = "No package works for both sides: walking away is the right outcome."
    return f"{head} Negotiated on {names}. {trade} {fit}"


def build_glossary(
    *,
    strategies: list[str],
    opponents: list[str],
    scenarios: list[Scenario],
    strategy_guidance: dict[str, str] | None = None,
    persona_guidance: dict[str, str] | None = None,
) -> dict[str, dict[str, str]]:
    """Descriptions for the names used in a run; custom ones fall back to guidance text."""

    def describe(name: str, known: dict[str, str], guidance: dict[str, str] | None) -> str:
        if name in known:
            return known[name]
        if guidance and name in guidance:
            return "Custom: " + _first_sentence(guidance[name])
        return ""

    return {
        "strategy": {s: describe(s, STRATEGY_DESCRIPTIONS, strategy_guidance) for s in strategies},
        "opponent": {o: describe(o, OPPONENT_DESCRIPTIONS, persona_guidance) for o in opponents},
        "scenario": {s.id: scenario_description(s) for s in scenarios},
    }
