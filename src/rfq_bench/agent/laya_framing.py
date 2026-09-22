"""Turn a numeric NegotiationState into Laya-friendly prose + typed questions.

Laya is an encoder doing textual-entailment-like matching: it rewards questions
about *what the text says* and states put *into words, not numbers*. So this module
verbalizes the party's own business view (reusing ``framing``) and, crucially,
does every numeric comparison in code — handing Laya a conclusion ("this offer is
a little above your bottom line") rather than utilities to compare.

Leakage rule is unchanged: only the party's own business view; never raw
utilities/weights/BATNAs, never the opponent's private values or role name.
"""

from __future__ import annotations

from typing import Any

from rfq_bench.agent.framing import priorities_view, walk_away_view
from rfq_bench.core.contracts import Issue, PartyPreferences
from rfq_bench.core.utility import ideal_utility, outcome_utility, reservation_utility
from rfq_bench.strategies.base import NegotiationState


def _phase(state: NegotiationState) -> str:
    p = state.progress
    if p < 0.34:
        return "It is early in the negotiation; there is time to hold out."
    if p < 0.75:
        return "The negotiation is about midway; some give and take is expected."
    return "The deadline is close; failing to agree soon means no deal."


def _describe_offer(outcome: dict[str, Any], issues: list[Issue]) -> str:
    parts = []
    for i in issues:
        if i.name in outcome:
            unit = f" {i.unit}" if i.unit else ""
            parts.append(f"{i.name.replace('_', ' ')} of {outcome[i.name]}{unit}")
    return "; ".join(parts) if parts else "an incomplete offer"


def offer_verdict(state: NegotiationState, standing_offer: dict[str, Any]) -> str:
    """Code-computed comparison of the offer to this party's own value range."""
    prefs, issues = state.prefs, state.issues
    u = outcome_utility(prefs, issues, standing_offer)
    d = reservation_utility(prefs)
    ideal = ideal_utility(prefs, issues)
    span = ideal - d
    eps = 1e-9
    if u < d - eps:
        base = "This offer is worse for you than your bottom line; you cannot accept it as is."
    elif span <= eps or u <= d + 0.15 * span:
        base = "This offer only barely meets your bottom line."
    elif u >= d + 0.7 * span:
        base = "This offer is very favourable to you — close to the best you could hope for."
    else:
        base = "This offer is comfortably above your bottom line but not the best you could get."
    # Trend: did their latest offer improve on their previous one (for you)?
    offers = state.opponent_offers
    if len(offers) >= 2:
        prev = outcome_utility(prefs, issues, offers[-2])
        if u > prev + eps:
            base += " They have moved in your favour since their last offer."
        elif u < prev - eps:
            base += " Their offer has gotten worse for you than before."
        else:
            base += " They have not moved since their last offer."
    return base


def laya_state(
    state: NegotiationState,
    standing_offer: dict[str, Any] | None,
    approach: str | None = None,
) -> dict[str, Any]:
    """The business-terms, number-free state payload the model may see.

    ``approach`` is the optional strategy-guidance text (the same one-liner the LLM
    agent gets), injected so the strategy can bias Laya's own offer choice.
    """
    prefs, issues = state.prefs, state.issues
    if standing_offer is not None:
        their_offer = _describe_offer(standing_offer, issues)
        assessment = offer_verdict(state, standing_offer)
    else:
        their_offer = "No offer yet — it is your turn to put the first offer on the table."
        assessment = "There is no offer to assess yet."
    payload: dict[str, Any] = {
        "your_role": state.role,
        "your_priorities": priorities_view(prefs, issues),
        "your_bottom_line": walk_away_view(prefs, issues),
        "timing": _phase(state),
        "the_other_partys_current_offer": their_offer,
        "assessment_of_their_offer": assessment,
    }
    if approach:
        payload["your_negotiation_approach"] = approach
    return payload


def _valence_labels(n: int) -> list[str]:
    """Best-first qualitative standings for ``n`` ranked options (this party's view)."""
    if n == 1:
        return ["your only option"]
    labels = []
    for rank in range(n):
        if rank == 0:
            labels.append("the best for you")
        elif rank == n - 1:
            labels.append("the least favourable you would accept")
        elif rank < (n - 1) / 2:
            labels.append("a good one for you")
        else:
            labels.append("a weak one for you")
    return labels


def _issue_choice(issue: Issue, prefs: PartyPreferences) -> dict[str, Any]:
    """A `choice` asking which value of one issue to offer.

    Each option is described by what it means **for this party** (best → least
    favourable), from the party's own preference order — the same own-view information
    already in ``your_priorities``, never the opponent's values. Without this valence
    the model anchors to whatever value is in the text (it picks the low price as a
    seller); with it, "the best for you" entails the value this party actually wants.
    """
    ip = prefs.preference_for(issue.name)
    ranked = sorted(
        zip(issue.values, ip.value_utilities, strict=True), key=lambda p: -p[1]
    )  # best first
    standing = {
        value: label
        for (value, _u), label in zip(ranked, _valence_labels(len(ranked)), strict=True)
    }
    unit = f" {issue.unit}" if issue.unit else ""
    label = issue.name.replace("_", " ")
    return {
        "type": "choice",
        "instructions": (
            f"For your next offer to the other party, which {label} do you propose? "
            f"Aim for the terms best for you, conceding only as needed to reach a deal."
        ),
        "criteria": {str(v): f"a {label} of {v}{unit} — {standing[v]}" for v in issue.values},
    }


def laya_questions(issues: list[Issue], prefs: PartyPreferences) -> dict[str, Any]:
    """All decisions in one forward pass: accept? still reachable? and the offer itself.

    Laya chooses every issue's value directly (one `choice` per issue), so the offer
    is Laya's decision — not a code-side concession schedule. ``prefs`` is used only to
    label each option by its standing for this party (its own view).
    """
    questions: dict[str, Any] = {
        "accept": {
            "type": "noul",
            "instructions": (
                "Is the other party's current offer good enough that you should accept it now "
                "rather than push for better terms?"
            ),
        },
        "reachable": {
            "type": "noul",
            "instructions": (
                "Is a deal you would be willing to accept still realistically within reach "
                "before the deadline?"
            ),
        },
    }
    for issue in issues:
        questions[issue.name] = _issue_choice(issue, prefs)
    return questions
