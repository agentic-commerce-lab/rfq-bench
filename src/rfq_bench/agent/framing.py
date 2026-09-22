"""Translate a party's hidden utility model into business terms for the prompt.

A real negotiator does not have a 0..1 utility for every option — they have
concrete options, a sense of which matter most, a preference direction, and a
walk-away (their bottom line). This module produces exactly that view, so the LLM
reasons like a rep instead of reading a utility table. The exact utilities and
weights never leave this module (scoring still uses them, off the trace).
"""

from __future__ import annotations

from typing import Any

from rfq_bench.core.contracts import Issue, PartyPreferences
from rfq_bench.core.utility import reservation_utility
from rfq_bench.strategies.base import select_outcome


def _priority_label(weight: float, total: float) -> str:
    """Coarse importance bucket from an issue's share of the total weight."""
    share = weight / total if total > 0 else 0.0
    if share >= 0.4:
        return "high"
    if share <= 0.2:
        return "low"
    return "medium"


def _preferred_order(issue: Issue, prefs: PartyPreferences) -> list[Any]:
    """The issue's options ranked best-first for this party (utilities hidden)."""
    ip = prefs.preference_for(issue.name)
    ranked = sorted(zip(issue.values, ip.value_utilities, strict=True), key=lambda p: -p[1])
    return [value for value, _ in ranked]


def priorities_view(prefs: PartyPreferences, issues: list[Issue]) -> list[dict[str, Any]]:
    """Per-issue business view: options, preference order, and importance."""
    total = sum(ip.weight for ip in prefs.issues)
    view: list[dict[str, Any]] = []
    for issue in issues:
        ip = prefs.preference_for(issue.name)
        view.append(
            {
                "issue": issue.name,
                "unit": issue.unit,
                "options": list(issue.values),
                "you_prefer_in_order": _preferred_order(issue, prefs),  # best first
                "importance": _priority_label(ip.weight, total),
            }
        )
    return view


def walk_away_view(prefs: PartyPreferences, issues: list[Issue]) -> dict[str, Any]:
    """The party's bottom line in business terms — never a utility number.

    Single issue: the least-favorable value it can still accept (a concrete floor,
    e.g. a seller's lowest sellable price). Multiple issues: a concrete break-even
    reference package, since a combined threshold has no single per-issue value.
    """
    d = reservation_utility(prefs)
    if len(issues) == 1:
        issue = issues[0]
        ip = prefs.preference_for(issue.name)
        acceptable = [
            (value, u)
            for value, u in zip(issue.values, ip.value_utilities, strict=True)
            if u >= d - 1e-9
        ]
        # The acceptable option closest to the floor is the break-even limit.
        limit = min(acceptable, key=lambda p: p[1])[0] if acceptable else None
        return {
            "kind": "limit",
            "issue": issue.name,
            "break_even": limit,
            "note": (
                f"Your bottom line: the least favourable {issue.name} you may agree to is "
                f"{limit}. Do not accept anything worse for you than that."
            ),
        }
    package = select_outcome(prefs, issues, d)
    return {
        "kind": "reference_package",
        "break_even_package": package,
        "note": (
            "Your bottom line: a deal you would consider roughly break-even is the package "
            "below. Only agree to deals that are clearly better for you than this."
        ),
    }
