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


def issues_view(prefs: PartyPreferences, issues: list[Issue]) -> list[dict[str, Any]]:
    """Compact per-issue view for the LLM payload: each option list sent once.

    ``options_best_first`` is both the legal set and this party's preference
    order. ``importance`` only appears when there is more than one issue (with
    one issue it says nothing). Utilities and weights stay hidden, as above.
    """
    total = sum(ip.weight for ip in prefs.issues)
    view: list[dict[str, Any]] = []
    for issue in issues:
        entry: dict[str, Any] = {"name": issue.name}
        if issue.unit:
            entry["unit"] = issue.unit
        entry["options_best_first"] = _preferred_order(issue, prefs)
        if len(issues) > 1:
            entry["importance"] = _priority_label(prefs.preference_for(issue.name).weight, total)
        view.append(entry)
    return view


def limit_direction(prefs: PartyPreferences, issue: Issue) -> str | None:
    """Which way a single-issue walk-away binds: "at_most", "at_least", or None.

    The direction follows the party's preference: if it prefers lower values (a
    buyer on price), the limit is a ceiling, "at_most"; if higher (a seller), a
    floor, "at_least". None when the values aren't numeric, the preference isn't
    one-directional, or the acceptable set isn't all on the preferred side of the
    limit — then no single direction describes it honestly.
    """
    view = walk_away_view(prefs, [issue])
    limit = view["break_even"]
    ip = prefs.preference_for(issue.name)
    pairs = list(zip(issue.values, ip.value_utilities, strict=True))
    if limit is None or not all(isinstance(v, int | float) for v, _ in pairs):
        return None
    ranked = sorted(pairs, key=lambda p: p[0])  # by value, low → high
    utils = [u for _, u in ranked]
    if all(a >= b for a, b in zip(utils, utils[1:], strict=False)) and utils[0] > utils[-1]:
        direction = "at_most"  # prefers lower values: the limit is a ceiling
    elif all(a <= b for a, b in zip(utils, utils[1:], strict=False)) and utils[-1] > utils[0]:
        direction = "at_least"  # prefers higher values: the limit is a floor
    else:
        return None
    d = reservation_utility(prefs)
    acceptable = [v for v, u in pairs if u >= d - 1e-9]
    if direction == "at_most":
        return direction if all(v <= limit for v in acceptable) else None
    return direction if all(v >= limit for v in acceptable) else None


def walk_away_terms(prefs: PartyPreferences, issues: list[Issue]) -> dict[str, Any]:
    """The walk-away as bare terms; the system prompt explains how to read it.

    Single issue: a directional limit, ``{issue: {"at_most": v}}`` (a buyer's
    ceiling) or ``{issue: {"at_least": v}}`` (a seller's floor), so the model can't
    misread which side of the number is acceptable; ``{"worst_acceptable": ...}``
    only when no single direction applies. Several issues:
    ``{"break_even_package": {...}}``. Same values as :func:`walk_away_view`.
    """
    view = walk_away_view(prefs, issues)
    if view["kind"] == "limit":
        issue = issues[0]
        direction = limit_direction(prefs, issue)
        if direction is not None:
            return {issue.name: {direction: view["break_even"]}}
        return {"worst_acceptable": {view["issue"]: view["break_even"]}}
    return {"break_even_package": view["break_even_package"]}


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
