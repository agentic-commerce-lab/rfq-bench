"""The closed outcome space and enumeration helpers.

The v0 outcome space is a Cartesian product of discrete, ordered issue values.
It is small enough to enumerate exhaustively, which the ZOPA and ideal-point
computations rely on.
"""

from __future__ import annotations

from itertools import product
from typing import Any

from rfq_bench.core.contracts import Issue

OUTCOME_SPACE_VERSION = "v0.1"


def enumerate_outcomes(issues: list[Issue]) -> list[dict[str, Any]]:
    """Return every legal outcome as an issue-name -> value mapping."""
    names = [i.name for i in issues]
    value_lists = [i.values for i in issues]
    return [dict(zip(names, combo, strict=True)) for combo in product(*value_lists)]


def is_legal(issues: list[Issue], outcome: dict[str, Any]) -> bool:
    """True iff ``outcome`` assigns exactly the issues, each to a legal value."""
    by_name = {i.name: i for i in issues}
    if set(outcome) != set(by_name):
        return False
    return all(value in by_name[name].values for name, value in outcome.items())
