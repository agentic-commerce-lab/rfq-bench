"""The four deterministic opponent policies."""

from __future__ import annotations

from collections.abc import Callable

from rfq_bench.strategies.base import ConcessionPolicy, NegotiationPolicy, TitForTatPolicy

_FACTORIES: dict[str, Callable[[], NegotiationPolicy]] = {
    # Barely concedes; holds an extreme target until the very end.
    "hardliner": lambda: ConcessionPolicy("hardliner", opening=1.0, beta=0.02),
    # Concedes quickly toward its reservation value.
    "fast_conceder": lambda: ConcessionPolicy("fast_conceder", opening=0.9, beta=6.0),
    # Reciprocates the agent's movement.
    "reciprocal": lambda: TitForTatPolicy(name="reciprocal"),
    # Trades across issues, seeking mutually beneficial packages.
    "integrative": lambda: ConcessionPolicy("integrative", opening=0.9, beta=1.0, integrative=True),
}

OPPONENTS: tuple[str, ...] = tuple(_FACTORIES)


def get_opponent(name: str) -> NegotiationPolicy:
    try:
        return _FACTORIES[name]()
    except KeyError:
        raise KeyError(f"unknown opponent {name!r}; known: {', '.join(OPPONENTS)}") from None
