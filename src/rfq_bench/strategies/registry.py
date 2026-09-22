"""The seven literature-backed strategy arms.

Openings and betas are chosen so the arms are behaviorally distinct while every
other benchmark dimension stays frozen. Anchoring shares Linear's schedule but
opens at the extreme (opening = 1.0) to isolate the first-offer anchor effect.
"""

from __future__ import annotations

from collections.abc import Callable

from rfq_bench.strategies.base import ConcessionPolicy, NegotiationPolicy, TitForTatPolicy

# Each factory builds a fresh policy instance (policies may carry per-episode state).
_FACTORIES: dict[str, Callable[[], NegotiationPolicy]] = {
    # Neutral baseline: moderate opening, steady concession, no anchoring.
    "control": lambda: ConcessionPolicy("control", opening=0.85, beta=1.0),
    # Hold near the opening, concede sharply near the deadline.
    "boulware": lambda: ConcessionPolicy("boulware", opening=1.0, beta=0.05),
    # Steady, approximately constant-rate concession.
    "linear": lambda: ConcessionPolicy("linear", opening=0.9, beta=1.0),
    # Concede rapidly early.
    "conceder": lambda: ConcessionPolicy("conceder", opening=0.9, beta=5.0),
    # Cooperative open, then reciprocate concessions.
    "tit_for_tat": lambda: TitForTatPolicy(),
    # Extreme-but-feasible anchor, then a fixed linear schedule.
    "anchoring": lambda: ConcessionPolicy("anchoring", opening=1.0, beta=1.0),
    # Trade issues across priorities toward the Pareto frontier (multi-issue).
    "logrolling": lambda: ConcessionPolicy("logrolling", opening=0.9, beta=1.0, integrative=True),
}

STRATEGIES: tuple[str, ...] = tuple(_FACTORIES)

# Strategies that only make sense with more than one issue.
MULTI_ISSUE_ONLY: frozenset[str] = frozenset({"logrolling"})


def get_strategy(name: str) -> NegotiationPolicy:
    try:
        return _FACTORIES[name]()
    except KeyError:
        raise KeyError(f"unknown strategy {name!r}; known: {', '.join(STRATEGIES)}") from None
