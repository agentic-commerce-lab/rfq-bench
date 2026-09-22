"""Strategy arms — the only experimental treatment.

Each strategy is a deterministic concession policy with an explicit opening
offer, concession schedule, acceptance rule, and tie-breaking rule. Only the
strategy changes between matched runs; everything else is frozen.
"""

from __future__ import annotations

from rfq_bench.strategies.base import NegotiationPolicy, NegotiationState
from rfq_bench.strategies.registry import STRATEGIES, get_strategy

__all__ = ["NegotiationPolicy", "NegotiationState", "STRATEGIES", "get_strategy"]
