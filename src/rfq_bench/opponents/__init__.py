"""Deterministic, seeded scripted opponents.

Opponents reuse the concession machinery from ``strategies.base`` so both sides
of the table are symmetric and reproducible. LLM opponents are out of scope for
the v0 causal comparison.
"""

from __future__ import annotations

from rfq_bench.opponents.registry import OPPONENTS, get_opponent

__all__ = ["OPPONENTS", "get_opponent"]
