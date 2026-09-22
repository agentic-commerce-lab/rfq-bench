"""Shared, deterministic benchmark core.

`core/` owns the scientific kernel: contracts, outcome space, utility/BATNA,
ZOPA, and scoring. It must stay free of NegMAS runtime and LLM I/O so scoring
can be recomputed from an immutable trace, offline and deterministically.
"""
