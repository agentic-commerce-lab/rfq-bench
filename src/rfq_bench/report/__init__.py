"""Aggregation and reporting.

The strategy score is reported separately from outcome-quality, validity/safety,
and operational metrics — never folded together.
"""

from __future__ import annotations

from rfq_bench.report.aggregate import Report, build_report
from rfq_bench.report.dashboard import build_dashboard, build_payload

__all__ = ["Report", "build_dashboard", "build_payload", "build_report"]
