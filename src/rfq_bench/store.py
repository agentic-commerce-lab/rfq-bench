"""Immutable trace storage as JSON Lines.

Traces are append-only: one JSON object per line, written once and never
mutated. Scoring and reporting read them back without re-running episodes.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from rfq_bench.core.contracts import Trace


class TraceWriter:
    """Append traces to a JSONL file. Existing lines are never rewritten."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, trace: Trace) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(trace.model_dump_json() + "\n")

    def extend(self, traces: Iterable[Trace]) -> int:
        n = 0
        with self.path.open("a", encoding="utf-8") as fh:
            for trace in traces:
                fh.write(trace.model_dump_json() + "\n")
                n += 1
        return n


def read_traces(path: str | Path) -> Iterator[Trace]:
    """Yield traces from a JSONL file."""
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield Trace.model_validate_json(line)
