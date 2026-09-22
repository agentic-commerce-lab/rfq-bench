"""Trace round-trips through the JSONL store unchanged."""

from __future__ import annotations

from rfq_bench.runners.offline import EpisodeSpec, run_episode
from rfq_bench.store import TraceWriter, read_traces


def test_trace_roundtrip(tmp_path, single_issue_scenario) -> None:
    spec = EpisodeSpec(
        scenario=single_issue_scenario,
        strategy="linear",
        opponent="fast_conceder",
        target_role="buyer",
        first_speaker="buyer",
        seed=0,
    )
    trace = run_episode(spec)
    path = tmp_path / "traces.jsonl"
    writer = TraceWriter(path)
    writer.append(trace)
    writer.append(trace)

    loaded = list(read_traces(path))
    assert len(loaded) == 2
    assert loaded[0] == trace
    assert loaded[0].model_dump() == trace.model_dump()
