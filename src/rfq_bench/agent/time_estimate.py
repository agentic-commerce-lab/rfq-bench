"""Pre-run wall-clock estimate for runs that call a model (llm / a2a / laya).

Built purely from **run history**: past traces in ``results/*.jsonl`` record each
episode's ``latency_s`` and its steps, so seconds per model call are measured,
not assumed. Only steps by a side that reports token usage count as calls (a
scripted opponent answers instantly). Errored episodes are ignored.

Two numbers come out:

- **typical** — per scenario, the mean number of calls past episodes of that
  scenario needed (agreements end early); scenarios without history fall back to
  the ceiling.
- **ceiling** — every episode runs to its deadline (the cost estimate's call count).

Both are divided across ``--concurrency`` workers and never drop below the
``--stagger`` ramp. Parallel calls can be slower than serial ones under provider
rate limits, and re-asks/retries come on top, so this is a rough guide; the
progress bar's live ETA takes over once the run starts.

When the model has no history, the rate of other (non-loopback) models is used
and labelled as such — model speed varies widely, so treat it as an order of
magnitude.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from rfq_bench.agent.cost_estimate import calls_ceiling

if TYPE_CHECKING:
    from rfq_bench.runners.offline import EpisodeSpec

_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})


@dataclass(frozen=True)
class CallHistory:
    """Measured call timings for one model (or the fallback pool)."""

    seconds: float  # summed episode latency
    calls: int  # summed LLM-side steps
    episodes: int
    # scenario id -> (episodes, LLM calls) for that scenario, per side count.
    by_scenario: dict[str, tuple[int, int]]
    files: tuple[str, ...]

    @property
    def s_per_call(self) -> float:
        return self.seconds / self.calls


@dataclass(frozen=True)
class TimeEstimate:
    model: str
    typical_s: float
    ceiling_s: float
    s_per_call: float
    basis: str
    same_model: bool


def call_history(
    model: str | None, results_dir: Path, *, exclude_model: str | None = None
) -> CallHistory | None:
    """Timings from past non-errored traces.

    ``model`` set: only that model's traces. ``model=None``: every remote model
    except ``exclude_model`` (the fallback pool; loopback servers like Laya are
    excluded since they say nothing about a remote API's speed).
    """
    seconds = 0.0
    calls = episodes = 0
    by_scenario: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    files: set[str] = set()
    for path in sorted(results_dir.glob("*.jsonl")):
        try:
            fh = path.open(encoding="utf-8")
        except OSError:
            continue
        with fh:
            for line in fh:
                if model is not None and model not in line:  # cheap pre-filter
                    continue
                try:
                    trace = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cfg = trace.get("llm_config") or {}
                m = cfg.get("model")
                role_models = trace.get("models_by_role") or {}
                if model is not None and m != model and model not in role_models.values():
                    continue
                if model is None:
                    host = urlparse(str(cfg.get("base_url") or "")).hostname or ""
                    if m is None or m == exclude_model or host in _LOOPBACK:
                        continue
                if trace.get("errored"):
                    continue
                latency = trace.get("latency_s")
                llm_roles = _llm_roles(trace)
                parties = [s.get("party") for s in trace.get("steps") or []]
                n_all = sum(1 for p in parties if p in llm_roles)
                if model is not None and role_models:
                    # Duel: count only this model's side(s); attribute the episode's
                    # latency in proportion to its share of the calls.
                    llm_roles = {r for r in llm_roles if role_models.get(r, m) == model}
                n = sum(1 for p in parties if p in llm_roles)
                if not n or not isinstance(latency, int | float) or latency <= 0:
                    continue
                seconds += float(latency) * n / n_all
                calls += n
                episodes += 1
                files.add(path.name)
                # Normalise to calls per LLM side, so llm and a2a history mix.
                per_side = n / max(1, len(llm_roles))
                agg = by_scenario[str(trace.get("scenario_id"))]
                agg[0] += 1
                agg[1] += round(per_side)
    if calls == 0:
        return None
    return CallHistory(
        seconds=seconds,
        calls=calls,
        episodes=episodes,
        by_scenario={k: (v[0], v[1]) for k, v in by_scenario.items()},
        files=tuple(sorted(files)),
    )


def _llm_roles(trace: dict[str, Any]) -> set[str]:
    """Sides that called a model: those with token usage (a scripted side reports 0).

    Older traces lack ``token_cost_by_role``; there the target is the model side,
    plus the buyer persona in A2A.
    """
    by_role = trace.get("token_cost_by_role")
    if isinstance(by_role, dict):
        return {r for r, t in by_role.items() if t}
    roles = {str(trace.get("target_role"))}
    if trace.get("persona"):
        roles |= {"buyer", "seller"}
    return roles


def split_time_estimate(parts: list[TimeEstimate | None]) -> TimeEstimate | None:
    """Combine per-model estimates for a duel, where each model makes half the calls."""
    ests = [e for e in parts if e is not None]
    if len(ests) != len(parts) or not ests:
        return None
    k = len(ests)
    return TimeEstimate(
        model=" + ".join(e.model for e in ests),
        typical_s=sum(e.typical_s for e in ests) / k,
        ceiling_s=sum(e.ceiling_s for e in ests) / k,
        s_per_call=sum(e.s_per_call for e in ests) / k,
        basis="half the calls per model — " + "; ".join(e.basis for e in ests),
        same_model=all(e.same_model for e in ests),
    )


def estimate_run_time(
    specs: Iterable[EpisodeSpec],
    *,
    agent: str,
    model: str,
    max_rounds: int | None,
    concurrency: int,
    stagger_s: float,
    results_dir: str | Path = "results",
) -> TimeEstimate | None:
    """Estimated wall-clock for the run, or None without any usable history."""
    if agent not in ("llm", "a2a", "laya"):
        return None
    specs = list(specs)
    if not specs:
        return None
    results_dir = Path(results_dir)
    hist = call_history(model, results_dir)
    same_model = hist is not None
    if hist is None and agent != "laya":
        hist = call_history(None, results_dir, exclude_model=model)
    if hist is None:
        return None

    sides = 2 if agent == "a2a" else 1
    kind = "a2a" if agent == "a2a" else "llm"  # laya: one model side, like llm
    ceiling_calls = calls_ceiling(specs, agent=kind, max_rounds=max_rounds)
    typical_calls = 0.0
    for spec in specs:
        cap = calls_ceiling([spec], agent=kind, max_rounds=max_rounds)
        seen = hist.by_scenario.get(spec.scenario.id)
        if seen and seen[0]:
            typical_calls += min(cap, seen[1] / seen[0] * sides)
        else:
            typical_calls += cap

    workers = max(1, min(concurrency, len(specs)))
    ramp = max(0.0, stagger_s) * (len(specs) - 1)

    def wall(n_calls: float) -> float:
        return max(n_calls * hist.s_per_call / workers, ramp)

    who = model if same_model else "other models (no history for this one)"
    basis = (
        f"{hist.s_per_call:.1f}s/call measured over {hist.calls:,} calls "
        f"({hist.episodes:,} episodes) of {who} in {', '.join(hist.files)}"
    )
    return TimeEstimate(
        model=model,
        typical_s=wall(typical_calls),
        ceiling_s=wall(ceiling_calls),
        s_per_call=hist.s_per_call,
        basis=basis,
        same_model=same_model,
    )
