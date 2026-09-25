"""Pre-run cost estimate for runs that call a model (``--agent llm`` / ``--agent a2a``).

The estimate is a **ceiling**: every episode is assumed to run to its deadline,
where each round costs one model call per LLM side (two in A2A, one with
``--agent llm``). Agreements end episodes early, so real spend is usually lower;
re-asks and transient retries come on top. The price per call comes from, in
order of preference:

1. **Run history** — past traces in ``results/*.jsonl`` for the same model. Their
   ``cost_usd_by_role`` is the provider-reported spend, so this already includes
   reasoning tokens, re-asks, and the real prompt size.
2. **Published pricing** — OpenRouter's ``/models`` $/token, applied to the prompt
   size measured from the actual system prompt, tool schema and payload, plus an
   assumed completion size (a reasoning model can use far more, up to max_tokens,
   so that worst case is reported alongside).

Network calls (pricing, remaining credit) are OpenRouter-only, short-timeout and
fail-soft: an estimate must never block or break a run.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from rfq_bench.agent.prompts import build_move_tool, build_user_payload
from rfq_bench.agent.settings import AgentSettings
from rfq_bench.strategies.base import NegotiationState

if TYPE_CHECKING:
    from rfq_bench.core.contracts import Role, Scenario
    from rfq_bench.runners.offline import EpisodeSpec

# (url, headers) -> parsed JSON. Injectable so tests never touch the network.
FetchJson = Callable[[str, dict[str, str]], Any]

_TIMEOUT_S = 5.0
# JSON-heavy payloads tokenize denser than prose (~4 chars/token). Calibrated on the
# DeepSeek price-v1 run with the earlier (spaced, duplicated) payload: 1,190 real
# tokens/call vs ~1,240 estimated. Only used for models without run history.
_CHARS_PER_TOKEN = 3.0
# A tool call with a one-line rationale; reasoning models can use far more.
ASSUMED_COMPLETION_TOKENS = 200
# Typical length of a strategy/persona guidance block (the grounded persona files).
_INSTRUCTION_CHARS = 400


@dataclass(frozen=True)
class CostEstimate:
    model: str
    calls_max: int  # every episode runs to its deadline; excludes re-asks/retries
    usd_per_call: float | None
    basis: str
    # Pricing basis only: every call spends max_tokens on completion.
    usd_per_call_worst: float | None = None
    credit_remaining: float | None = None

    @property
    def max_usd(self) -> float | None:
        return None if self.usd_per_call is None else self.calls_max * self.usd_per_call

    @property
    def worst_usd(self) -> float | None:
        if self.usd_per_call_worst is None:
            return None
        return self.calls_max * self.usd_per_call_worst


def estimate_run_cost(
    specs: Iterable[EpisodeSpec],
    *,
    agent: str,
    max_rounds: int | None,
    settings: AgentSettings,
    results_dir: str | Path = "results",
    fetch: FetchJson | None = None,
) -> CostEstimate | None:
    """Estimate a run's model spend, or None when the agent makes no paid calls."""
    if agent not in ("llm", "a2a"):
        return None
    specs = list(specs)
    fetch = fetch or _http_json
    calls = calls_ceiling(specs, agent=agent, max_rounds=max_rounds)
    online = _is_openrouter(settings)
    credit = credit_remaining(settings, fetch) if online else None

    hist = history_rate(settings.model, Path(results_dir))
    if hist is not None:
        usd, n, files = hist
        return CostEstimate(
            model=settings.model,
            calls_max=calls,
            usd_per_call=usd,
            basis=f"${usd:.6f}/call measured over {n:,} past calls in {', '.join(files)}",
            credit_remaining=credit,
        )

    pricing = model_pricing(settings, fetch) if online else None
    if pricing is None:
        return CostEstimate(
            model=settings.model,
            calls_max=calls,
            usd_per_call=None,
            basis="no run history for this model and no published pricing from the endpoint",
            credit_remaining=credit,
        )
    p_in, p_out = pricing
    prompt = mean_prompt_tokens(specs, settings, max_rounds=max_rounds)
    return CostEstimate(
        model=settings.model,
        calls_max=calls,
        usd_per_call=prompt * p_in + ASSUMED_COMPLETION_TOKENS * p_out,
        usd_per_call_worst=prompt * p_in + settings.max_tokens * p_out,
        basis=(
            f"published pricing ${p_in * 1e6:.3g}/M in, ${p_out * 1e6:.3g}/M out; "
            f"~{prompt:,.0f} prompt tokens/call measured, ~{ASSUMED_COMPLETION_TOKENS} "
            "completion tokens assumed (no run history for this model yet)"
        ),
        credit_remaining=credit,
    )


def split_estimate(parts: list[CostEstimate | None]) -> CostEstimate | None:
    """Combine per-model estimates for a duel, where each model makes half the calls."""
    ests = [e for e in parts if e is not None]
    if len(ests) != len(parts) or not ests:
        return None
    calls = ests[0].calls_max
    per_call = (
        None
        if any(e.usd_per_call is None for e in ests)
        else sum(e.usd_per_call or 0.0 for e in ests) / len(ests)
    )
    worst = (
        None
        if any(e.usd_per_call_worst is None for e in ests)
        else sum(e.usd_per_call_worst or 0.0 for e in ests) / len(ests)
    )
    return CostEstimate(
        model=" + ".join(e.model for e in ests),
        calls_max=calls,
        usd_per_call=per_call,
        usd_per_call_worst=worst,
        basis="half the calls per model — " + "; ".join(f"{e.model}: {e.basis}" for e in ests),
        credit_remaining=ests[0].credit_remaining,
    )


def _deadline(scenario: Scenario, max_rounds: int | None) -> int:
    # Mirrors run_episode: the cap never extends beyond the scenario's own deadline.
    d = scenario.deadline_rounds
    return min(d, max(1, max_rounds)) if max_rounds is not None else d


def calls_ceiling(specs: Iterable[EpisodeSpec], *, agent: str, max_rounds: int | None) -> int:
    """Model calls if every episode runs to its deadline (one per LLM side per round)."""
    per_round = 2 if agent == "a2a" else 1
    return sum(_deadline(s.scenario, max_rounds) * per_round for s in specs)


def history_rate(model: str, results_dir: Path) -> tuple[float, int, tuple[str, ...]] | None:
    """Mean provider-reported $/call for ``model`` across past traces.

    Each trace's ``cost_usd_by_role`` is divided by that role's step count: one
    step is one ``act`` (re-asks included in its cost), and only LLM sides report
    a cost, so a scripted opponent never dilutes the rate.
    """
    total = 0.0
    calls = 0
    files: set[str] = set()
    for path in sorted(results_dir.glob("*.jsonl")):
        try:
            fh = path.open(encoding="utf-8")
        except OSError:
            continue
        with fh:
            for line in fh:
                if model not in line:  # cheap pre-filter before parsing
                    continue
                try:
                    trace = json.loads(line)
                except json.JSONDecodeError:
                    continue
                default_model = (trace.get("llm_config") or {}).get("model")
                # Duel traces run a different model per side.
                role_models = trace.get("models_by_role") or {}
                parties = [s.get("party") for s in trace.get("steps") or []]
                for role, cost in (trace.get("cost_usd_by_role") or {}).items():
                    if role_models.get(role, default_model) != model:
                        continue
                    n = parties.count(role)
                    if n and cost is not None:
                        total += float(cost)
                        calls += n
                        files.add(path.name)
    if calls == 0:
        return None
    return total / calls, calls, tuple(sorted(files))


def mean_prompt_tokens(
    specs: Iterable[EpisodeSpec], settings: AgentSettings, *, max_rounds: int | None
) -> float:
    """Average prompt size per call, measured by rendering the real prompt.

    For each distinct (scenario, role): system prompt + tool schema + payload,
    rendered at the first turn (empty history) and the last (a full history of
    offers from both sides). The history grows linearly, so the mean of the two
    is the per-call average. Messages (A2A, optional) are not assumed.
    """
    seen: dict[tuple[str, Role, int], float] = {}
    roles: tuple[Role, ...] = ("buyer", "seller")
    instruction = "x" * _INSTRUCTION_CHARS
    for spec in specs:
        sc = spec.scenario
        deadline = _deadline(sc, max_rounds)
        for role in roles:
            key = (sc.id, role, deadline)
            if key in seen:
                continue
            issues = list(sc.issues)
            prefs = sc.preferences(role)
            offer = {i.name: i.values[-1] for i in issues}
            fixed = len(settings.system_prompt_for(role)) + len(json.dumps(build_move_tool(issues)))
            first = NegotiationState(
                role=role, prefs=prefs, issues=issues, round=0, deadline=deadline
            )
            full = [
                {"round": r, "by": by, "offer": offer, "message": None}
                for r in range(deadline - 1)
                for by in roles
            ]
            last = NegotiationState(
                role=role,
                prefs=prefs,
                issues=issues,
                round=deadline - 1,
                deadline=deadline,
                history=full,
            )
            start = len(build_user_payload(first, instruction, None))
            end = len(build_user_payload(last, instruction, offer))
            seen[key] = (fixed + (start + end) / 2) / _CHARS_PER_TOKEN
    return sum(seen.values()) / len(seen) if seen else 0.0


def _is_openrouter(settings: AgentSettings) -> bool:
    return (urlparse(settings.base_url).hostname or "").endswith("openrouter.ai")


def _http_json(url: str, headers: dict[str, str]) -> Any:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
        return json.load(resp)


def model_pricing(settings: AgentSettings, fetch: FetchJson) -> tuple[float, float] | None:
    """OpenRouter's published ($/prompt token, $/completion token) for the model."""
    try:
        data = fetch(settings.base_url.rstrip("/") + "/models", {})
    except Exception:
        return None
    for m in (data or {}).get("data", []) if isinstance(data, dict) else []:
        if m.get("id") == settings.model:
            p = m.get("pricing") or {}
            try:
                return float(p["prompt"]), float(p["completion"])
            except (KeyError, TypeError, ValueError):
                return None
    return None


def credit_remaining(settings: AgentSettings, fetch: FetchJson) -> float | None:
    """Remaining credit on the OpenRouter key (None when unlimited or unknown)."""
    if not settings.api_key:
        return None
    try:
        data = fetch(
            settings.base_url.rstrip("/") + "/key",
            {"Authorization": f"Bearer {settings.api_key}"},
        )
    except Exception:
        return None
    rem = (
        ((data or {}).get("data") or {}).get("limit_remaining") if isinstance(data, dict) else None
    )
    try:
        return float(rem) if rem is not None else None
    except (TypeError, ValueError):
        return None
