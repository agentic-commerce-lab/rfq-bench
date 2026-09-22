"""OfflineRunner: a bilateral SAO session on NegMAS, producing an immutable trace.

NegMAS owns the mechanism, the alternating-offers protocol, the deadline, and the
negotiator lifecycle (``negmas_engine``). This module owns the experiment matrix
and turns the recorded episode into our immutable trace. Scoring happens entirely
outside NegMAS, on the returned trace.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from rfq_bench.core.contracts import (
    LLMConfig,
    OutcomeKind,
    Role,
    Scenario,
    Step,
    Trace,
)
from rfq_bench.core.outcome_space import is_legal
from rfq_bench.core.utility import outcome_utility, reservation_utility
from rfq_bench.core.zopa import compute_zopa
from rfq_bench.opponents import get_opponent
from rfq_bench.strategies import get_strategy
from rfq_bench.strategies.base import NegotiationPolicy

_STUB_LLM = LLMConfig(
    base_url="offline://scripted", model="scripted", temperature=0.0, max_tokens=0
)


def _other(role: Role) -> Role:
    return "seller" if role == "buyer" else "buyer"


@dataclass
class EpisodeSpec:
    """One matched cell of the experiment matrix.

    In offline mode ``strategy`` is the tested arm and ``opponent`` the scripted
    environment. In A2A mode the seller strategy is still the tested arm
    (``strategy``, ``target_role='seller'``) and the buyer **persona** is the
    opponent — carried on ``opponent`` (and mirrored on ``persona`` for
    provenance) so the standard report/dashboard treat it like any other opponent.
    """

    scenario: Scenario
    strategy: str
    opponent: str
    target_role: Role
    first_speaker: Role
    seed: int
    mode: str = "offline"
    persona: str | None = None


def run_episode(
    spec: EpisodeSpec,
    *,
    agent_policy: NegotiationPolicy | None = None,
    opponent_policy: NegotiationPolicy | None = None,
    llm_config: LLMConfig | None = None,
    max_rounds: int | None = None,
) -> Trace:
    """Run one SAO episode on a NegMAS mechanism and return an immutable trace.

    ``agent_policy`` defaults to the deterministic strategy policy; an LLM
    negotiator implementing the same protocol can be injected here.
    ``opponent_policy`` defaults to the scripted opponent for ``spec.opponent``;
    inject a second ``LLMNegotiator`` here for A2A self-play (LLM vs LLM). Both
    sides' token and USD spend are then recorded per role on the trace.

    ``max_rounds`` caps the per-episode deadline at ``min(scenario.deadline_rounds,
    max_rounds)`` (``None`` = the scenario's own deadline). The cap drives both the
    mechanism's step budget and the concession schedules, so a shorter deadline
    changes outcomes — use it for faster/cheaper exploratory runs, not for results
    compared against full-deadline runs.
    """
    # Imported here so the heavy NegMAS import is paid only when actually running.
    from negmas.sao import SAOMechanism

    from rfq_bench.runners.negmas_engine import (
        Event,
        PolicyNegotiator,
        build_outcome_space,
        build_ufun,
    )

    scenario = spec.scenario
    target = spec.target_role
    opp_role = _other(target)

    # Cap the deadline if requested; never extend beyond the scenario's own deadline.
    deadline = scenario.deadline_rounds
    if max_rounds is not None:
        deadline = min(deadline, max(1, max_rounds))

    agent_pol = agent_policy if agent_policy is not None else get_strategy(spec.strategy)
    opp_pol = opponent_policy if opponent_policy is not None else get_opponent(spec.opponent)

    outcome_space, _ = build_outcome_space(scenario)
    events: list[Event] = []

    agent_neg = PolicyNegotiator(
        policy=agent_pol,
        role=target,
        scenario=scenario,
        deadline=deadline,
        events=events,
        name=f"agent-{target}",
        ufun=build_ufun(scenario.preferences(target), scenario, outcome_space),
    )
    opp_neg = PolicyNegotiator(
        policy=opp_pol,
        role=opp_role,
        scenario=scenario,
        deadline=deadline,
        events=events,
        name=f"opponent-{opp_role}",
        ufun=build_ufun(scenario.preferences(opp_role), scenario, outcome_space),
    )

    mechanism = SAOMechanism(outcome_space=outcome_space, n_steps=deadline)
    # Add negotiators in first-speaker order so the mechanism starts with them.
    ordered = [agent_neg, opp_neg] if spec.first_speaker == target else [opp_neg, agent_neg]
    for neg in ordered:
        mechanism.add(neg)

    started = time.perf_counter()
    mechanism.run()
    latency_s = time.perf_counter() - started

    issues = list(scenario.issues)
    agreement_tuple = mechanism.state.agreement
    agreement = (
        {i.name: v for i, v in zip(issues, agreement_tuple, strict=True)}
        if agreement_tuple is not None
        else None
    )
    steps = [
        Step(
            round=e.round,
            party=e.party,
            action=e.action,
            outcome=e.outcome,
            rationale=e.rationale,
            adjusted=e.adjusted,
            adjust_reason=e.adjust_reason,
            intended_action=e.intended_action,
            intended_outcome=e.intended_outcome,
            error=e.error,
            error_reason=e.error_reason,
            raw_response=e.raw_response,
            reasoning=e.reasoning,
        )
        for e in events
    ]
    terminated = bool(mechanism.state.broken)
    # Combined episode cost. Offline: the scripted opponent adds 0, so this stays
    # the tested agent's spend. A2A: both LLM sides are summed.
    token_cost = agent_neg.total_tokens + opp_neg.total_tokens
    cost_reported = agent_neg.cost_reported or opp_neg.cost_reported
    cost_usd = (agent_neg.total_cost + opp_neg.total_cost) if cost_reported else None
    token_cost_by_role: dict[Role, int] = {
        target: agent_neg.total_tokens,
        opp_role: opp_neg.total_tokens,
    }
    cost_usd_by_role: dict[Role, float] = {}
    if agent_neg.cost_reported:
        cost_usd_by_role[target] = agent_neg.total_cost
    if opp_neg.cost_reported:
        cost_usd_by_role[opp_role] = opp_neg.total_cost
    # An unusable reply from EITHER side aborts the episode; it is excluded from
    # scoring (events are a shared log, so an opponent error is captured too).
    error_events = [e for e in events if e.error]
    errored = bool(error_events)
    error_reason = error_events[0].error_reason if error_events else None

    return _finalize(
        spec,
        steps,
        agreement,
        latency_s,
        llm_config or _STUB_LLM,
        token_cost,
        terminated,
        cost_usd=cost_usd,
        errored=errored,
        error_reason=error_reason,
        token_cost_by_role=token_cost_by_role,
        cost_usd_by_role=cost_usd_by_role,
    )


def _finalize(
    spec: EpisodeSpec,
    steps: list[Step],
    agreement: dict[str, Any] | None,
    latency_s: float,
    llm_config: LLMConfig,
    token_cost: int = 0,
    terminated: bool = False,
    cost_usd: float | None = None,
    errored: bool = False,
    error_reason: str | None = None,
    token_cost_by_role: dict[Role, int] | None = None,
    cost_usd_by_role: dict[Role, float] | None = None,
) -> Trace:
    scenario = spec.scenario
    zopa = compute_zopa(scenario)

    if agreement is not None:
        utilities = {
            "buyer": outcome_utility(scenario.buyer, scenario.issues, agreement),
            "seller": outcome_utility(scenario.seller, scenario.issues, agreement),
        }
        outcome_kind: OutcomeKind = "agreement"
    else:
        # No deal -> both parties fall back to their BATNA.
        utilities = {
            "buyer": reservation_utility(scenario.buyer),
            "seller": reservation_utility(scenario.seller),
        }
        # Refusing when no ZOPA exists is correct behavior, not a failure.
        outcome_kind = "walk_away" if (terminated or not zopa.exists) else "no_deal"

    target = spec.target_role
    validity = {
        "agreement_reached": agreement is not None,
        "legal_agreement": agreement is None or is_legal(scenario.issues, agreement),
        "no_batna_violation": utilities[target]
        >= reservation_utility(scenario.preferences(target)) - 1e-9,
        "correct_walk_away": (not zopa.exists) and agreement is None,
    }

    return Trace(
        scenario_id=scenario.id,
        outcome_space_version=scenario.outcome_space_version,
        strategy=spec.strategy,
        opponent=spec.opponent,
        target_role=target,
        first_speaker=spec.first_speaker,
        seed=spec.seed,
        llm_config=llm_config,
        steps=steps,
        outcome_kind=outcome_kind,
        agreement=agreement,
        utilities=utilities,
        rounds_to_close=len(steps),
        latency_s=latency_s,
        token_cost=token_cost,
        cost_usd=cost_usd,
        mode="a2a" if spec.mode == "a2a" else "offline",
        persona=spec.persona,
        token_cost_by_role=token_cost_by_role or {},
        cost_usd_by_role=cost_usd_by_role or {},
        errored=errored,
        error_reason=error_reason,
        validity=validity,
    )


def build_matrix(
    scenarios: list[Scenario],
    *,
    strategies: list[str],
    opponents: list[str],
    seeds: list[int],
    roles: tuple[Role, ...] = ("buyer", "seller"),
    first_speakers: tuple[Role, ...] = ("buyer", "seller"),
) -> Iterator[EpisodeSpec]:
    """Expand the fully balanced experiment matrix.

    Every cell is mirrored across role and first-speaker order and repeated per
    seed. ``logrolling`` is skipped on single-issue scenarios where it is not
    defined.
    """
    from rfq_bench.strategies.registry import MULTI_ISSUE_ONLY

    for scenario in scenarios:
        multi = len(scenario.issues) > 1
        for strategy in strategies:
            if strategy in MULTI_ISSUE_ONLY and not multi:
                continue
            for opponent in opponents:
                for role in roles:
                    for first in first_speakers:
                        for seed in seeds:
                            yield EpisodeSpec(
                                scenario=scenario,
                                strategy=strategy,
                                opponent=opponent,
                                target_role=role,
                                first_speaker=first,
                                seed=seed,
                            )


def build_a2a_matrix(
    scenarios: list[Scenario],
    *,
    personas: list[str],
    strategies: list[str],
    seeds: list[int],
    first_speakers: tuple[Role, ...] = ("buyer", "seller"),
) -> Iterator[EpisodeSpec]:
    """Expand the A2A self-play matrix: LLM seller strategy vs LLM buyer persona.

    Mapped onto the standard benchmark axes so the offline report and dashboard
    work unchanged: the **seller strategy is the treatment** (the scored agent,
    ``target_role='seller'``, with a ``control`` arm for Δ) and the **buyer
    persona is the opponent** (the swept environment, carried on ``opponent`` — the
    same axis scripted opponents use). Roles are pinned, so only the first-speaker
    order is swept. ``logrolling`` is skipped on single-issue scenarios.
    """
    from rfq_bench.strategies.registry import MULTI_ISSUE_ONLY

    for scenario in scenarios:
        multi = len(scenario.issues) > 1
        for persona in personas:
            for strategy in strategies:
                if strategy in MULTI_ISSUE_ONLY and not multi:
                    continue
                for first in first_speakers:
                    for seed in seeds:
                        yield EpisodeSpec(
                            scenario=scenario,
                            strategy=strategy,
                            opponent=persona,
                            target_role="seller",
                            first_speaker=first,
                            seed=seed,
                            mode="a2a",
                            persona=persona,
                        )
