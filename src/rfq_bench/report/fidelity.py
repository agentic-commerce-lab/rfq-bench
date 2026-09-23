"""Strategy fidelity: did the tested agent actually play the strategy it was given?

A manipulation check, computed on the persisted trace like every other metric and
never folded into the strategy score. Each arm has an operational definition in
the strategy registry (opening, concession shape ``beta``, acceptance rule). We
replay that reference policy over the *actual* episode history and compare, turn
by turn, what the tested agent did with what the strategy prescribes:

- **offer gap** — the agent's offer vs the reference offer, as own-utility on the
  agent's [BATNA, ideal] range: ``(u_agent - u_ref) / (I - d)``. Positive = held
  out harder than the strategy, negative = conceded more. The reference offer is
  what the scripted policy would propose (it already absorbs grid quantization).
- **opening gap** — the offer gap of the agent's first offer (isolates Anchoring).
- **accept rule** — at every turn facing an opponent offer: an *early accept*
  (accepted though the strategy would not) or a *missed accept* (countered or
  walked though the strategy would have accepted).
- **fitted beta** — the concession shape fitted to the agent's own offers, next to
  the same fit on the reference offers (apples to apples under quantization).

The reference is conditional on what really happened (Tit-for-Tat reciprocates
the opponent's actual moves), so it answers "given this history, what would the
strategy have done?". Scripted-agent traces therefore replay with zero gap — the
tests use that as the correctness anchor. Strategies without a registry
definition (file-defined guidance under a new name) have no reference and are
skipped.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from rfq_bench.core.contracts import Scenario, Trace
from rfq_bench.core.utility import ideal_utility, outcome_utility, reservation_utility
from rfq_bench.strategies import STRATEGIES, get_strategy
from rfq_bench.strategies.base import NegotiationState

# An offer counts as "on strategy" when within this share of the [d, I] range.
DEFAULT_TOLERANCE = 0.10
# Log-spaced beta grid for the concession-shape fit (0.01 .. 100).
_BETA_GRID = tuple(10 ** (k / 50) for k in range(-100, 101))
_EPS = 1e-9


@dataclass(frozen=True)
class OfferCheck:
    round: int
    progress: float
    agent_utility: float
    reference_utility: float
    gap: float  # (agent - reference) / (I - d)
    # The offer the strategy would have made this turn (for replay overlays).
    reference_outcome: dict[str, Any]
    # Both utilities normalized onto the agent's range: (u - d) / (I - d).
    agent_level: float
    reference_level: float


@dataclass(frozen=True)
class EpisodeFidelity:
    scenario_id: str
    strategy: str
    opponent: str
    offers: list[OfferCheck]
    early_accepts: int
    missed_accepts: int
    accept_decisions: int  # turns where the agent faced an opponent offer
    walked: bool
    fitted_beta: float | None
    reference_beta: float | None
    deadline: int

    @property
    def opening_gap(self) -> float | None:
        return self.offers[0].gap if self.offers else None

    @property
    def mean_gap(self) -> float | None:
        return statistics.fmean(o.gap for o in self.offers) if self.offers else None

    @property
    def mean_abs_gap(self) -> float | None:
        return statistics.fmean(abs(o.gap) for o in self.offers) if self.offers else None


def _effective_deadline(trace: Trace, scenario: Scenario) -> int:
    """The deadline the episode actually ran under.

    Recorded on newer traces. Older traces lack it: a timed-out no-deal reveals it
    exactly (last round + 1); otherwise fall back to the scenario's own deadline,
    which is only wrong for traces run with ``max_rounds`` that closed early.
    """
    if trace.deadline is not None:
        return trace.deadline
    ran_out = trace.outcome_kind == "no_deal" and not any(
        s.action == "terminate" for s in trace.steps
    )
    if ran_out and trace.steps:
        return max(s.round for s in trace.steps) + 1
    return scenario.deadline_rounds


def _fit_beta(points: list[tuple[float, float]]) -> float | None:
    """Least-squares beta for ``f(t) = f0 * (1 - t ** (1 / beta))``.

    ``points`` are (progress, normalized own-utility above BATNA); ``f0`` is the
    first point's level. Needs >= 3 offers, some time elapsed, and some room to
    concede, otherwise the shape is not identified and None is returned.
    """
    if len(points) < 3:
        return None
    f0 = points[0][1]
    later = [(t, f) for t, f in points[1:] if t > _EPS]
    if f0 <= _EPS or len(later) < 2:
        return None

    def sse(beta: float) -> float:
        return float(sum((f - f0 * (1.0 - t ** (1.0 / beta))) ** 2 for t, f in later))

    return float(min(_BETA_GRID, key=sse))


def episode_fidelity(trace: Trace, scenario: Scenario) -> EpisodeFidelity | None:
    """Replay the reference strategy over one trace; None if not checkable.

    Not checkable: an unknown strategy (no reference definition), an errored
    episode (excluded from scoring too), or a degenerate range (``I == d``).
    """
    if trace.strategy not in STRATEGIES or trace.errored:
        return None
    role = trace.target_role
    prefs = scenario.preferences(role)
    issues = list(scenario.issues)
    d = reservation_utility(prefs)
    span = ideal_utility(prefs, issues) - d
    if span <= _EPS:
        return None

    deadline = _effective_deadline(trace, scenario)
    policy = get_strategy(trace.strategy)  # fresh instance: per-episode state
    offers: list[OfferCheck] = []
    agent_pts: list[tuple[float, float]] = []
    ref_pts: list[tuple[float, float]] = []
    early = missed = decisions = 0
    walked = False
    mine: list[dict[str, Any]] = []
    theirs: list[dict[str, Any]] = []
    standing: dict[str, Any] | None = None

    for step in trace.steps:
        if step.party != role:
            if step.action == "offer" and step.outcome is not None:
                standing = step.outcome
                if not theirs or theirs[-1] != standing:
                    theirs.append(standing)
            continue
        state = NegotiationState(
            role=role,
            prefs=prefs,
            issues=issues,
            round=step.round,
            deadline=deadline,
            opponent_offers=list(theirs),
            my_offers=list(mine),
        )
        # Same call order as the live scripted turn (accepts, then propose), which
        # keeps stateful references (Tit-for-Tat's target memo) faithful.
        if standing is not None:
            decisions += 1
            ref_accepts = policy.accepts(standing, state)
            if step.action == "accept" and not ref_accepts:
                early += 1
            elif step.action != "accept" and ref_accepts:
                missed += 1
        if step.action == "terminate":
            walked = True
        if step.action != "offer" or step.outcome is None:
            continue
        ref_offer = policy.propose(state)
        ref_u = outcome_utility(prefs, issues, ref_offer)
        own_u = outcome_utility(prefs, issues, step.outcome)
        check = OfferCheck(
            round=step.round,
            progress=state.progress,
            agent_utility=own_u,
            reference_utility=ref_u,
            gap=(own_u - ref_u) / span,
            reference_outcome=ref_offer,
            agent_level=(own_u - d) / span,
            reference_level=(ref_u - d) / span,
        )
        offers.append(check)
        agent_pts.append((check.progress, check.agent_level))
        ref_pts.append((check.progress, check.reference_level))
        mine.append(step.outcome)
        standing = None  # our counter replaces the standing offer

    return EpisodeFidelity(
        scenario_id=trace.scenario_id,
        strategy=trace.strategy,
        opponent=trace.opponent,
        offers=offers,
        early_accepts=early,
        missed_accepts=missed,
        accept_decisions=decisions,
        walked=walked,
        fitted_beta=_fit_beta(agent_pts),
        reference_beta=_fit_beta(ref_pts),
        deadline=deadline,
    )


@dataclass(frozen=True)
class ArmFidelity:
    strategy: str
    n_episodes: int
    n_offers: int
    mean_opening_gap: float | None
    mean_gap: float | None  # signed bias; + = tougher than the strategy
    mean_abs_gap: float | None
    within_tolerance: float | None  # share of offers with |gap| <= tolerance
    early_accept_rate: float | None  # per accept decision
    missed_accept_rate: float | None
    median_fitted_beta: float | None
    median_reference_beta: float | None
    n_beta_fits: int


def _mean(xs: list[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def _median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def arm_fidelity(
    strategy: str, episodes: list[EpisodeFidelity], *, tolerance: float = DEFAULT_TOLERANCE
) -> ArmFidelity:
    gaps = [o.gap for e in episodes for o in e.offers]
    decisions = sum(e.accept_decisions for e in episodes)
    fits = [e for e in episodes if e.fitted_beta is not None]
    return ArmFidelity(
        strategy=strategy,
        n_episodes=len(episodes),
        n_offers=len(gaps),
        mean_opening_gap=_mean([g for e in episodes if (g := e.opening_gap) is not None]),
        mean_gap=_mean(gaps),
        mean_abs_gap=_mean([abs(g) for g in gaps]),
        within_tolerance=(sum(abs(g) <= tolerance + _EPS for g in gaps) / len(gaps))
        if gaps
        else None,
        early_accept_rate=(sum(e.early_accepts for e in episodes) / decisions)
        if decisions
        else None,
        missed_accept_rate=(sum(e.missed_accepts for e in episodes) / decisions)
        if decisions
        else None,
        median_fitted_beta=_median([e.fitted_beta for e in fits if e.fitted_beta is not None]),
        median_reference_beta=_median(
            [e.reference_beta for e in fits if e.reference_beta is not None]
        ),
        n_beta_fits=len(fits),
    )


@dataclass(frozen=True)
class FidelityReport:
    arms: list[ArmFidelity]
    n_checked: int
    n_skipped: int
    skipped_strategies: list[str]
    tolerance: float

    def render(self) -> str:
        lines = [
            "Strategy fidelity (separate track — does the agent play its strategy?)",
        ]
        if not self.arms:
            lines.append("no checkable episodes (no known strategy with offers in these traces)")
            return "\n".join(lines)
        w = max(14, *(len(a.strategy) + 2 for a in self.arms))
        header = (
            f"{'strategy':<{w}}{'open':>8}{'bias':>8}{'|gap|':>8}{'on-tgt':>8}"
            f"{'early✗':>8}{'missed✗':>9}{'β fit':>8}{'β ref':>8}{'n':>6}"
        )
        lines.append(header)
        lines.append("-" * len(header))

        def pct(x: float | None) -> str:
            return "—" if x is None else f"{x * 100:.0f}%"

        def num(x: float | None, spec: str = "+.2f") -> str:
            return "—" if x is None or math.isnan(x) else format(x, spec)

        for a in self.arms:
            lines.append(
                f"{a.strategy:<{w}}{num(a.mean_opening_gap):>8}{num(a.mean_gap):>8}"
                f"{num(a.mean_abs_gap, '.2f'):>8}{pct(a.within_tolerance):>8}"
                f"{pct(a.early_accept_rate):>8}{pct(a.missed_accept_rate):>9}"
                f"{num(a.median_fitted_beta, '.2g'):>8}{num(a.median_reference_beta, '.2g'):>8}"
                f"{a.n_episodes:>6}"
            )
        lines.append(
            "(gaps = agent offer − strategy's offer, as a share of the agent's BATNA→ideal "
            "range; + = tougher, − = conceded more · open = first offer · "
            f"on-tgt = offers within ±{self.tolerance:.2f} · early✗/missed✗ = accepts the "
            "strategy would not make / accepts it would have made, per decision · "
            "β = median fitted concession shape (<1 holds, >1 concedes early), "
            "agent vs strategy reference)"
        )
        if self.n_skipped:
            extra = f" (unknown strategy: {', '.join(self.skipped_strategies)})"
            lines.append(
                f"{self.n_skipped} episodes not checked (errored, degenerate, or no "
                f"reference){extra if self.skipped_strategies else ''}"
            )
        return "\n".join(lines)


def build_fidelity_report(
    traces: list[Trace],
    scenarios: dict[str, Scenario],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> FidelityReport:
    by_arm: dict[str, list[EpisodeFidelity]] = defaultdict(list)
    skipped = 0
    for t in traces:
        ep = episode_fidelity(t, scenarios[t.scenario_id])
        if ep is None:
            skipped += 1
        else:
            by_arm[ep.strategy].append(ep)
    order = [s for s in STRATEGIES if s in by_arm]
    return FidelityReport(
        arms=[arm_fidelity(s, by_arm[s], tolerance=tolerance) for s in order],
        n_checked=sum(len(v) for v in by_arm.values()),
        n_skipped=skipped,
        skipped_strategies=sorted({t.strategy for t in traces if t.strategy not in STRATEGIES}),
        tolerance=tolerance,
    )
