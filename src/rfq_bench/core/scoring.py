"""Deterministic scoring: episode score q, strategy score S_s, effect delta.

Everything here is a pure function of immutable traces plus the scenarios that
define private values. No NegMAS, no network. Given the same inputs it returns
the same numbers, which is what makes the benchmark auditable.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from rfq_bench.core.contracts import Role, Scenario, Trace
from rfq_bench.core.utility import ideal_utility, reservation_utility


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def episode_score(u: float, batna: float, ideal: float) -> float | None:
    """Normalized episode score q = clip((U - d) / (I - d), 0, 1).

    Returns ``None`` when the scenario is degenerate (I == d): the score is
    undefined and the episode must be excluded, not treated as zero.
    """
    if ideal == batna:
        return None
    return clip((u - batna) / (ideal - batna), 0.0, 1.0)


@dataclass(frozen=True)
class ScoredEpisode:
    scenario_id: str
    strategy: str
    opponent: str
    role: str
    first_speaker: str
    q: float | None
    degenerate: bool
    errored: bool = False

    @property
    def scorable(self) -> bool:
        return self.q is not None and not self.degenerate and not self.errored


def score_trace(trace: Trace, scenario: Scenario, *, role: Role | None = None) -> ScoredEpisode:
    """Score one immutable trace against its scenario's private values.

    By default the scored side is ``trace.target_role`` (the tested agent). Pass
    ``role`` to score the other party instead — e.g. the buyer in A2A self-play,
    where the buyer plays the (opponent-axis) persona but its captured value is
    still worth reporting. Both parties' realized utilities are on every trace, so
    this needs no rerun.

    An episode the agent aborted with an unusable reply (``trace.errored``) is
    excluded from scoring: it is a failed measurement, not a negotiated outcome.
    """
    scored_role = role or trace.target_role
    if trace.errored:
        return ScoredEpisode(
            scenario_id=trace.scenario_id,
            strategy=trace.strategy,
            opponent=trace.opponent,
            role=scored_role,
            first_speaker=trace.first_speaker,
            q=None,
            degenerate=False,
            errored=True,
        )
    prefs = scenario.preferences(scored_role)
    u = trace.utilities[scored_role]
    d = reservation_utility(prefs)
    ideal = ideal_utility(prefs, scenario.issues)
    q = episode_score(u, d, ideal)
    return ScoredEpisode(
        scenario_id=trace.scenario_id,
        strategy=trace.strategy,
        opponent=trace.opponent,
        role=scored_role,
        first_speaker=trace.first_speaker,
        q=q,
        degenerate=(q is None),
    )


def strategy_score(episodes: Iterable[ScoredEpisode]) -> float:
    """S_s = 100 * macro-average of q with **equal weight per cell**.

    A cell is (scenario, opponent, role, first_speaker); repetitions (seeds) are
    averaged within the cell, then cells are averaged equally. This honours the
    "every cell equally weighted" contract, so an unbalanced scenario mix (uneven
    degenerate/error exclusions, or logrolling running only on multi-issue) does not
    let busier scenarios dominate the score — unlike a flat per-episode mean.
    """
    cells: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for ep in episodes:
        if ep.scorable:
            assert ep.q is not None
            cells[(ep.scenario_id, ep.opponent, ep.role, ep.first_speaker)].append(ep.q)
    if not cells:
        return float("nan")
    cell_means = [float(np.mean(qs)) for qs in cells.values()]
    return 100.0 * float(np.mean(cell_means))


def _by_scenario(episodes: Iterable[ScoredEpisode]) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for ep in episodes:
        if ep.scorable:
            assert ep.q is not None
            grouped[ep.scenario_id].append(ep.q)
    return grouped


@dataclass(frozen=True)
class EffectResult:
    strategy: str
    score: float  # S_s
    control_score: float  # S_control
    delta: float  # S_s - S_control
    ci_low: float
    ci_high: float
    n_episodes: int
    n_degenerate: int


def _cluster_bootstrap_delta(
    arm_by_scenario: dict[str, list[float]],
    control_by_scenario: dict[str, list[float]],
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """95% CI for the delta of means, resampling scenarios (clusters) w/ replacement."""
    scenarios = sorted(set(arm_by_scenario) & set(control_by_scenario))
    if not scenarios:
        return (float("nan"), float("nan"))
    idx = np.arange(len(scenarios))
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.choice(idx, size=len(idx), replace=True)
        arm_vals: list[float] = []
        ctl_vals: list[float] = []
        for j in picked:
            s = scenarios[j]
            arm_vals.extend(arm_by_scenario[s])
            ctl_vals.extend(control_by_scenario[s])
        deltas[b] = 100.0 * (float(np.mean(arm_vals)) - float(np.mean(ctl_vals)))
    return (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))


def strategy_effects(
    episodes: Sequence[ScoredEpisode],
    *,
    control: str = "control",
    n_boot: int = 2000,
    seed: int = 0,
    require_control: bool = True,
    key: Callable[[ScoredEpisode], str] | None = None,
) -> list[EffectResult]:
    """Compute S_s and delta-vs-control with scenario-clustered 95% CIs.

    The bootstrap is seeded so the reported intervals are reproducible. When the
    ``control`` condition is absent from ``episodes`` and ``require_control`` is
    True (the default, guarding the full benchmark), this raises. Pass
    ``require_control=False`` — as the report/dashboard do for exploratory subset
    runs — to instead report each strategy's S_s with an **undefined** effect
    (``delta``/CI = NaN), since there is nothing to compare against.

    ``key`` chooses the arm each episode belongs to (default: its ``strategy``).
    Pass ``lambda e: e.opponent`` to score the opponent axis instead — used for the
    A2A buyer-persona view, where the persona is the arm and ``control`` is a
    baseline persona (e.g. ``neutral``).
    """
    arm_of = key if key is not None else (lambda e: e.strategy)
    rng = np.random.default_rng(seed)
    by_strategy: dict[str, list[ScoredEpisode]] = defaultdict(list)
    degen_by_strategy: dict[str, int] = defaultdict(int)
    for ep in episodes:
        arm = arm_of(ep)
        by_strategy[arm].append(ep)
        if ep.degenerate:
            degen_by_strategy[arm] += 1

    has_control = control in by_strategy
    if not has_control and require_control:
        raise ValueError(f"control condition {control!r} not present in episodes")

    control_by_scenario = _by_scenario(by_strategy[control]) if has_control else {}
    nan = float("nan")

    results: list[EffectResult] = []
    for strategy, eps in by_strategy.items():
        s_score = strategy_score(eps)
        n_scorable = sum(1 for e in eps if e.scorable)
        n_degen = degen_by_strategy[strategy]
        if not has_control:
            # No control in this run: score is defined, effect is not.
            results.append(EffectResult(strategy, s_score, nan, nan, nan, nan, n_scorable, n_degen))
            continue
        arm_by_scenario = _by_scenario(eps)
        # Match control to the arm's own scenario set. An arm that only runs on a
        # subset (e.g. logrolling on multi-issue) must be compared against control
        # on the SAME scenarios, so the point delta and the CI share one basis.
        matched = sorted(set(arm_by_scenario) & set(control_by_scenario))
        matched_control = [q for s in matched for q in control_by_scenario[s]]
        control_score = 100.0 * float(np.mean(matched_control)) if matched_control else nan
        matched_arm = [q for s in matched for q in arm_by_scenario[s]]
        arm_score_matched = 100.0 * float(np.mean(matched_arm)) if matched_arm else nan
        if strategy == control:
            ci_low = ci_high = 0.0
            delta = 0.0
        else:
            ci_low, ci_high = _cluster_bootstrap_delta(
                arm_by_scenario, control_by_scenario, n_boot=n_boot, rng=rng
            )
            delta = arm_score_matched - control_score
        results.append(
            EffectResult(
                strategy=strategy,
                score=s_score,
                control_score=control_score,
                delta=delta,
                ci_low=ci_low,
                ci_high=ci_high,
                n_episodes=n_scorable,
                n_degenerate=n_degen,
            )
        )
    if has_control:
        results.sort(key=lambda r: (r.strategy != control, -r.delta))
    else:
        # No deltas to rank by; fall back to S_s descending.
        results.sort(key=lambda r: -r.score)
    return results
