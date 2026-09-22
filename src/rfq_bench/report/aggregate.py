"""Build the full report: strategy effects plus separate-track metrics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from rfq_bench.core.contracts import Role, Scenario, Trace
from rfq_bench.core.scoring import EffectResult, ScoredEpisode, score_trace, strategy_effects
from rfq_bench.report.metrics import SideMetrics, side_metrics


@dataclass(frozen=True)
class Report:
    effects: list[EffectResult]
    side: dict[str, SideMetrics]
    n_episodes: int
    n_degenerate: int
    control: str
    control_present: bool = True
    # Column label for the arm (the thing that varies): "strategy" for the seller
    # treatment, "persona" for the A2A buyer view.
    arm_label: str = "strategy"
    # Which side's captured value this table scores ("seller"/"buyer"); shown in
    # the header so a buyer table is never mistaken for the seller one.
    scored_role: str | None = None

    def render(self) -> str:
        lines: list[str] = []
        label = self.arm_label
        ctl = f"{self.control!r}" if self.control_present else f"{self.control!r} (ABSENT)"
        scored = f", scoring the {self.scored_role}" if self.scored_role else ""
        lines.append(
            f"rfq-bench report — {self.n_episodes} episodes "
            f"({self.n_degenerate} degenerate excluded), control = {ctl}{scored}"
        )
        if not self.control_present:
            lines.append(
                f"note: control {self.control!r} is not in these traces — "
                "Δ vs control is undefined (S_s still shown)."
            )
        lines.append("")
        lines.append(
            f"{label.capitalize()} score S_s and effect Δ vs control (95% CI clustered by scenario)"
        )
        header = f"{label:<14}{'S_s':>8}{'Δ':>9}{'95% CI':>20}{'n':>7}"
        lines.append(header)
        lines.append("-" * len(header))
        for e in self.effects:
            is_ctl = self.control_present and e.strategy == self.control
            delta = "—" if is_ctl or e.delta != e.delta else f"{e.delta:+.1f}"
            ci = "—" if is_ctl or e.ci_low != e.ci_low else f"[{e.ci_low:+.1f}, {e.ci_high:+.1f}]"
            lines.append(f"{e.strategy:<14}{e.score:>8.1f}{delta:>9}{ci:>20}{e.n_episodes:>7}")
        lines.append("")
        lines.append("Separate tracks (not part of the score)")
        show_cost = any(m.mean_cost_usd is not None for m in self.side.values())
        h2 = f"{self.arm_label:<14}{'agree%':>8}{'joint':>8}{'pareto%':>9}{'rounds':>8}{'walk✓':>8}"
        if show_cost:
            h2 += f"{'$/ep':>11}"
        lines.append(h2)
        lines.append("-" * len(h2))
        for name in [e.strategy for e in self.effects]:
            m = self.side[name]
            nan_walk = m.walk_away_accuracy != m.walk_away_accuracy
            walk = "—" if nan_walk else f"{m.walk_away_accuracy * 100:.0f}"
            row = (
                f"{name:<14}{m.agreement_rate * 100:>7.0f}%{m.mean_joint_surplus:>8.2f}"
                f"{m.pareto_rate * 100:>8.0f}%{m.mean_rounds_to_close:>8.1f}{walk:>8}"
            )
            if show_cost:
                cost = "—" if m.mean_cost_usd is None else f"${m.mean_cost_usd:.5f}"
                row += f"{cost:>11}"
            lines.append(row)
        if show_cost:
            lines.append("($/ep = mean real spend per episode, from the provider's usage.cost)")
        return "\n".join(lines)


def build_report(
    traces: list[Trace],
    scenarios: dict[str, Scenario],
    *,
    control: str = "control",
    n_boot: int = 2000,
    seed: int = 0,
    role: Role | None = None,
    group_key: str = "strategy",
    arm_label: str = "strategy",
    require_control: bool = False,
) -> Report:
    """Score ``traces`` and aggregate into a :class:`Report`.

    Defaults reproduce the offline report: score the target agent, group by
    ``strategy``. For the A2A buyer view pass ``role="buyer"``,
    ``group_key="opponent"`` (the persona axis), ``control="neutral"``, and
    ``arm_label="persona"`` — the buyer's captured value per persona, from the
    same traces.
    """
    episodes: list[ScoredEpisode] = [
        score_trace(t, scenarios[t.scenario_id], role=role) for t in traces
    ]
    arm_of: Callable[[ScoredEpisode], str] = (
        (lambda e: e.opponent) if group_key == "opponent" else (lambda e: e.strategy)
    )
    trace_arm: Callable[[Trace], str] = (
        (lambda t: t.opponent) if group_key == "opponent" else (lambda t: t.strategy)
    )
    control_present = any(arm_of(ep) == control for ep in episodes)
    # Degrade gracefully for exploratory subset runs that omit the control arm:
    # report S_s, leave Δ undefined, rather than crash.
    effects = strategy_effects(
        episodes,
        control=control,
        n_boot=n_boot,
        seed=seed,
        require_control=require_control,
        key=arm_of,
    )

    by_arm: dict[str, list[Trace]] = defaultdict(list)
    for t in traces:
        by_arm[trace_arm(t)].append(t)
    side = {name: side_metrics(name, ts, scenarios) for name, ts in by_arm.items()}

    return Report(
        effects=effects,
        side=side,
        n_episodes=sum(1 for e in episodes if e.scorable),
        n_degenerate=sum(1 for e in episodes if e.degenerate),
        control=control,
        control_present=control_present,
        arm_label=arm_label,
        scored_role=role,
    )
