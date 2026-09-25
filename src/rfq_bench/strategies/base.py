"""Shared negotiation-policy machinery for strategies and opponents.

A policy maps the current negotiation state to a concrete offer and an
accept/reject decision. Behavior is driven by a normalized *target utility*
schedule over the party's own [reservation, ideal] range, following the
time-dependent concession functions of Faratin, Sierra & Jennings (1998):

    target(t) = reservation + f(t) * (ideal - reservation)
    f(t)      = opening * (1 - (t / T) ** (1 / beta))

- beta < 1  -> Boulware  (hold near the opening, concede late)
- beta = 1  -> Linear    (steady concession)
- beta > 1  -> Conceder  (concede early)

Both the fixed agent's strategy and the deterministic opponents use this same
machinery, which keeps the two sides symmetric and fully reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from rfq_bench.core.contracts import Issue, PartyPreferences, Role
from rfq_bench.core.outcome_space import enumerate_outcomes
from rfq_bench.core.utility import ideal_utility, outcome_utility, reservation_utility

_EPS = 1e-9


@dataclass
class NegotiationState:
    """Everything a policy may look at. Only the party's own preferences are known."""

    role: Role
    prefs: PartyPreferences
    issues: list[Issue]
    round: int
    deadline: int
    # Outcomes the opponent has proposed to us, oldest first.
    opponent_offers: list[dict[str, Any]] = field(default_factory=list)
    # Outcomes we have proposed, oldest first.
    my_offers: list[dict[str, Any]] = field(default_factory=list)
    # Both parties' moves so far in the order they happened, as
    # {"round", "by" (role), "offer", "message"}; the LLM payload renders this
    # append-only transcript (cache-friendly). Scripted policies ignore it.
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def progress(self) -> float:
        """Fraction of the deadline elapsed, in [0, 1]."""
        if self.deadline <= 1:
            return 1.0
        return min(1.0, self.round / (self.deadline - 1))

    @property
    def is_last_round(self) -> bool:
        return self.round >= self.deadline - 1


@dataclass
class Move:
    """A single decision: accept the standing offer, counter, or walk away."""

    action: Literal["offer", "accept", "terminate"]
    outcome: dict[str, Any] | None = None
    token_cost: int = 0
    # Real spend in USD for this turn, when the provider reports it; None otherwise.
    cost_usd: float | None = None
    # The agent's own one-line explanation (LLM agent only); None for scripted policies.
    rationale: str | None = None
    # The played move was adjusted from the model's raw output by a benign, faithful
    # rule (snap an off-grid value to the nearest legal tier, or clamp a below-floor
    # quote up to the shop's reservation floor). NOT a scripted-strategy substitution
    # and NOT an error — the episode is still scored.
    adjusted: bool = False
    adjust_reason: str | None = None
    # Which policy constraint fired, for counting: "snap" (off-grid value moved to
    # the nearest legal option), "floor_offer" (offer below the walk-away raised to
    # it), "floor_accept" (below-walk-away accept turned into a floor counter).
    adjust_kind: str | None = None
    # What the model literally tried, recorded whenever the move was adjusted.
    intended_action: str | None = None
    intended_outcome: dict[str, Any] | None = None
    # The model produced no usable move (unparseable / missing issue / non-snappable
    # value). The episode is aborted and excluded from scoring; the error is logged.
    error: bool = False
    error_reason: str | None = None
    # The model's raw reply (verbatim JSON) and its chain-of-thought, when the LLM
    # agent produced them; None for scripted policies.
    raw_response: str | None = None
    reasoning: str | None = None
    # The upstream provider that served this model call (OpenRouter's `provider`).
    provider: str | None = None
    # Number of corrective re-asks the LLM agent needed this turn to produce a
    # usable move (0 = got it on the first call). Recorded for audit and to detect
    # a model that only complies under pressure; scripted policies leave it 0.
    reask_count: int = 0
    # A2A message channel: text the other party reads alongside this move (None =
    # no message). Stored in full; delivery truncates to MESSAGE_MAX_CHARS and
    # ``message_truncated`` records that it happened.
    message: str | None = None
    message_truncated: bool = False
    # A policy constraint fired on this move, so its message was NOT delivered: it
    # would quote the invalid terms the model tried, not the corrected ones played.
    # The message is still kept here for audit.
    message_withheld: bool = False
    # Prompt tokens sent this turn and how many the provider served from its
    # prompt cache (summed across re-asks/retries); 0 when not reported.
    prompt_tokens: int = 0
    cached_tokens: int = 0


@runtime_checkable
class NegotiationPolicy(Protocol):
    name: str

    def propose(self, state: NegotiationState) -> dict[str, Any]:
        """Return the outcome to offer this round."""

    def accepts(self, opponent_outcome: dict[str, Any], state: NegotiationState) -> bool:
        """Return True to accept the opponent's current outcome."""


def decide(
    policy: NegotiationPolicy,
    standing_offer: dict[str, Any] | None,
    state: NegotiationState,
) -> Move:
    """Resolve one turn.

    A policy may implement ``act`` to make a unified decision (used by the LLM
    negotiator). Otherwise the scripted accept-then-counter path is used.
    """
    act = getattr(policy, "act", None)
    if callable(act):
        move: Move = act(standing_offer, state)
        return move
    if standing_offer is not None and policy.accepts(standing_offer, state):
        return Move("accept", standing_offer)
    return Move("offer", policy.propose(state))


def select_outcome(
    prefs: PartyPreferences,
    issues: list[Issue],
    target: float,
    *,
    integrative: bool = False,
    opponent_last: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pick the most generous own-offer that still meets ``target``.

    Among outcomes whose own utility is >= target we offer the *least* good for
    ourselves (most acceptable to the opponent). For the integrative policy we
    break ties toward the opponent's last revealed offer — conceding on issues
    we value little, i.e. logrolling without seeing the opponent's private
    utility. If nothing meets the target, we offer our best remaining outcome.
    """
    outcomes = enumerate_outcomes(issues)
    feasible = [o for o in outcomes if outcome_utility(prefs, issues, o) >= target - _EPS]
    if not feasible:
        return max(outcomes, key=lambda o: outcome_utility(prefs, issues, o))

    def sort_key(o: dict[str, Any]) -> tuple[float, ...]:
        own = outcome_utility(prefs, issues, o)
        if integrative and opponent_last is not None:
            matches = sum(1 for k, v in opponent_last.items() if o.get(k) == v)
            return (own, -matches)
        return (own,)

    return min(feasible, key=sort_key)


@dataclass
class ConcessionPolicy:
    """Time-dependent concession policy parameterized by opening and beta."""

    name: str
    opening: float = 0.9
    beta: float = 1.0
    integrative: bool = False

    def target_utility(self, state: NegotiationState) -> float:
        d = reservation_utility(state.prefs)
        ideal = ideal_utility(state.prefs, state.issues)
        t = state.progress
        f = self.opening * (1.0 - t ** (1.0 / self.beta))
        return float(d + max(0.0, f) * (ideal - d))

    def propose(self, state: NegotiationState) -> dict[str, Any]:
        target = self.target_utility(state)
        opponent_last = state.opponent_offers[-1] if state.opponent_offers else None
        return select_outcome(
            state.prefs,
            state.issues,
            target,
            integrative=self.integrative,
            opponent_last=opponent_last,
        )

    def accepts(self, opponent_outcome: dict[str, Any], state: NegotiationState) -> bool:
        own = outcome_utility(state.prefs, state.issues, opponent_outcome)
        if own < reservation_utility(state.prefs) - _EPS:
            return False  # never accept below BATNA
        return own >= self.target_utility(state) - _EPS


@dataclass
class TitForTatPolicy:
    """Open cooperatively, then reciprocate the opponent's concession magnitude.

    Concession is measured on our own utility scale: as the opponent's offers
    become better for us, we concede by the same amount (Baarslag et al. 2013).
    """

    name: str = "tit_for_tat"
    opening: float = 0.85
    integrative: bool = False
    # Per-episode memo of the target we actually used, keyed by round. Reciprocation
    # must build on our *previous target*, not on our last offer's own-utility (which
    # is >= the target on a coarse grid and would understate concession). Memoizing
    # also makes ``_target`` idempotent across the multiple calls within one turn.
    _targets: dict[int, float] = field(default_factory=dict)

    def _target(self, state: NegotiationState) -> float:
        cached = self._targets.get(state.round)
        if cached is not None:
            return cached
        d = reservation_utility(state.prefs)
        ideal = ideal_utility(state.prefs, state.issues)
        base = d + self.opening * (ideal - d)
        offers = state.opponent_offers
        if len(offers) < 2 or not self._targets:
            target = base
        else:
            prev_target = self._targets[max(self._targets)]  # our last round's target
            gain = outcome_utility(state.prefs, state.issues, offers[-1]) - outcome_utility(
                state.prefs, state.issues, offers[-2]
            )
            # Reciprocate: concede as much as the opponent just conceded to us.
            conceded_so_far = base - prev_target
            target = max(d, min(ideal, base - conceded_so_far - max(0.0, gain)))
        self._targets[state.round] = target
        return target

    def target_utility(self, state: NegotiationState) -> float:
        return self._target(state)

    def propose(self, state: NegotiationState) -> dict[str, Any]:
        target = self._target(state)
        opponent_last = state.opponent_offers[-1] if state.opponent_offers else None
        return select_outcome(
            state.prefs,
            state.issues,
            target,
            integrative=self.integrative,
            opponent_last=opponent_last,
        )

    def accepts(self, opponent_outcome: dict[str, Any], state: NegotiationState) -> bool:
        own = outcome_utility(state.prefs, state.issues, opponent_outcome)
        if own < reservation_utility(state.prefs) - _EPS:
            return False
        return own >= self._target(state) - _EPS
