"""NegMAS bindings: outcome space, utility functions, and the policy negotiator.

NegMAS supplies the negotiation mechanism (SAOMechanism), the alternating-offers
protocol, deadlines, and the negotiator lifecycle. This module maps our
scenario/utility/strategy contracts onto NegMAS objects:

- issues and outcome space  <- Scenario.issues (values ordered => outcome tuples)
- utility functions          <- weighted-additive PartyPreferences, exact
- a strategy/opponent policy  -> one SAONegotiator via the unified ``decide``

Scoring still happens entirely outside NegMAS, on the immutable trace we build
from the recorded events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from negmas import ResponseType
from negmas.outcomes import Issue as NegIssue
from negmas.outcomes import make_issue, make_os
from negmas.preferences import MappingUtilityFunction
from negmas.sao import SAONegotiator, SAOResponse, SAOState

from rfq_bench.core.contracts import Issue, PartyPreferences, Role, Scenario
from rfq_bench.core.outcome_space import enumerate_outcomes
from rfq_bench.core.utility import outcome_utility, reservation_utility
from rfq_bench.strategies.base import Move, NegotiationPolicy, NegotiationState, decide


@dataclass(frozen=True)
class Event:
    """One recorded action, kept in a shared log across both negotiators."""

    round: int
    party: Role
    action: str
    outcome: dict[str, Any] | None
    rationale: str | None = None
    # The played move was adjusted (snapped to grid / clamped to the shop floor).
    adjusted: bool = False
    adjust_reason: str | None = None
    adjust_kind: str | None = None
    intended_action: str | None = None
    intended_outcome: dict[str, Any] | None = None
    # The move was unusable and aborted the episode.
    error: bool = False
    error_reason: str | None = None
    raw_response: str | None = None
    reasoning: str | None = None
    # Corrective re-asks the LLM agent needed to produce this move (0 = first try).
    reask_count: int = 0
    # A2A message channel: the public message sent with this move (full text).
    message: str | None = None
    message_truncated: bool = False
    message_withheld: bool = False


def _event(
    round_: int, party: Role, action: str, outcome: dict[str, Any] | None, move: Move
) -> Event:
    return Event(
        round=round_,
        party=party,
        action=action,
        outcome=outcome,
        rationale=move.rationale,
        adjusted=move.adjusted,
        adjust_reason=move.adjust_reason,
        adjust_kind=move.adjust_kind,
        intended_action=move.intended_action,
        intended_outcome=move.intended_outcome,
        error=move.error,
        error_reason=move.error_reason,
        raw_response=move.raw_response,
        reasoning=move.reasoning,
        reask_count=move.reask_count,
        message=move.message,
        message_truncated=move.message_truncated,
        message_withheld=move.message_withheld,
    )


def _history(events: list[Event]) -> list[dict[str, Any]]:
    """Both parties' offers so far, oldest first, as the LLM payload's transcript.

    Only offers can precede a turn (an accept, walk-away, or error ends the
    episode). Messages are carried in full; the payload truncates on delivery. A
    withheld message (a policy constraint fired on that move) is never delivered.
    ``policy`` records a correction so the payload can tell the *author* (only) what
    it sent and why the played offer differs.
    """
    return [
        {
            "round": e.round,
            "by": e.party,
            "offer": e.outcome,
            "message": None if e.message_withheld else e.message,
            "policy": (
                {
                    "kind": e.adjust_kind,
                    "action": e.intended_action,
                    "sent": e.intended_outcome,
                    "withheld": e.message_withheld,
                }
                if e.adjusted
                else None
            ),
        }
        for e in events
        if e.action == "offer" and e.outcome is not None
    ]


def to_tuple(outcome: dict[str, Any], issues: list[Issue]) -> tuple[Any, ...]:
    return tuple(outcome[i.name] for i in issues)


def to_dict(outcome: tuple[Any, ...], issues: list[Issue]) -> dict[str, Any]:
    return {i.name: v for i, v in zip(issues, outcome, strict=True)}


def build_outcome_space(scenario: Scenario) -> tuple[Any, list[NegIssue]]:
    """Build the NegMAS outcome space, preserving issue order for tuple mapping."""
    neg_issues = [make_issue(list(i.values), name=i.name) for i in scenario.issues]
    return make_os(neg_issues), neg_issues


def build_ufun(
    prefs: PartyPreferences, scenario: Scenario, outcome_space: Any
) -> MappingUtilityFunction:
    """Exact weighted-additive ufun as a mapping over the enumerated outcomes.

    The reserved value is the party's BATNA so NegMAS shares our reservation.
    """
    issues = list(scenario.issues)
    mapping = {
        to_tuple(o, issues): outcome_utility(prefs, issues, o) for o in enumerate_outcomes(issues)
    }
    return MappingUtilityFunction(
        mapping, outcome_space=outcome_space, reserved_value=reservation_utility(prefs)
    )


class PolicyNegotiator(SAONegotiator):  # type: ignore[misc]  # NegMAS ships no type stubs
    """Adapts a rfq-bench negotiation policy to a NegMAS SAO negotiator."""

    def __init__(
        self,
        *,
        policy: NegotiationPolicy,
        role: Role,
        scenario: Scenario,
        deadline: int,
        events: list[Event],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._policy = policy
        self._party = role
        self._scenario = scenario
        self._issues = list(scenario.issues)
        self._deadline = deadline
        self._events = events
        self._made: list[dict[str, Any]] = []
        self._received: list[dict[str, Any]] = []
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_cached_tokens = 0
        self.total_cost = 0.0
        self.cost_reported = False

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        standing: dict[str, Any] | None = None
        offer = state.current_offer
        if (
            offer is not None
            and state.current_proposer is not None
            and state.current_proposer != self.id
        ):
            standing = to_dict(offer, self._issues)
            if not self._received or self._received[-1] != standing:
                self._received.append(standing)

        neg_state = NegotiationState(
            role=self._party,
            prefs=self._scenario.preferences(self._party),
            issues=self._issues,
            round=state.step,
            deadline=self._deadline,
            opponent_offers=list(self._received),
            my_offers=list(self._made),
            history=_history(self._events),
        )
        move = decide(self._policy, standing, neg_state)
        self.total_tokens += move.token_cost
        self.total_prompt_tokens += move.prompt_tokens
        self.total_cached_tokens += move.cached_tokens
        if move.cost_usd is not None:
            self.total_cost += move.cost_usd
            self.cost_reported = True

        if move.error:
            # Unusable model reply -> abort the episode (excluded from scoring).
            self._events.append(_event(state.step, self._party, "terminate", None, move))
            return SAOResponse(ResponseType.END_NEGOTIATION, None)
        if move.action == "accept" and standing is not None:
            self._events.append(_event(state.step, self._party, "accept", standing, move))
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)
        if move.action == "terminate":
            self._events.append(_event(state.step, self._party, "terminate", None, move))
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        outcome = move.outcome if move.outcome is not None else self._policy.propose(neg_state)
        self._made.append(outcome)
        self._events.append(_event(state.step, self._party, "offer", outcome, move))
        return SAOResponse(ResponseType.REJECT_OFFER, to_tuple(outcome, self._issues))
