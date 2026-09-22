"""LayaNegotiator: a negotiator driven by the Laya decision model.

Laya decides the move via typed questions in a single forward pass:
- ``accept`` / ``reachable`` (yes/no) → accept the standing offer, or walk away;
- one ``choice`` per issue → **Laya picks each issue's value**, and those picks are
  the offered package. There is no code-side concession schedule — the offer is
  Laya's decision. The strategy (if any) is injected into the state as guidance text
  (like the LLM agent), so it can bias Laya's choice.

Two faithful shop rails still apply (legality, not strategy): an offer Laya picks
below the reservation floor is raised to the floor, and a below-floor accept is not
allowed. An unusable reply is a hard error that excludes the episode.

Caveat (see docs/laya-agent-concept.md): Laya is weak on graded/"what to do"
questions and has no memory across turns, so its offer choice may not concede
sensibly. That is a measurement, not a bug.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from rfq_bench.agent.laya_framing import laya_questions, laya_state
from rfq_bench.core.utility import outcome_utility, reservation_utility
from rfq_bench.strategies.base import Move, NegotiationState, select_outcome

_EPS = 1e-9
logger = logging.getLogger("rfq_bench.agent.laya")


class LayaLike(Protocol):
    def predict(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]: ...


class LayaNegotiator:
    """Non-generative decision-model negotiator (Laya) that picks its own offer."""

    def __init__(
        self,
        client: LayaLike,
        *,
        strategy: str | None = None,
        instruction: str | None = None,
        accept_threshold: float = 0.6,
        walk_threshold: float = 0.25,
    ) -> None:
        self.strategy = strategy
        self.name = "laya" if strategy is None else f"laya:{strategy}"
        self._instruction = instruction  # strategy-guidance text shown to Laya
        self._client = client
        self._accept_threshold = accept_threshold
        self._walk_threshold = walk_threshold

    def act(self, standing_offer: dict[str, Any] | None, state: NegotiationState) -> Move:
        payload = laya_state(state, standing_offer, self._instruction)
        try:
            result = self._client.predict(payload, laya_questions(state.issues, state.prefs))
        except Exception as exc:  # transport error -> let the runner's retry handle it
            logger.error("Laya predict failed: %s", exc)
            raise
        move = self._interpret(result, standing_offer, state)
        if move.error:
            logger.error(
                "Laya unusable decision: %s | raw=%.300s", move.error_reason, move.raw_response
            )
        return move

    def _interpret(
        self, result: dict[str, Any], standing_offer: dict[str, Any] | None, state: NegotiationState
    ) -> Move:
        answers = result.get("answers")
        usage = result.get("usage") or {}
        tokens = int(usage.get("input_tokens", 0) or 0)
        if not isinstance(answers, dict):
            return self._error("Laya returned no answers", tokens)
        try:
            accept_p = float(answers["accept"]["noul"])
            reach_p = float(answers["reachable"]["noul"])
        except (KeyError, TypeError, ValueError):
            return self._error(f"Laya accept/reachable missing: {answers!r}", tokens)

        prefs, issues = state.prefs, state.issues
        d = reservation_utility(prefs)
        raw = json.dumps(answers)

        # Laya's chosen package: one legal value per issue from its `choice` answers.
        package: dict[str, Any] = {}
        for issue in issues:
            ans = answers.get(issue.name)
            choice = ans.get("choice") if isinstance(ans, dict) else None
            if choice is None:
                return self._error(f"Laya made no choice for issue {issue.name!r}", tokens)
            match = next((v for v in issue.values if str(v) == str(choice)), None)
            if match is None:
                return self._error(
                    f"Laya chose {choice!r} for {issue.name!r}, not a legal option", tokens
                )
            package[issue.name] = match

        rationale = f"{self.name}: accept={accept_p:.2f} reachable={reach_p:.2f} offer={package}"

        # Accept the standing offer only if Laya says so AND it clears the floor.
        if standing_offer is not None:
            if (
                accept_p >= self._accept_threshold
                and outcome_utility(prefs, issues, standing_offer) >= d - _EPS
            ):
                return Move(
                    "accept",
                    standing_offer,
                    token_cost=tokens,
                    rationale=rationale,
                    raw_response=raw,
                )
            if reach_p < self._walk_threshold:
                return Move(
                    "terminate", None, token_cost=tokens, rationale=rationale, raw_response=raw
                )

        # Otherwise play Laya's chosen package, kept at/above the reservation floor.
        if outcome_utility(prefs, issues, package) < d - _EPS:
            floor = select_outcome(prefs, issues, d)
            return Move(
                "offer",
                floor,
                token_cost=tokens,
                rationale=rationale,
                raw_response=raw,
                adjusted=True,
                adjust_reason="shop policy: offer below the reservation floor; raised to the floor",
                intended_action="offer",
                intended_outcome=package,
            )
        return Move("offer", package, token_cost=tokens, rationale=rationale, raw_response=raw)

    def _error(self, reason: str, tokens: int) -> Move:
        return Move("terminate", None, error=True, error_reason=reason, token_cost=tokens)

    # Protocol compatibility (unused: the engine always calls ``act``).
    def propose(self, state: NegotiationState) -> dict[str, Any]:
        return select_outcome(state.prefs, state.issues, reservation_utility(state.prefs))

    def accepts(self, opponent_outcome: dict[str, Any], state: NegotiationState) -> bool:
        floor = reservation_utility(state.prefs)
        return outcome_utility(state.prefs, state.issues, opponent_outcome) >= floor - _EPS
