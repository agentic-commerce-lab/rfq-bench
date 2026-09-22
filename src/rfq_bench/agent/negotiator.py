"""LLMNegotiator: the fixed agent as a NegMAS-style SAO negotiator.

The LLM decides everything — its offers and its accept/terminate calls stand as
given. The negotiator NEVER substitutes a scripted concession value; it only
applies two faithful, real-shop rules and otherwise records an error:

1. **Snap to grid.** An off-grid numeric value is projected onto the nearest
   legal tier (like an ordering system rounding to an orderable option). Logged
   as an adjustment, not an error; the episode is still scored.
2. **Shop reservation floor.** A quote (offer or accept) below the agent's own
   reservation is blocked by policy and clamped up to the floor — a static
   business limit, not a negotiation strategy. Logged as an adjustment.
3. **Hard error.** A reply with no usable move (unparseable, missing an issue, a
   value that cannot be snapped) is logged as an error and aborts the episode,
   which is then excluded from scoring.

Bad-but-legal decisions above the floor are played as-is and scored — a model
that negotiates itself into a poor (but floor-respecting) deal earns that result.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Protocol

from rfq_bench.agent.llm_client import LLMResponse
from rfq_bench.agent.personas import persona_instruction
from rfq_bench.agent.prompts import (
    SYSTEM_PROMPT,
    build_move_tool,
    build_user_payload,
    strategy_instruction,
)
from rfq_bench.core.contracts import Issue
from rfq_bench.core.utility import outcome_utility, reservation_utility
from rfq_bench.strategies.base import Move, NegotiationState, select_outcome

_EPS = 1e-9
logger = logging.getLogger("rfq_bench.agent")


class ChatClient(Protocol):
    def complete(self, system: str, user: str, tool: dict[str, Any]) -> LLMResponse: ...


def _to_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _match_or_snap(value: Any, legal: list[Any]) -> tuple[Any | None, bool]:
    """Resolve ``value`` to a legal tier. Returns (chosen, was_snapped).

    Exact matches (including "100"->100 and 100.0->100 numeric coercion) are not
    snaps. An off-grid number snaps to the nearest legal tier (ties break to the
    lower value). A value that cannot be resolved returns (None, False).
    """
    for lv in legal:
        if type(value) is type(lv) and value == lv:
            return lv, False
    num = _to_number(value)
    if num is not None:
        pairs: list[tuple[Any, float]] = []
        for lv in legal:
            n = _to_number(lv)
            if n is not None:
                pairs.append((lv, n))
        for lv, n in pairs:
            if n == num:
                return lv, False
        if pairs:
            nearest = min(pairs, key=lambda p: (abs(p[1] - num), p[1]))
            return nearest[0], True
    return None, False


def _snap_outcome(
    raw: dict[str, Any], issues: list[Issue]
) -> tuple[dict[str, Any] | None, list[str]]:
    """Project a raw outcome onto the legal grid. Returns (outcome|None, notes)."""
    result: dict[str, Any] = {}
    notes: list[str] = []
    for issue in issues:
        if issue.name not in raw:
            return None, [f"missing issue {issue.name!r}"]
        chosen, snapped = _match_or_snap(raw[issue.name], issue.values)
        if chosen is None:
            return None, [
                f"value {raw[issue.name]!r} for {issue.name!r} is not a legal tier "
                f"and cannot be snapped"
            ]
        result[issue.name] = chosen
        if snapped:
            notes.append(f"{issue.name} {raw[issue.name]}→{chosen}")
    return result, notes


BehaviorKind = Literal["strategy", "persona"]


class LLMNegotiator:
    """LLM agent driven by an OpenAI-compatible client, steered by one behavior.

    ``behavior_kind`` selects which registry resolves ``behavior`` to the single
    per-arm ``approach_instruction`` in the prompt: a seller **strategy** (the
    offline treatment, and the seller side in A2A self-play) or a buyer
    **persona** (the buyer side in A2A self-play). Everything else — the frozen
    system prompt, tool schema, snap-to-grid and reservation-floor rules — is
    identical across both, so the two sides stay symmetric.
    """

    def __init__(
        self,
        behavior: str,
        client: ChatClient,
        *,
        behavior_kind: BehaviorKind = "strategy",
        strict: bool = True,
        system_prompt: str = SYSTEM_PROMPT,
        instruction: str | None = None,
    ) -> None:
        self.behavior = behavior
        self.behavior_kind: BehaviorKind = behavior_kind
        self._system_prompt = system_prompt
        # Back-compat alias: callers/traces that expect ``.strategy`` still work
        # for the strategy side; it is None for a persona-driven negotiator.
        self.strategy = behavior if behavior_kind == "strategy" else None
        self.name = f"llm:{behavior_kind}:{behavior}"
        # ``instruction`` lets the caller inject file-defined guidance (see
        # load_strategy_guidance / load_persona_guidance); otherwise resolve the
        # built-in guidance for this behavior name.
        if instruction is not None:
            self._instruction = instruction
        elif behavior_kind == "persona":
            self._instruction = persona_instruction(behavior)
        else:
            self._instruction = strategy_instruction(behavior)
        self._client = client
        self._strict = strict

    def act(self, standing_offer: dict[str, Any] | None, state: NegotiationState) -> Move:
        user = build_user_payload(state, self._instruction, standing_offer)
        tool = build_move_tool(state.issues, strict=self._strict)
        resp = self._client.complete(self._system_prompt, user, tool)
        move = self._interpret(resp.data, standing_offer, state)
        move.token_cost = resp.tokens
        move.cost_usd = resp.cost_usd
        rationale = resp.data.get("rationale")
        move.rationale = rationale if isinstance(rationale, str) and rationale else None
        move.raw_response = resp.content or None
        move.reasoning = resp.reasoning
        if move.error:
            # Distinguish a harness-induced truncation (hit max_tokens) from a
            # genuine model failure, and log enough to diagnose it.
            if resp.finish_reason == "length":
                move.error_reason = (
                    "response truncated at max_tokens before a complete move "
                    "(raise RFQ_BENCH_MAX_TOKENS)"
                )
            logger.error(
                "LLM agent %s unusable move: %s | finish_reason=%s | raw=%.300s",
                self.name,
                move.error_reason,
                resp.finish_reason,
                resp.content or "<empty>",
            )
        return move

    def _interpret(
        self,
        data: dict[str, Any],
        standing_offer: dict[str, Any] | None,
        state: NegotiationState,
    ) -> Move:
        action = data.get("action")
        d = reservation_utility(state.prefs)

        if action == "terminate":
            return Move("terminate", None)

        if action == "accept" and standing_offer is not None:
            if outcome_utility(state.prefs, state.issues, standing_offer) >= d - _EPS:
                return Move("accept", standing_offer)
            # Below-floor accept -> shop policy block: counter at the floor instead.
            return Move(
                "offer",
                self._floor_offer(state),
                adjusted=True,
                adjust_reason=(
                    "shop policy: cannot accept a deal below the reservation floor; "
                    "countered at the floor"
                ),
                intended_action="accept",
                intended_outcome=standing_offer,
            )

        # Anything else is treated as an offer attempt. The package comes from the
        # flat top-level issue fields; tolerate a nested {"outcome": {...}} too.
        raw = {i.name: data[i.name] for i in state.issues if i.name in data}
        if not raw and isinstance(data.get("outcome"), dict):
            raw = dict(data["outcome"])
        if not raw:
            return self._error_move(
                reason=f"no package in reply (action={action!r}); expected a value per issue"
            )
        snapped, notes = _snap_outcome(raw, state.issues)
        if snapped is None:
            return self._error_move(reason="; ".join(notes) or "unusable outcome")

        was_snapped = bool(notes)
        if outcome_utility(state.prefs, state.issues, snapped) < d - _EPS:
            # Below-floor offer -> shop policy block: raise to the floor.
            return Move(
                "offer",
                self._floor_offer(state),
                adjusted=True,
                adjust_reason="shop policy: offer below the reservation floor; raised to the floor",
                intended_action="offer",
                intended_outcome=raw,
            )
        if was_snapped:
            return Move(
                "offer",
                snapped,
                adjusted=True,
                adjust_reason="snapped off-grid value(s) to nearest legal tier: "
                + ", ".join(notes),
                intended_action="offer",
                intended_outcome=raw,
            )
        return Move("offer", snapped)

    def _floor_offer(self, state: NegotiationState) -> dict[str, Any]:
        """Most generous legal offer still at/above the reservation floor."""
        return select_outcome(state.prefs, state.issues, reservation_utility(state.prefs))

    def _error_move(self, *, reason: str) -> Move:
        # Logged once in ``act`` with finish_reason + raw args for diagnosis.
        return Move("terminate", None, error=True, error_reason=reason)

    # Protocol compatibility (unused: the engine always calls ``act``).
    def propose(self, state: NegotiationState) -> dict[str, Any]:
        return self._floor_offer(state)

    def accepts(self, opponent_outcome: dict[str, Any], state: NegotiationState) -> bool:
        floor = reservation_utility(state.prefs)
        return outcome_utility(state.prefs, state.issues, opponent_outcome) >= floor - _EPS
