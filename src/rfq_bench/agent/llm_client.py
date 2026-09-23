"""Thin OpenAI-compatible client for the fixed agent.

Targets any endpoint that speaks the OpenAI chat/completions API; the base URL,
model, and decoding parameters come from :class:`AgentSettings`. Nothing here is
provider-specific beyond the wire format.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rfq_bench.agent.settings import AgentSettings

logger = logging.getLogger("rfq_bench.agent")

# HTTP statuses worth retrying unchanged: timeout, conflict, rate limit (+ any 5xx).
_TRANSIENT_STATUS = frozenset({408, 409, 429})
# SDK/stdlib exception classes that signal a dropped or timed-out connection.
_TRANSIENT_EXC_NAMES = frozenset({"APIConnectionError", "APITimeoutError"})


@dataclass
class LLMResponse:
    data: dict[str, Any]
    tokens: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Prompt tokens the provider served from its prompt cache (billed at the
    # cache-read rate); 0 when not reported.
    cached_tokens: int = 0
    # Real spend in USD, when the provider reports it (e.g. OpenRouter's
    # ``usage.cost``); None when the endpoint returns no cost.
    cost_usd: float | None = None
    # The verbatim function-call arguments the model submitted (or message
    # content if it returned no tool call), for audit.
    content: str = ""
    # The model's chain-of-thought, when a reasoning model exposes it separately
    # (OpenRouter ``message.reasoning`` / DeepSeek ``reasoning_content``); else None.
    reasoning: str | None = None
    # The choice's finish_reason ("stop" | "tool_calls" | "length" | ...). "length"
    # means the reply was truncated at max_tokens; "no_choices" means the provider
    # returned no completion at all.
    finish_reason: str | None = None
    # Set when the provider returned no completion even after transient retries:
    # its error body (e.g. OpenRouter's ``{"error": {...}}``) or the finish_reason.
    # Marks an infrastructure failure, not a model one.
    provider_error: str | None = None


class LLMClient:
    """Chat client returning a parsed JSON object plus a token count."""

    def __init__(self, settings: AgentSettings) -> None:
        self._settings = settings
        self._client: Any | None = None
        # Injectable so tests can exercise the backoff without actually waiting.
        self._sleep: Callable[[float], None] = time.sleep

    def _ensure(self) -> Any:
        if self._client is None:
            # Imported lazily so the package works without the openai extra installed
            # for the deterministic (scripted) path.
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self._settings.base_url,
                api_key=self._settings.api_key or "not-needed",
                timeout=self._settings.timeout,
            )
        return self._client

    def complete(self, system: str, user: str, tool: dict[str, Any]) -> LLMResponse:
        """Force a single call to ``tool`` and return its parsed arguments.

        The move is submitted via real function/tool calling (never free-form
        JSON): ``tool_choice`` forces the model to call the negotiation function,
        and we read the arguments off ``message.tool_calls``. Models that do not
        support function calling are out of scope.
        """
        client = self._ensure()
        tool_name = tool["function"]["name"]
        kwargs: dict[str, Any] = {
            "model": self._settings.model,
            "temperature": self._settings.temperature,
            "max_tokens": self._settings.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "tools": [tool],
            "tool_choice": {"type": "function", "function": {"name": tool_name}},
        }
        if self._settings.seed is not None:
            kwargs["seed"] = self._settings.seed
        if self._settings.include_cost:
            # OpenRouter returns actual USD spend in usage.cost when asked to.
            kwargs["extra_body"] = {"usage": {"include": True}}
        # Transient provider failures are retried with the IDENTICAL request and
        # exponential backoff. The model did nothing wrong, so there is no
        # corrective nudge here (that is the negotiator's re-ask, for format
        # failures). Usage/cost accumulate across attempts — a failed call may
        # still be billed.
        attempts = max(0, self._settings.transient_retries) + 1
        tokens = prompt = completion = cached = 0
        cost = 0.0
        cost_seen = False
        for attempt in range(attempts):
            last = attempt == attempts - 1
            try:
                resp = client.chat.completions.create(**kwargs)
            except Exception as exc:
                if last or not _is_transient_exc(exc):
                    raise  # exhausted or non-transient: the episode runner handles it
                self._backoff(attempt, attempts, f"{type(exc).__name__}: {exc}")
                continue

            usage = getattr(resp, "usage", None)
            tokens += int(getattr(usage, "total_tokens", 0) or 0)
            prompt += int(getattr(usage, "prompt_tokens", 0) or 0)
            completion += int(getattr(usage, "completion_tokens", 0) or 0)
            cached += _extract_cached_tokens(usage)
            c = _extract_cost(usage)
            if c is not None:
                cost += c
                cost_seen = True

            choices = getattr(resp, "choices", None) or []
            choice = choices[0] if choices else None
            message = getattr(choice, "message", None) if choice is not None else None
            if message is None:
                # No completion at all: OpenRouter returns a 200 with choices=null
                # and an ``error`` body when the upstream provider fails; others do
                # the same on a moderation block. Retry the former; a content
                # filter would block the identical prompt again, so don't.
                finish = getattr(choice, "finish_reason", None) or "no_choices"
                detail = _extract_provider_error(resp) or f"finish_reason={finish}"
                if finish != "content_filter" and not last:
                    self._backoff(attempt, attempts, f"no completion ({detail})")
                    continue
                logger.error(
                    "provider returned no completion after %d attempt(s): %s", attempt + 1, detail
                )
                return LLMResponse(
                    data={},
                    tokens=tokens,
                    prompt_tokens=prompt,
                    completion_tokens=completion,
                    cached_tokens=cached,
                    cost_usd=cost if cost_seen else None,
                    content="",
                    reasoning=None,
                    finish_reason=finish,
                    provider_error=detail,
                )

            data, raw_args = _parse_tool_call(message)
            return LLMResponse(
                data=data,
                tokens=tokens,
                prompt_tokens=prompt,
                completion_tokens=completion,
                cached_tokens=cached,
                cost_usd=cost if cost_seen else None,
                content=raw_args,
                reasoning=_extract_reasoning(message),
                finish_reason=getattr(choice, "finish_reason", None),
            )
        raise RuntimeError("unreachable")  # pragma: no cover

    def _backoff(self, attempt: int, attempts: int, why: str) -> None:
        delay = self._settings.transient_backoff_s * (2**attempt)
        logger.warning(
            "transient provider failure (%s); retry %d/%d in %.1fs",
            why,
            attempt + 1,
            attempts - 1,
            delay,
        )
        self._sleep(delay)


def _extract_cached_tokens(usage: Any) -> int:
    """Prompt tokens served from the provider's cache (0 when not reported).

    OpenAI-compatible endpoints (OpenRouter included) report it as
    ``usage.prompt_tokens_details.cached_tokens``; the details may arrive as an
    object or, for fields the SDK doesn't model, as a plain dict.
    """
    details = getattr(usage, "prompt_tokens_details", None) if usage is not None else None
    if details is None and usage is not None:
        extra = getattr(usage, "model_extra", None)
        if isinstance(extra, dict):
            details = extra.get("prompt_tokens_details")
    if details is None:
        return 0
    value = (
        details.get("cached_tokens")
        if isinstance(details, dict)
        else getattr(details, "cached_tokens", None)
    )
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _is_transient_exc(exc: BaseException) -> bool:
    """A dropped connection, timeout, rate limit, or 5xx — worth retrying unchanged."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in _TRANSIENT_STATUS or status >= 500
    if isinstance(exc, TimeoutError | ConnectionError):
        return True
    return any(cls.__name__ in _TRANSIENT_EXC_NAMES for cls in type(exc).__mro__)


def _extract_provider_error(resp: Any) -> str | None:
    """The provider's error body on a no-completion reply, formatted for the log.

    OpenRouter returns ``{"error": {"code": ..., "message": ...}}`` alongside a
    null ``choices``; the OpenAI SDK keeps that unknown field in ``model_extra``.
    """
    err = getattr(resp, "error", None)
    if err is None:
        extra = getattr(resp, "model_extra", None)
        if isinstance(extra, dict):
            err = extra.get("error")
    if err is None:
        return None
    if isinstance(err, dict):
        code, msg = err.get("code"), err.get("message")
        parts = [str(p) for p in (code, msg) if p not in (None, "")]
        return ": ".join(parts) if parts else json.dumps(err, default=str)
    return str(err)


def _extract_cost(usage: Any) -> float | None:
    """Pull provider-reported USD spend out of the usage object, if present.

    OpenRouter puts it on ``usage.cost``; the OpenAI SDK stashes such non-standard
    fields in ``model_extra``. Standard OpenAI/Azure/vLLM return nothing here.
    """
    if usage is None:
        return None
    cost = getattr(usage, "cost", None)
    if cost is None:
        extra = getattr(usage, "model_extra", None)
        if isinstance(extra, dict):
            cost = extra.get("cost")
    try:
        return float(cost) if cost is not None else None
    except (TypeError, ValueError):
        return None


def _extract_reasoning(message: Any) -> str | None:
    """Pull a reasoning model's chain-of-thought off the message, if present.

    OpenRouter exposes it as ``message.reasoning``; DeepSeek as
    ``reasoning_content``. The OpenAI SDK stashes such non-standard fields in
    ``model_extra``. Standard models return nothing here.
    """
    for attr in ("reasoning", "reasoning_content"):
        val = getattr(message, attr, None)
        if isinstance(val, str) and val.strip():
            return val
    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict):
        for key in ("reasoning", "reasoning_content"):
            val = extra.get(key)
            if isinstance(val, str) and val.strip():
                return val
    return None


def _parse_tool_call(message: Any) -> tuple[dict[str, Any], str]:
    """Extract the move from the model's tool call. Returns (data, raw).

    The **tool call is the only authoritative channel**: ``tool_choice`` forces
    the negotiation function and the move is read from its arguments. We
    deliberately do NOT recover a move from message content or the reasoning
    trace — content is the wrong channel, and the reasoning trace holds
    intermediate, non-final offers, so neither is the model's *submitted*
    decision. A reply with no usable tool call yields ``({}, <content>)``: the
    negotiator re-asks (to elicit a real tool call) and, if that is exhausted,
    treats it as a hard error. ``raw`` is the tool arguments when present, else
    the message content, kept for audit/logging only.
    """
    calls = getattr(message, "tool_calls", None) or []
    for call in calls:
        fn = getattr(call, "function", None)
        args = getattr(fn, "arguments", None) if fn is not None else None
        if isinstance(args, str) and args.strip():
            parsed = _parse_json(args)
            if parsed:
                return parsed, args
    # No usable tool call: keep the message content for audit/logging only, never
    # as a move. The negotiator will re-ask for a proper tool call.
    content = getattr(message, "content", None) or ""
    return {}, content


def _parse_json(content: str) -> dict[str, Any]:
    """Parse tool-call arguments into an object. Returns ``{}`` on failure.

    Tool arguments are normally strict JSON; we tolerate surrounding whitespace
    or a single code fence, but nothing more — free-form prose is not a move
    channel. Never raises, so a malformed argument becomes a recoverable error
    (re-asked, then excluded) upstream rather than crashing.
    """
    text = content.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Tolerate an arg wrapped in a code fence or stray prose by taking the
        # outermost {...} span. Not prose mining — tool args are meant to be JSON.
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return obj if isinstance(obj, dict) else {}
