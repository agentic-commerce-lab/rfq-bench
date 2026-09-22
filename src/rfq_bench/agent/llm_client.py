"""Thin OpenAI-compatible client for the fixed agent.

Targets any endpoint that speaks the OpenAI chat/completions API; the base URL,
model, and decoding parameters come from :class:`AgentSettings`. Nothing here is
provider-specific beyond the wire format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from rfq_bench.agent.settings import AgentSettings


@dataclass
class LLMResponse:
    data: dict[str, Any]
    tokens: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
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
    # means the reply was truncated at max_tokens.
    finish_reason: str | None = None


class LLMClient:
    """Chat client returning a parsed JSON object plus a token count."""

    def __init__(self, settings: AgentSettings) -> None:
        self._settings = settings
        self._client: Any | None = None

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
        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        message = choice.message
        data, raw_args = _parse_tool_call(message)
        usage = getattr(resp, "usage", None)
        total = int(getattr(usage, "total_tokens", 0) or 0)
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        return LLMResponse(
            data=data,
            tokens=total,
            prompt_tokens=prompt,
            completion_tokens=completion,
            cost_usd=_extract_cost(usage),
            content=raw_args,
            reasoning=_extract_reasoning(message),
            finish_reason=getattr(choice, "finish_reason", None),
        )


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
    """Extract the move the model submitted. Returns (data, raw).

    Prefers the forced function call's arguments. But some providers — reasoning
    models especially — ignore ``tool_choice`` and return the move as **message
    content** (a JSON object, sometimes wrapped in prose or a code fence) with an
    empty ``tool_calls``. So when no usable tool-call arguments are present, fall
    back to recovering a JSON move from the message content. Only a genuinely empty
    reply (no tool call and no JSON in content) yields ``{}``, which the negotiator
    turns into a hard error. ``raw`` is the verbatim text used, for audit.
    """
    calls = getattr(message, "tool_calls", None) or []
    for call in calls:
        fn = getattr(call, "function", None)
        args = getattr(fn, "arguments", None) if fn is not None else None
        if isinstance(args, str) and args.strip():
            parsed = _parse_json(args)
            if parsed:
                return parsed, args
    # No usable tool call: try to recover the move from the message content.
    content = getattr(message, "content", None) or ""
    if content.strip():
        parsed = _parse_json(content)
        if parsed:
            return parsed, content
    return {}, content


def _parse_json(content: str) -> dict[str, Any]:
    """Parse a JSON string, recovering the first {...} block if needed.

    Never raises: malformed output returns ``{}`` so the negotiator records a
    hard error (the episode is excluded) rather than crashing.
    """
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        # Best-effort recovery: extract the first {...} block, tolerating junk
        # (prose, code fences, a trailing reasoning trace) around it.
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            obj = json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return obj if isinstance(obj, dict) else {}
