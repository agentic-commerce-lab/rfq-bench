"""Cost extraction from OpenAI-compatible usage objects (no network)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rfq_bench.agent.llm_client import _extract_cost, _parse_tool_call


def _msg(*, tool_args=None, content=None, reasoning=None):
    tool_calls = None
    if tool_args is not None:
        tool_calls = [SimpleNamespace(function=SimpleNamespace(arguments=tool_args))]
    return SimpleNamespace(tool_calls=tool_calls, content=content, reasoning=reasoning)


def test_parse_prefers_tool_call_arguments() -> None:
    data, raw = _parse_tool_call(_msg(tool_args='{"action":"offer","price":100}'))
    assert data == {"action": "offer", "price": 100}
    assert raw == '{"action":"offer","price":100}'


def test_content_is_not_treated_as_a_move() -> None:
    # Tool calls are the only move channel: a JSON move in message content is NOT
    # accepted. It is kept as raw for audit; the negotiator re-asks for a tool call.
    content = 'Analyzing buyer demands...\n{"action":"offer","price":150}'
    data, raw = _parse_tool_call(_msg(tool_args="", content=content))
    assert data == {}
    assert raw == content


def test_reasoning_trace_is_never_mined_for_a_move() -> None:
    # A move-shaped object in the chain-of-thought is intermediate, not submitted.
    reasoning = 'Thinking... I will offer {"action":"offer","price":150}'
    data, _ = _parse_tool_call(_msg(content="", reasoning=reasoning))
    assert data == {}


def test_parse_empty_when_no_tool_call() -> None:
    # Pure reasoning prose, no tool call -> recoverable error upstream (re-asked).
    data, raw = _parse_tool_call(_msg(content="Analyzing buyer demands, no offer yet."))
    assert data == {}
    assert "Analyzing" in raw


def test_tool_args_tolerate_a_code_fence() -> None:
    # Some providers fence the tool arguments; that is still the tool channel.
    args = '```json\n{"action":"offer","price":100}\n```'
    data, _ = _parse_tool_call(_msg(tool_args=args))
    assert data == {"action": "offer", "price": 100}


def test_cost_from_direct_attribute() -> None:
    # OpenRouter exposes usage.cost (USD).
    usage = SimpleNamespace(total_tokens=1500, cost=0.0042)
    assert _extract_cost(usage) == 0.0042


def test_cost_from_model_extra() -> None:
    # The OpenAI SDK stashes non-standard fields in model_extra.
    usage = SimpleNamespace(total_tokens=1500, cost=None, model_extra={"cost": 0.001})
    assert _extract_cost(usage) == 0.001


def test_no_cost_returns_none() -> None:
    # Plain OpenAI/Azure/vLLM: usage has no cost.
    usage = SimpleNamespace(total_tokens=1500)
    assert _extract_cost(usage) is None
    assert _extract_cost(None) is None


def test_non_numeric_cost_is_ignored() -> None:
    usage = SimpleNamespace(cost="n/a")
    assert _extract_cost(usage) is None


class _FakeCreate:
    """Plays back queued outcomes (a response object, or an exception to raise)."""

    def __init__(self, outcomes: list) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0
        self.kwargs: dict = {}

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        out = self._outcomes[min(self.calls, len(self._outcomes) - 1)]
        self.calls += 1
        if isinstance(out, BaseException):
            raise out
        return out


def _client(outcomes: list, *, retries: int = 3, tool_choice: str = "forced"):
    from rfq_bench.agent.llm_client import LLMClient
    from rfq_bench.agent.settings import AgentSettings

    settings = AgentSettings(
        RFQ_BENCH_TRANSIENT_RETRIES=retries,
        RFQ_BENCH_TRANSIENT_BACKOFF=1.0,
        RFQ_BENCH_TOOL_CHOICE=tool_choice,
    )
    c = LLMClient(settings)
    create = _FakeCreate(outcomes)
    # Bypass _ensure() by pre-seeding a fake OpenAI-compatible client.
    c._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    c.delays = []  # type: ignore[attr-defined]
    c._sleep = c.delays.append  # record the backoff instead of waiting
    return c, create


_TOOL = {"function": {"name": "submit_move"}}


def _ok(price: int = 100, tokens: int = 12):
    message = SimpleNamespace(
        tool_calls=[
            SimpleNamespace(
                function=SimpleNamespace(arguments=f'{{"action":"offer","price":{price}}}')
            )
        ],
        content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    return SimpleNamespace(choices=[choice], usage=SimpleNamespace(total_tokens=tokens))


def _no_choices(error=None, tokens: int = 3):
    return SimpleNamespace(choices=None, usage=SimpleNamespace(total_tokens=tokens), error=error)


def test_complete_parses_a_normal_tool_call() -> None:
    c, create = _client([_ok()])
    out = c.complete("sys", "user", _TOOL)
    assert out.data == {"action": "offer", "price": 100}
    assert out.finish_reason == "tool_calls" and out.provider_error is None
    assert out.tokens == 12 and create.calls == 1 and c.delays == []


def test_null_choices_is_retried_then_recovers() -> None:
    # OpenRouter upstream failure (200, choices=null) once, then a real reply.
    c, create = _client([_no_choices({"code": 502, "message": "upstream error"}), _ok()])
    out = c.complete("sys", "user", _TOOL)
    assert create.calls == 2
    assert out.data == {"action": "offer", "price": 100}
    assert out.provider_error is None
    assert out.tokens == 15  # usage of the failed attempt is still counted
    assert c.delays == [1.0]


def test_null_choices_exhausts_with_exponential_backoff() -> None:
    c, create = _client([_no_choices({"code": 502, "message": "upstream error"})], retries=3)
    out = c.complete("sys", "user", _TOOL)
    assert create.calls == 4  # initial + 3 retries
    assert c.delays == [1.0, 2.0, 4.0]
    assert out.data == {} and out.finish_reason == "no_choices"
    assert out.provider_error == "502: upstream error"


def test_error_body_is_read_from_model_extra() -> None:
    # The OpenAI SDK keeps unknown fields like OpenRouter's `error` in model_extra.
    resp = SimpleNamespace(
        choices=None, usage=None, model_extra={"error": {"message": "overloaded"}}
    )
    c, _ = _client([resp], retries=0)
    assert c.complete("sys", "user", _TOOL).provider_error == "overloaded"


def test_content_filter_is_not_retried() -> None:
    # The identical prompt would be blocked again — don't burn retries on it.
    choice = SimpleNamespace(message=None, finish_reason="content_filter")
    c, create = _client([SimpleNamespace(choices=[choice], usage=None)])
    out = c.complete("sys", "user", _TOOL)
    assert create.calls == 1 and c.delays == []
    assert out.finish_reason == "content_filter"
    assert out.provider_error == "finish_reason=content_filter"


class _StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class APIConnectionError(Exception):  # name-matched like the openai SDK class
    pass


def test_rate_limit_and_connection_errors_are_retried() -> None:
    c, create = _client([_StatusError(429), APIConnectionError("reset"), _ok()])
    out = c.complete("sys", "user", _TOOL)
    assert create.calls == 3 and c.delays == [1.0, 2.0]
    assert out.data == {"action": "offer", "price": 100}


def test_non_transient_error_is_raised_immediately() -> None:
    import pytest

    c, create = _client([_StatusError(400), _ok()])
    with pytest.raises(_StatusError):
        c.complete("sys", "user", _TOOL)
    assert create.calls == 1 and c.delays == []


def test_transient_error_is_raised_once_exhausted() -> None:
    import pytest

    c, create = _client([_StatusError(503)], retries=2)
    with pytest.raises(_StatusError):
        c.complete("sys", "user", _TOOL)
    assert create.calls == 3  # the episode runner's --retries takes over from here


def test_cached_tokens_are_read_from_either_details_shape() -> None:
    from rfq_bench.agent.llm_client import _extract_cached_tokens

    as_object = SimpleNamespace(prompt_tokens_details=SimpleNamespace(cached_tokens=1024))
    as_dict = SimpleNamespace(
        prompt_tokens_details=None, model_extra={"prompt_tokens_details": {"cached_tokens": 512}}
    )
    assert _extract_cached_tokens(as_object) == 1024
    assert _extract_cached_tokens(as_dict) == 512
    assert _extract_cached_tokens(SimpleNamespace()) == 0
    assert _extract_cached_tokens(None) == 0


def test_cached_tokens_accumulate_across_transient_retries() -> None:
    first = _no_choices({"code": 502, "message": "upstream"})
    first.usage = SimpleNamespace(
        total_tokens=3, prompt_tokens=100, prompt_tokens_details=SimpleNamespace(cached_tokens=80)
    )
    ok = _ok()
    ok.usage = SimpleNamespace(
        total_tokens=12, prompt_tokens=100, prompt_tokens_details=SimpleNamespace(cached_tokens=90)
    )
    c, _ = _client([first, ok])
    out = c.complete("sys", "user", _TOOL)
    assert out.prompt_tokens == 200 and out.cached_tokens == 170


def test_tool_choice_forced_names_the_function_by_default() -> None:
    c, create = _client([_ok()])
    c.complete("sys", "user", _TOOL)
    assert create.kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "submit_move"},
    }


def test_tool_choice_auto_is_forwarded_and_recorded() -> None:
    c, create = _client([_ok()], tool_choice="auto")
    c.complete("sys", "user", _TOOL)
    assert create.kwargs["tool_choice"] == "auto"
    assert c._settings.to_llm_config().tool_choice == "auto"


class _BadRequest(Exception):
    status_code = 400


def test_tool_choice_rejection_fails_fast_with_hint() -> None:
    err = _BadRequest('tool_choice: type "tool" and "any" are not supported for this model.')
    c, create = _client([err])
    with pytest.raises(RuntimeError, match="RFQ_BENCH_TOOL_CHOICE=auto"):
        c.complete("sys", "user", _TOOL)
    assert create.calls == 1 and c.delays == []


def test_legacy_llm_config_defaults_to_forced() -> None:
    from rfq_bench.core.contracts import LLMConfig

    cfg = LLMConfig.model_validate(
        {"base_url": "u", "model": "m", "temperature": 0.0, "max_tokens": 1}
    )
    assert cfg.tool_choice == "forced"
