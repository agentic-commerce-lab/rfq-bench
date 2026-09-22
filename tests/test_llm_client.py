"""Cost extraction from OpenAI-compatible usage objects (no network)."""

from __future__ import annotations

from types import SimpleNamespace

from rfq_bench.agent.llm_client import _extract_cost, _parse_tool_call


def _msg(*, tool_args=None, content=None):
    tool_calls = None
    if tool_args is not None:
        tool_calls = [SimpleNamespace(function=SimpleNamespace(arguments=tool_args))]
    return SimpleNamespace(tool_calls=tool_calls, content=content)


def test_parse_prefers_tool_call_arguments() -> None:
    data, raw = _parse_tool_call(_msg(tool_args='{"action":"offer","price":100}'))
    assert data == {"action": "offer", "price": 100}
    assert raw == '{"action":"offer","price":100}'


def test_parse_recovers_move_from_content_when_tool_call_empty() -> None:
    # Reasoning model: empty tool call, move JSON returned as message content.
    content = 'Analyzing buyer demands...\n{"action":"offer","price":150}'
    data, raw = _parse_tool_call(_msg(tool_args="", content=content))
    assert data == {"action": "offer", "price": 150}
    assert raw == content


def test_parse_recovers_from_content_when_no_tool_calls_at_all() -> None:
    data, _ = _parse_tool_call(_msg(content='{"action":"accept"}'))
    assert data == {"action": "accept"}


def test_parse_empty_when_nothing_usable() -> None:
    # Pure reasoning prose, no JSON anywhere -> genuine hard error upstream.
    data, raw = _parse_tool_call(_msg(content="Analyzing buyer demands, no offer yet."))
    assert data == {}
    assert "Analyzing" in raw


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
