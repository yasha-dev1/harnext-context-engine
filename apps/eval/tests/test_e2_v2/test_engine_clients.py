"""Engine model clients over a mocked HTTP transport: infra-error classification with bounded
retries, provider errors, usage and cost (reported or computed at list price), config hash."""

import asyncio
import json

import httpx
import pytest
from harnext_eval.e2.v2.engine.clients import (
    ChatClient,
    InfraError,
    JevClient,
    Meter,
    ProviderError,
    price,
)
from harnext_eval.e2.v2.engine.config import EngineConfig, config_hash, for_endpoint

CONFIG = for_endpoint("openrouter", retries=2, backoff_s=0.0)
JEV_OK = {"model": "typesafe/jev-1.13-x", "answers": {"q": {"type": "noul", "noul": 0.7}},
          "usage": {"input_tokens": 400, "output_tokens": 20, "cost": 1.7e-05}}  # fmt: skip
CHAT_OK = {"model": "openai/gpt-oss-20b", "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],  # fmt: skip
           "usage": {"prompt_tokens": 1000, "completion_tokens": 100}}  # fmt: skip


def client(kind, replies, config=CONFIG):
    """A client whose transport answers from `replies` (status, body) or raises exceptions."""
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        status, body = reply
        return httpx.Response(status, json=body)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    made = (JevClient if kind == "jev" else ChatClient)(config, key="k", http=http)
    return made, seen


def jev(replies):
    made, seen = client("jev", replies)
    return asyncio.run(made.decide({"question": "x"}, {"q": {"type": "noul", "instructions": "`question`?"}})), seen  # fmt: skip


def chat(replies, config=CONFIG):
    made, seen = client("chat", replies, config)
    return asyncio.run(made.complete([{"role": "user", "content": "x"}], purpose="t")), seen


def test_jev_usage_and_reported_cost():
    (answers, call), seen = jev([(200, JEV_OK)])
    assert answers["q"]["noul"] == 0.7
    assert (call.kind, call.input_tokens, call.output_tokens, call.cost_usd, call.cost_reported) == ("jev", 400, 20, 1.7e-05, True)  # fmt: skip
    assert seen[0]["model"] == "typesafe/jev-1.13" and "q" in seen[0]["questions"]


def test_chat_cost_is_computed_at_list_price_when_absent():
    (message, call), seen = chat([(200, CHAT_OK)])
    assert message["content"] == "hi" and message["finish_reason"] == "stop"
    assert call.cost_reported is False
    assert call.cost_usd == pytest.approx(price(1000, 100, (0.018, 0.09))) == pytest.approx(2.7e-05)  # fmt: skip
    assert seen[0]["usage"] == {"include": True} and seen[0]["reasoning"] == {"effort": "low"}
    reported = {**CHAT_OK, "usage": {**CHAT_OK["usage"], "cost": 5e-06}}
    (_, call), _ = chat([(200, reported)])
    assert (call.cost_usd, call.cost_reported) == (5e-06, True)


def test_local_ollama_defaults():
    config = EngineConfig(backoff_s=0)
    reply = {**CHAT_OK, "choices": [{"message": {"role": "assistant", "content": "{}", "reasoning": "think"}, "finish_reason": "stop"}]}  # fmt: skip
    made, seen = client("chat", [(200, reply)], config)
    made.key = None
    fmt = {"type": "json_schema", "json_schema": {"name": "a", "schema": {}}}
    _, call = asyncio.run(made.complete([{"role": "user", "content": "x"}], response_format=fmt))
    body = seen[0]
    assert body["model"] == "gpt-oss-20b-ctx32k" and body["reasoning_effort"] == "low"
    assert body["seed"] == 7 and body["temperature"] == 0 and body["max_tokens"] == 2048
    assert "response_format" not in body  # it would hide the reasoning tokens from usage
    assert call.reasoning is True and call.cost_reported is False
    with pytest.raises(ProviderError, match="truncated"):
        chat(
            [(200, {**CHAT_OK, "usage": {"prompt_tokens": 32768, "completion_tokens": 3}})], config
        )


def test_endpoint_options_are_configurable():
    config = EngineConfig(chat_base_url="http://localhost:11434/v1/", chat_model="gpt-oss:20b", chat_key_env=None,  # fmt: skip
                          chat_options={"options": {"num_ctx": 32768}}, temperature=None, backoff_s=0)  # fmt: skip
    urls = []
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: urls.append((str(r.url), r.headers.get("authorization"), json.loads(r.content))) or httpx.Response(200, json=CHAT_OK)))  # fmt: skip
    made = ChatClient(config, http=http)
    asyncio.run(made.complete([{"role": "user", "content": "x"}]))
    url, auth, body = urls[0]
    assert url == "http://localhost:11434/v1/chat/completions" and auth is None
    assert body["model"] == "gpt-oss:20b" and body["options"] == {"num_ctx": 32768} and "temperature" not in body  # fmt: skip


@pytest.mark.parametrize("status", [401, 402, 403])
def test_auth_and_credit_failures_are_infra_immediately(status):
    replies = [(status, {"error": "no"}), (200, JEV_OK)]
    with pytest.raises(InfraError, match=f"HTTP {status}"):
        jev(replies)
    assert len(replies) == 1  # not retried


def test_rate_limits_and_server_errors_are_retried_then_infra():
    (answers, _), seen = jev([(429, {}), (503, {}), (200, JEV_OK)])
    assert answers["q"]["noul"] == 0.7 and len(seen) == 3
    with pytest.raises(InfraError, match="after 3 attempts"):
        jev([(429, {}), (500, {}), (502, {})])


def test_timeouts_and_unreachable_endpoints_are_infra():
    with pytest.raises(InfraError, match="timeout"):
        chat([httpx.ReadTimeout("slow")] * 3)
    with pytest.raises(InfraError, match="unreachable"):
        chat([httpx.ConnectError("refused")] * 3)
    (message, _), _ = chat([httpx.ConnectError("refused"), (200, CHAT_OK)])
    assert message["content"] == "hi"


def test_errors_inside_a_200_are_classified():
    with pytest.raises(InfraError, match="402"):
        chat([(200, {"error": {"code": 402, "message": "Insufficient credits"}})])
    with pytest.raises(ProviderError):
        chat([(200, {"error": {"code": 400, "message": "bad tool schema"}})])
    with pytest.raises(ProviderError, match="HTTP 400"):
        jev([(400, {"error": "questions invalid"})])
    with pytest.raises(ProviderError, match="lacks"):
        jev([(200, {**JEV_OK, "answers": {}})])


def test_meter_splits_usage_by_model():
    meter = Meter()
    (_, jev_call), _ = jev([(200, JEV_OK)])
    (_, chat_call), _ = chat([(200, CHAT_OK)])
    meter.add(jev_call)
    meter.add(chat_call)
    summary = meter.summary()
    assert summary["jev"]["calls"] == summary["chat"]["calls"] == 1
    assert summary["cost_usd"] == pytest.approx(1.7e-05 + 2.7e-05)
    assert summary["input_tokens"] == 1400 and summary["chat"]["cost_reported"] is False


def test_config_hash_is_stable_and_ignores_transport():
    base = config_hash(EngineConfig())
    assert base == config_hash(EngineConfig())
    assert base == config_hash(EngineConfig(chat_base_url="http://localhost:11434/v1", timeout_s=5))  # fmt: skip
    assert base != config_hash(EngineConfig(sql_fallback="never"))
    assert base != config_hash(EngineConfig(chat_model="gpt-oss:20b"))
    assert base != config_hash(for_endpoint("openrouter"))


def test_a_slow_reply_is_cut_by_the_total_timeout():
    async def slow(request):
        await asyncio.sleep(2)
        return httpx.Response(200, json=CHAT_OK)

    config = for_endpoint("openrouter", timeout_s=0.2, retries=1, backoff_s=0)
    made = ChatClient(config, key="k", http=httpx.AsyncClient(transport=httpx.MockTransport(slow)))
    with pytest.raises(InfraError, match="timeout after 0.2 s"):
        asyncio.run(made.complete([{"role": "user", "content": "x"}]))


def test_408_is_retried_and_413_is_a_prompt_error():
    (answers, call), seen = jev([(408, {}), (200, JEV_OK)])
    assert (
        answers["q"]["noul"] == 0.7
        and call.attempts == 2
        and call.failures[0].startswith("HTTP 408")
    )
    assert call.unmetered_usd > 0  # the failed attempt returned no usage: an upper bound is kept
    from harnext_eval.e2.v2.engine.clients import PromptTooLarge

    with pytest.raises(PromptTooLarge, match="HTTP 413"):
        chat([(413, {"error": "too large"})])


def test_failed_calls_carry_their_metering():
    made, _ = client("jev", [(429, {}), (500, {}), (502, {})])
    with pytest.raises(InfraError) as caught:
        asyncio.run(made.decide({"q": 1}, {"q": {"type": "noul", "instructions": "`q`?"}}))
    assert caught.value.call.attempts == 3 and caught.value.call.cost_usd == 0 and caught.value.call.unmetered_usd > 0  # fmt: skip
    made, _ = client("jev", [(200, {**JEV_OK, "answers": {}})])
    with pytest.raises(ProviderError) as caught:
        asyncio.run(made.decide({"q": 1}, {"q": {"type": "noul", "instructions": "`q`?"}}))
    assert caught.value.call.cost_usd == 1.7e-05  # answered and billed: metered


def test_meter_reports_retries_cancellations_and_unmetered_cost():
    from harnext_eval.e2.v2.engine.clients import Call

    meter = Meter()
    meter.add(Call("chat", "m", "p", 10, 1, 0.1, True, 1.0, attempts=3, unmetered_usd=0.02))
    meter.add(Call("chat", "m", "p", 0, 0, 0.0, False, 5.0, cancelled=True, unmetered_usd=0.03))
    totals = meter.summary()
    assert totals["chat"]["retries"] == 2 and totals["chat"]["cancelled"] == 1
    assert totals["unmetered_upper_usd"] == pytest.approx(0.05)


def test_the_local_endpoint_rejects_an_overlong_prompt():
    """Against the running local Ollama (skipped when it is not reachable): a prompt beyond the
    32k context comes back truncated, which the client turns into PromptTooLarge."""
    from harnext_eval.e2.v2.engine.clients import PromptTooLarge

    try:
        httpx.get("http://localhost:11434/api/version", timeout=2).raise_for_status()
    except httpx.HTTPError:
        pytest.skip("local Ollama not reachable")
    made = ChatClient(EngineConfig(max_tokens=8, timeout_s=300))
    words = " ".join(f"w{i}" for i in range(40_000))
    with pytest.raises(PromptTooLarge, match="truncated") as caught:
        asyncio.run(made.complete([{"role": "user", "content": f"Count the words: {words}"}]))
    assert caught.value.call.input_tokens >= 32768


def test_an_unparsable_tool_call_is_a_model_error_not_retried():
    from harnext_eval.e2.v2.engine.clients import ModelOutputError

    body = {"error": {"message": "error parsing tool call: raw='We need the key. So answer 3.', err=invalid character"}}  # fmt: skip
    replies = [(500, body), (200, CHAT_OK)]
    with pytest.raises(ModelOutputError) as caught:
        chat(replies)
    assert caught.value.raw == "We need the key. So answer 3." and len(replies) == 1  # not retried


def test_reasoning_and_cached_tokens_are_recorded():
    usage = {"prompt_tokens": 1000, "completion_tokens": 100, "cost": 1e-05,
             "completion_tokens_details": {"reasoning_tokens": 40}, "prompt_tokens_details": {"cached_tokens": 600}}  # fmt: skip
    (_, call), _ = chat([(200, {**CHAT_OK, "usage": usage})])
    assert (call.reasoning_tokens, call.cached_tokens, call.cost_usd) == (40, 600, 1e-05)
    meter = Meter()
    meter.add(call)
    assert (
        meter.summary()["chat"]["reasoning_tokens"] == 40
        and meter.summary()["chat"]["cached_tokens"] == 600
    )
