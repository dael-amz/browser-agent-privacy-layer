from __future__ import annotations

import json
from typing import Any

import httpx

from plva_proxy.providers import ProviderSpec
from plva_proxy.tool_probe import (
    CHANNELS,
    ChannelResult,
    Invocation,
    ProbeReport,
    ProbeRunner,
    ProbeState,
    _synthetic_request,
    consumed_result,
    create_app,
    grammar_permits,
    model_emitted,
    parse_invocation,
    schema_summary,
)


def captured_request() -> dict[str, Any]:
    return {
        "model": "runtime-model",
        "messages": [
            {
                "role": "system",
                "content": "[PLVA_TOOL_PROBE_SKILL_BEGIN] synthetic skill",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "private runtime text"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,private-runtime-image"},
                    },
                ],
            },
        ],
        "structured_outputs": {
            "json": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string"},
                    "tool_calls": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "tool_name": {"type": "string", "enum": ["click", "answer"]},
                                "content": {"type": "string"},
                            },
                        },
                    },
                },
            }
        },
        "temperature": 0,
        "max_tokens": 2048,
    }


def completion(*, content: str, tool_calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                }
            }
        ]
    }


def test_schema_summary_and_channel_permissions_are_value_free() -> None:
    summary = schema_summary(captured_request())

    assert summary["has_tools"] is False
    assert summary["action_enums"] == ["answer", "click"]
    assert summary["tool_name_enums"] == ["answer", "click"]
    assert summary["tool_name_unconstrained"] is False
    assert "private runtime text" not in json.dumps(summary)
    assert grammar_permits("native_skill", summary) is False
    assert grammar_permits("structured_action", summary) is False
    assert grammar_permits("free_text_marker", summary) is True


def test_unconstrained_tool_name_permits_novel_structured_action() -> None:
    request = captured_request()
    request["structured_outputs"]["json"]["properties"]["tool_calls"]["items"]["properties"][
        "tool_name"
    ] = {"type": "string"}

    summary = schema_summary(request)

    assert summary["tool_name_unconstrained"] is True
    assert grammar_permits("structured_action", summary) is True


def test_channel_parsers_accept_only_the_exact_synthetic_invocation() -> None:
    native = completion(
        content="",
        tool_calls=[
            {
                "type": "function",
                "function": {
                    "name": "plva_add",
                    "arguments": '{"a":3,"b":4,"request_id":"p65"}',
                },
            }
        ],
    )
    structured = completion(
        content=json.dumps(
            {
                "tool_calls": [
                    {
                        "tool_name": "plva_add",
                        "a": 3,
                        "b": 4,
                        "request_id": "p65",
                    }
                ]
            }
        )
    )
    marker = completion(content='{"thought":"⟦PLVA_TOOL:add:3:4:p65⟧"}')

    for channel, response in zip(CHANNELS, (native, structured, marker), strict=True):
        assert model_emitted(channel, response) is True
        assert parse_invocation(channel, response) == Invocation("plva_add", 3, 4, "p65")

    assert parse_invocation("free_text_marker", completion(content="no marker")) is None
    assert consumed_result(completion(content='{"content":"PLVA_TOOL_ACK:p65:7"}')) is True


def test_synthetic_request_discards_runtime_user_text_and_pixels() -> None:
    request = _synthetic_request(captured_request(), "provider-model", "free_text_marker")
    serialized = json.dumps(request)

    assert "private runtime text" not in serialized
    assert "private-runtime-image" not in serialized
    assert "PLVA_TOOL:add:3:4:p65" in serialized
    assert request["model"] == "provider-model"


async def test_runner_executes_all_three_round_trips_without_real_tools() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        serialized = json.dumps(payload)
        if "PLVA_TOOL_RESULT" in serialized:
            return httpx.Response(
                200,
                json=completion(content='{"content":"PLVA_TOOL_ACK:p65:7"}'),
            )
        if "native PLVA Tool Probe" in serialized:
            return httpx.Response(
                200,
                json=completion(
                    content="",
                    tool_calls=[
                        {
                            "function": {
                                "name": "plva_add",
                                "arguments": '{"a":3,"b":4,"request_id":"p65"}',
                            }
                        }
                    ],
                ),
            )
        if "structured Holo action" in serialized:
            call = {
                "tool_calls": [
                    {
                        "tool_name": "plva_add",
                        "a": 3,
                        "b": 4,
                        "request_id": "p65",
                    }
                ]
            }
            return httpx.Response(200, json=completion(content=json.dumps(call)))
        return httpx.Response(
            200,
            json=completion(content='{"thought":"⟦PLVA_TOOL:add:3:4:p65⟧"}'),
        )

    runner = ProbeRunner(
        provider_name="test",
        provider=ProviderSpec("https://probe.test/v1", "probe-model", ("KEY",)),
        api_key="synthetic-key",
        transport=httpx.MockTransport(respond),
    )

    report = await runner.run(captured_request())

    assert report.skill_loaded is True
    assert all(result.parser_accepted and result.round_trip for result in report.channels.values())
    assert report.recommendation == "native_skill"


async def test_probe_app_retains_only_safe_report() -> None:
    report = ProbeReport(
        provider="test",
        model="probe-model",
        runtime_request_captured=True,
        runtime_exit=None,
        skill_loaded=True,
        schema={"has_tools": False},
        channels={
            name: ChannelResult(False, False, False, False, 200, None, "none") for name in CHANNELS
        },
        recommendation="proxy_app_pseudo_tool",
        fallback="local injection",
    )

    class FakeRunner:
        async def run(self, captured: dict[str, Any]) -> ProbeReport:
            assert "private runtime text" in json.dumps(captured)
            return report

    state = ProbeState(FakeRunner(), "runtime-model")  # type: ignore[arg-type]
    app = create_app(state)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://probe.test"
    ) as client:
        models = await client.get("/v1/models")
        response = await client.post("/v1/chat/completions", json=captured_request())
        status = await client.get("/_probe/status")

    assert models.json()["data"][0]["id"] == "runtime-model"
    assert response.status_code == 200
    assert (
        json.loads(response.json()["choices"][0]["message"]["content"])["tool_calls"][0][
            "tool_name"
        ]
        == "answer"
    )
    assert status.json()["report"]["recommendation"] == "proxy_app_pseudo_tool"
    assert "private runtime text" not in status.text
