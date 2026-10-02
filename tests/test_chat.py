"""``llm.chat`` and ``llm.chat_structured`` against Ollama's ``/api/chat``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .conftest import chat_reply, connection

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Look a value up.",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string", "const": "probe"}},
            "required": ["key"],
            "additionalProperties": False,
        },
    },
}
SCHEMA = {
    "title": "answer",
    "type": "object",
    "properties": {
        "status": {"type": "string", "const": "ok"},
        "const": {"type": "integer"},  # a property may itself be called "const"
    },
    "required": ["status", "const"],
    "additionalProperties": False,
}


def _chat(runtime, structured: bool = False, **payload):
    request = {
        "connection": connection(api_key="key-1"),
        "input": {"model_id": "gpt-oss:20b", "messages": [], **payload},
    }
    return runtime.chat(request, structured=structured)


def test_a_plain_chat_returns_the_canonical_result(runtime, ollama) -> None:
    ollama.chat_replies = [chat_reply({"content": "basic-ok", "thinking": "short"})]

    result = _chat(
        runtime,
        messages=[{"role": "user", "content": "Return exactly basic-ok."}],
        temperature=0,
        max_tokens=64,
    )

    assert result == json.loads((FIXTURES / "llm.chat.json").read_text())["result"]
    (call,) = ollama.calls
    assert call["url"] == "https://ollama.com/api/chat"
    assert call["headers"]["Authorization"] == "Bearer key-1"
    assert call["body"] == {
        "model": "gpt-oss:20b",
        "messages": [{"role": "user", "content": "Return exactly basic-ok."}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 64},
    }


def test_tool_calls_and_results_translate_both_ways(runtime, ollama) -> None:
    ollama.chat_replies = [
        chat_reply(
            {
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "lookup", "arguments": {"key": "probe"}}},
                    # Some models send arguments as a JSON string.
                    {"function": {"name": "lookup", "arguments": '{"key": "probe"}'}},
                ],
            }
        )
    ]

    result = _chat(
        runtime,
        tools=[PROBE_TOOL],
        messages=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_0", "name": "lookup", "arguments": {"key": "x"}}],
                "continuation": {"thinking": "earlier reasoning"},
            },
            {"role": "tool", "tool_call_id": "call_0", "content": {"value": "probe-ok"}},
        ],
    )

    sent = ollama.calls[0]["body"]
    assert sent["messages"] == [
        {
            "role": "assistant",
            "content": "",
            "thinking": "earlier reasoning",
            "tool_calls": [
                {"type": "function", "function": {"name": "lookup", "arguments": {"key": "x"}}}
            ],
        },
        {"role": "tool", "content": '{"value":"probe-ok"}', "tool_name": "lookup"},
    ]
    # ``const`` is rewritten as ``enum`` so model templates keep the constraint.
    assert sent["tools"][0]["function"]["parameters"]["properties"]["key"] == {
        "type": "string",
        "enum": ["probe"],
    }
    assert result["message"]["tool_calls"] == [
        {"id": "call_1", "name": "lookup", "arguments": {"key": "probe"}},
        {"id": "ollama_call_1", "name": "lookup", "arguments": {"key": "probe"}},
    ]


def test_tool_choice_none_withholds_the_tools(runtime, ollama) -> None:
    ollama.chat_replies = [chat_reply({"content": "final answer"})]

    _chat(runtime, tools=[PROBE_TOOL], tool_choice="none")

    assert "tools" not in ollama.calls[0]["body"]
    assert "tool_choice" not in ollama.calls[0]["body"]  # Ollama has no such field


@pytest.mark.parametrize(
    ("payload", "reply"),
    [
        # "required" but the model answered in text.
        ({"tools": [PROBE_TOOL], "tool_choice": "required"}, {"content": "no tool"}),
        # "none" but the model called a tool anyway.
        (
            {"tools": [PROBE_TOOL], "tool_choice": "none"},
            {"tool_calls": [{"function": {"name": "lookup", "arguments": {}}}]},
        ),
        # One call per turn was asked for, two came back.
        (
            {"tools": [PROBE_TOOL], "parallel_tool_calls": False},
            {
                "tool_calls": [
                    {"function": {"name": "lookup", "arguments": {}}},
                    {"function": {"name": "lookup", "arguments": {}}},
                ]
            },
        ),
    ],
)
def test_tool_choice_is_enforced_on_the_response(runtime, ollama, payload, reply) -> None:
    ollama.chat_replies = [chat_reply({"content": "", **reply})]

    with pytest.raises(RuntimeError, match="capability_mismatch"):
        _chat(runtime, **payload)


@pytest.mark.parametrize(
    "payload",
    [{"tool_choice": "required"}, {"tool_choice": "any"}],
)
def test_an_unsatisfiable_tool_choice_is_refused_before_the_request(
    runtime, ollama, payload
) -> None:
    with pytest.raises(RuntimeError, match="capability_mismatch"):
        _chat(runtime, **payload)
    assert ollama.calls == []


@pytest.mark.parametrize(
    "reply",
    [
        {"done": True},
        {"message": {"content": "ok"}, "done": False},
        {"message": {"content": "ok"}},
        {"message": "ok", "done": True},
    ],
)
def test_an_incomplete_answer_is_invalid(runtime, ollama, reply) -> None:
    ollama.chat_replies = [reply]

    with pytest.raises(RuntimeError, match="invalid_response"):
        _chat(runtime)


@pytest.mark.parametrize(
    "error",
    ["authentication_error", "billing_quota_exceeded", "rate_limited", "timeout"],
)
def test_upstream_failures_propagate_their_category(runtime, ollama, error) -> None:
    ollama.chat_replies = [RuntimeError(error)]

    with pytest.raises(RuntimeError, match=error):
        _chat(runtime)


# --------------------------------------------------------------------------- #
# Structured output                                                            #
# --------------------------------------------------------------------------- #


def test_structured_output_is_one_forced_tool_call(runtime, ollama) -> None:
    ollama.chat_replies = [
        chat_reply(
            {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": runtime.STRUCTURED_OUTPUT_TOOL,
                            "arguments": {"status": "ok", "const": 7},
                        }
                    }
                ],
            }
        )
    ]

    result = _chat(
        runtime,
        structured=True,
        schema=SCHEMA,
        schema_name="answer",
        tools=[PROBE_TOOL],  # the caller's tools are not offered here
        messages=[{"role": "user", "content": "Answer."}],
    )

    assert result == json.loads((FIXTURES / "llm.chat_structured.json").read_text())["result"]
    sent = ollama.calls[0]["body"]
    (tool,) = sent["tools"]
    assert tool["function"]["name"] == runtime.STRUCTURED_OUTPUT_TOOL
    portable = tool["function"]["parameters"]
    assert portable["properties"]["status"] == {"type": "string", "enum": ["ok"]}
    assert portable["properties"]["const"] == {"type": "integer"}  # a name, not a keyword
    # A local Ollama enforces ``format``; ollama.com ignores it, hence the tool.
    assert sent["format"] == portable
    # The schema is also spelled out, since templates may drop constraints.
    instruction = sent["messages"][-1]
    assert instruction["role"] == "user"
    assert json.dumps(SCHEMA, separators=(",", ":")) in instruction["content"]


def test_structured_output_falls_back_to_json_content(runtime, ollama) -> None:
    # A local Ollama that honoured ``format`` may answer in content instead.
    ollama.chat_replies = [chat_reply({"content": '{"status": "ok", "const": 7}'})]

    result = _chat(runtime, structured=True, schema=SCHEMA)

    assert result["structured_output"] == {"status": "ok", "const": 7}


@pytest.mark.parametrize("content", ["", "not json"])
def test_structured_output_without_an_answer_is_a_mismatch(runtime, ollama, content) -> None:
    ollama.chat_replies = [chat_reply({"content": content})]

    with pytest.raises(RuntimeError, match="capability_mismatch"):
        _chat(runtime, structured=True, schema=SCHEMA)


@pytest.mark.parametrize("schema", [None, {}, "object"])
def test_structured_output_needs_a_schema(runtime, ollama, schema) -> None:
    with pytest.raises(RuntimeError, match="capability_mismatch"):
        _chat(runtime, structured=True, schema=schema)
    assert ollama.calls == []


def test_portable_schema_keeps_an_explicit_enum_and_nested_definitions(runtime) -> None:
    schema = {
        "$defs": {"Item": {"type": "object", "properties": {"id": {"const": 7}}}},
        "properties": {
            "mode": {"const": "a", "enum": ["a", "b"]},
            "items": {"type": "array", "items": {"$ref": "#/$defs/Item"}},
        },
        "anyOf": [{"const": 1}],
        "default": {"const": "kept verbatim"},
    }

    assert runtime._portable_schema(schema) == {
        "$defs": {"Item": {"type": "object", "properties": {"id": {"enum": [7]}}}},
        "properties": {
            "mode": {"const": "a", "enum": ["a", "b"]},  # not loosened
            "items": {"type": "array", "items": {"$ref": "#/$defs/Item"}},
        },
        "anyOf": [{"enum": [1]}],
        "default": {"const": "kept verbatim"},
    }
