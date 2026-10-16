"""The subprocess contract with the Flow Steward host (``main.py``).

The host writes one JSON request to stdin and reads one JSON object from
stdout. Exit 0 with ``ok: true`` is success; exit 2 with the structured error
envelope (``error_code`` / ``error`` / ``errors``) is a provider failure that
does not count against the extension's circuit breaker.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from .conftest import chat_reply

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = {
    "provider_secrets": {"api_key": "key-1"},
    "connection_config": {"upstream_base_url": "https://llm.example.com"},
}
# The final extension_host_v2 execution scope: all four keys, at most one owner.
ROOT_STEP_SCOPE = {
    "group_id": None,
    "for_each_id": None,
    "step_id": "ask_model",
    "step_execution_id": "7c1d2e3f-4a5b-4c6d-8e9f-0a1b2c3d0001",
}
GROUP_STEP_SCOPE = {**ROOT_STEP_SCOPE, "group_id": "draft_answers"}
FOR_EACH_STEP_SCOPE = {
    **ROOT_STEP_SCOPE,
    "for_each_id": "each_ticket",
    "step_execution_id": "7c1d2e3f-4a5b-4c6d-8e9f-0a1b2c3d0002",
}
# Provider settings and model discovery run outside any workflow: all four null.
WORKFLOW_SCOPE = dict.fromkeys(ROOT_STEP_SCOPE)


def _envelope(body: dict, scope: dict | None = None, **context) -> dict:
    return {
        "contract_version": "extension_host_v2",
        **body,
        "runtime_context": {
            "account_id": "acc-1",
            "resources": RESOURCES,
            **(WORKFLOW_SCOPE if scope is None else scope),
            **context,
        },
    }


def _query(query_id: str, params: dict | None = None, scope: dict | None = None) -> dict:
    return _envelope(
        {"mode": "query", "query": {"query_id": query_id, "params": params or {}}},
        scope,
        network_policy={"allow_private_addresses": True},
    )


def _action(action_id: str, payload: dict | None = None, scope: dict | None = None) -> dict:
    return _envelope(
        {"mode": "action", "action": {"action_id": action_id, "input": payload or {}}}, scope
    )


def _assert_structured_error(response: dict, code: str) -> None:
    # The exact shape the host recognises as a domain error, not a crash.
    assert response["ok"] is False
    assert response["error_code"] == response["errors"][0]["code"] == code
    assert response["error"] == response["errors"][0]["message"]


@pytest.fixture
def recorded(monkeypatch, main):
    calls: list[tuple[str, dict, bool]] = []

    def fake(name):
        def handler(request, *, structured=False):
            calls.append((name, request, structured))
            return {"handled_by": name}

        return handler

    monkeypatch.setitem(main.QUERIES, "llm.list_models", fake("list_models"))
    monkeypatch.setattr(main.runtime, "chat", fake("chat"))
    return calls


def test_a_query_gets_its_params_and_the_provider_settings(main, recorded) -> None:
    response, code = main.handle(_query("llm.list_models", {"cursor": "", "limit": 200}))

    assert (response, code) == (
        {"ok": True, "result": {"handled_by": "list_models"}, "errors": []},
        0,
    )
    (name, request, _) = recorded[0]
    assert request == {
        "connection": {
            "connection_config": {"upstream_base_url": "https://llm.example.com"},
            "credentials": {"api_key": "key-1"},
        },
        "network_policy": {"allow_private_addresses": True},
        "query": {"cursor": "", "limit": 200},
    }


@pytest.mark.parametrize(
    ("action_id", "structured"),
    [("llm.chat", False), ("llm.chat_structured", True)],
)
def test_chat_actions_route_to_chat(main, recorded, action_id, structured) -> None:
    response, code = main.handle(_action(action_id, {"model_id": "m"}))

    assert code == 0
    assert response["result"] == {"handled_by": "chat"}
    (_, request, was_structured) = recorded[0]
    assert request["input"] == {"model_id": "m"}
    assert was_structured is structured


@pytest.mark.parametrize(
    "payload",
    [
        _query("llm.unknown"),
        _action("llm.unknown"),
        _envelope({"mode": "event"}),
        _envelope({}),
    ],
)
def test_an_unknown_operation_is_a_structured_error(main, recorded, payload) -> None:
    response, code = main.handle(payload)

    assert code == 2
    _assert_structured_error(response, "invalid_response")
    assert recorded == []


@pytest.mark.parametrize(
    ("raised", "code"),
    [
        (RuntimeError("rate_limited"), "rate_limited"),
        (RuntimeError("timeout: after 60s"), "timeout"),
        # Anything unexpected leaves as a generic category, never as its text.
        (KeyError("key-1 leaked?"), "upstream_error"),
        (ValueError("Bearer key-1"), "upstream_error"),
    ],
)
def test_handler_failures_leave_as_one_safe_category(monkeypatch, main, raised, code) -> None:
    def fail(_request):
        raise raised

    monkeypatch.setitem(main.QUERIES, "llm.list_models", fail)

    response, exit_code = main.handle(_query("llm.list_models"))

    assert exit_code == 2
    _assert_structured_error(response, code)
    assert "key-1" not in json.dumps(response)


def test_discovery_failures_are_retryable_but_chat_failures_are_not(main, runtime) -> None:
    read = runtime._failure_response("upstream_error", request_semantics="safe_read")
    write = runtime._failure_response("upstream_error", request_semantics="mutation")
    refused = runtime._failure_response("billing_quota_exceeded", request_semantics="mutation")

    assert (read["retryable"], read["definitely_no_external_effect"]) == (True, True)
    # A failed chat may already have been billed upstream: never retry blindly.
    assert (write["retryable"], write["external_effect_status"]) == (False, "timeout_unknown")
    assert (refused["retryable"], refused["definitely_no_external_effect"]) == (False, True)


def _run(stdin: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "main.py"],
        cwd=ROOT,
        input=stdin,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize(
    ("stdin", "code"),
    [
        ("not json", "invalid_response"),
        ("[]", "invalid_response"),
        # An empty request is an object without the host contract.
        ("", "invalid_host_envelope"),
    ],
)
def test_the_process_answers_bad_input_with_a_structured_error(stdin, code) -> None:
    result = _run(stdin)

    assert result.returncode == 2
    _assert_structured_error(json.loads(result.stdout), code)


@pytest.mark.parametrize(
    "scope", [ROOT_STEP_SCOPE, GROUP_STEP_SCOPE, FOR_EACH_STEP_SCOPE, WORKFLOW_SCOPE]
)
def test_every_final_scope_owner_lists_models_and_chats(main, ollama, scope) -> None:
    ollama.tags = [{"name": "gpt-oss:20b", "digest": "sha256:a"}]
    ollama.show = {"gpt-oss:20b": {"capabilities": ["completion", "tools"]}}
    ollama.chat_replies = [chat_reply({"content": "scoped-ok"})]

    listed, list_code = main.handle(_query("llm.list_models", {"limit": 25}, scope))
    chatted, chat_code = main.handle(
        _action(
            "llm.chat",
            {"model_id": "gpt-oss:20b", "messages": [{"role": "user", "content": "hi"}]},
            scope,
        )
    )

    assert (list_code, chat_code) == (0, 0)
    assert [model["model_id"] for model in listed["result"]["models"]] == ["gpt-oss:20b"]
    assert chatted["ok"] is True
    assert len(ollama.calls_to("/api/chat")) == 1
    assert all(call["url"].startswith("https://llm.example.com/") for call in ollama.calls)


@pytest.mark.parametrize(
    ("contract_version", "scope", "reason"),
    [
        ("extension_host_v1", FOR_EACH_STEP_SCOPE, "Unsupported host contract"),
        ("", FOR_EACH_STEP_SCOPE, "Unsupported host contract"),
        (
            "extension_host_v2",
            {k: v for k, v in FOR_EACH_STEP_SCOPE.items() if k != "for_each_id"},
            "misses execution scope",
        ),
        (
            "extension_host_v2",
            {**FOR_EACH_STEP_SCOPE, "group_id": "draft_answers"},
            "both a Group and a For each",
        ),
        (
            "extension_host_v2",
            {**ROOT_STEP_SCOPE, "step_execution_id": None},
            "without its logical execution",
        ),
    ],
)
@pytest.mark.parametrize("mode", ["query", "action"])
def test_foreign_or_incomplete_envelope_is_refused_before_any_request(
    main, ollama, mode, contract_version, scope, reason
) -> None:
    ollama.chat_replies = [chat_reply({"content": "must not be asked"})]
    if mode == "query":
        payload = _query("llm.list_models", {"limit": 25})
    else:
        payload = _action("llm.chat", {"model_id": "gpt-oss:20b", "messages": []})
    payload["contract_version"] = contract_version
    context = payload["runtime_context"]
    for key in ROOT_STEP_SCOPE:
        context.pop(key)
    context.update(scope)

    response, exit_code = main.handle(payload)

    assert exit_code == 2
    _assert_structured_error(response, "invalid_host_envelope")
    assert reason in response["error"]
    assert response["definitely_no_external_effect"] is True
    assert response["retryable"] is False
    assert ollama.calls == []


def test_the_process_writes_exactly_one_json_object() -> None:
    result = _run(json.dumps(_query("llm.unknown")))

    assert result.returncode == 2
    assert result.stdout.count("\n") == 1
    assert json.loads(result.stdout)["ok"] is False
