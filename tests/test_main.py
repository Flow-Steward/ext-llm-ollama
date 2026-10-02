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

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = {
    "provider_secrets": {"api_key": "key-1"},
    "connection_config": {"upstream_base_url": "https://llm.example.com"},
}


def _query(query_id: str, params: dict | None = None) -> dict:
    return {
        "mode": "query",
        "query": {"query_id": query_id, "params": params or {}},
        "runtime_context": {"account_id": "acc-1", "resources": RESOURCES},
    }


def _action(action_id: str, payload: dict | None = None) -> dict:
    return {
        "mode": "action",
        "action": {"action_id": action_id, "input": payload or {}},
        "runtime_context": {"account_id": "acc-1", "resources": RESOURCES},
    }


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
        {"mode": "event"},
        {},
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


@pytest.mark.parametrize("stdin", ["not json", "[]", ""])
def test_the_process_answers_bad_input_with_a_structured_error(stdin) -> None:
    result = _run(stdin)

    assert result.returncode == 2
    _assert_structured_error(json.loads(result.stdout), "invalid_response")


def test_the_process_writes_exactly_one_json_object() -> None:
    result = _run(json.dumps(_query("llm.unknown")))

    assert result.returncode == 2
    assert result.stdout.count("\n") == 1
    assert json.loads(result.stdout)["ok"] is False
