from __future__ import annotations

import importlib.util
import json
import socket
import threading
import time
from pathlib import Path

import pytest


def _runtime():
    path = Path(__file__).parents[1] / "runtime.py"
    spec = importlib.util.spec_from_file_location("ollama_runtime", path)
    module = importlib.util.module_from_spec(spec)
    assert spec
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_failure_response_retries_discovery_but_not_ambiguous_chat() -> None:
    runtime = _runtime()
    discovery = runtime._failure_response("upstream_error", request_semantics="safe_read")
    chat = runtime._failure_response("upstream_error", request_semantics="mutation")

    assert discovery["retryable"] is True
    assert discovery["definitely_no_external_effect"] is True
    assert discovery["external_effect_status"] == "failed"
    assert chat["retryable"] is False
    assert chat["definitely_no_external_effect"] is False
    assert chat["external_effect_status"] == "timeout_unknown"


def test_list_models_fixture_is_a_complete_snapshot() -> None:
    fixture = json.loads(
        (Path(__file__).parents[1] / "fixtures" / "llm.list_models.json").read_text()
    )
    assert fixture["result"]["snapshot_complete"] is True


def test_chat_action_reuses_completed_result_for_same_idempotency_key() -> None:
    runtime = _runtime()
    calls = 0
    request = {"idempotency_key": "chat:account:model:request", "request": {"model": "m"}}

    def invoke():
        nonlocal calls
        calls += 1
        return {"message": {"role": "assistant", "content": "ok"}}

    first = runtime._execute_idempotent_action(request, "llm.chat", invoke)
    second = runtime._execute_idempotent_action(request, "llm.chat", invoke)

    assert first == second
    assert calls == 1


def test_chat_action_coalesces_concurrent_requests() -> None:
    runtime = _runtime()
    calls = 0
    request = {"idempotency_key": "chat:concurrent", "request": {"model": "m"}}
    started = threading.Event()
    release = threading.Event()
    results: list[dict] = []

    def invoke():
        nonlocal calls
        calls += 1
        started.set()
        release.wait(timeout=1)
        return {"message": {"role": "assistant", "content": "ok"}}

    def run():
        results.append(runtime._execute_idempotent_action(request, "llm.chat", invoke))

    first = threading.Thread(target=run)
    second = threading.Thread(target=run)
    first.start()
    assert started.wait(timeout=1)
    second.start()
    time.sleep(0.02)
    release.set()
    first.join(timeout=1)
    second.join(timeout=1)

    assert calls == 1
    assert len(results) == 2
    assert results[0] == results[1]


def _allow_public_upstream(monkeypatch, runtime) -> None:
    monkeypatch.setattr(
        runtime,
        "_public_https_url",
        lambda _url: ("ollama.example", "203.0.113.8", 443, "/"),
    )


def test_discovery_keeps_digest_for_stale_detection(monkeypatch) -> None:
    runtime = _runtime()
    _allow_public_upstream(monkeypatch, runtime)
    monkeypatch.setattr(
        runtime,
        "_request_json",
        lambda url, **_kwargs: (
            {"models": [{"name": "qwen2.5:7b", "digest": "sha256:abc"}]}
            if url.endswith("/api/tags")
            else {"model_info": {"general.architecture": "qwen2"}}
        ),
    )
    result = runtime.list_models(
        {"connection": {"connection_config": {"upstream_base_url": "https://ollama.example"}}}
    )
    assert result["models"][0]["model_id"] == "qwen2.5:7b"
    assert result["models"][0]["upstream_digest"] == "sha256:abc"
    assert set(result["models"][0]["capabilities"]) == {"chat", "tools", "structured_output"}


def test_discovery_rejects_too_many_models_before_show_requests(monkeypatch) -> None:
    runtime = _runtime()
    _allow_public_upstream(monkeypatch, runtime)
    show_calls = []

    def fake_request(url, **_kwargs):
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {"name": f"model-{index}", "digest": f"sha256:{index}"}
                    for index in range(runtime.MAX_DISCOVERY_MODELS + 1)
                ]
            }
        show_calls.append(url)
        return {"model_info": {}}

    monkeypatch.setattr(runtime, "_request_json", fake_request)

    with pytest.raises(RuntimeError, match="invalid_response"):
        runtime.list_models(
            {"connection": {"connection_config": {"upstream_base_url": "https://ollama.example"}}}
        )
    assert show_calls == []


def test_discovery_bounds_per_model_and_aggregate_metadata(monkeypatch) -> None:
    runtime = _runtime()
    _allow_public_upstream(monkeypatch, runtime)
    monkeypatch.setattr(runtime, "MAX_MODEL_INFO_BYTES", 80)
    monkeypatch.setattr(runtime, "MAX_DISCOVERY_RESULT_BYTES", 1_000)
    monkeypatch.setattr(
        runtime,
        "_request_json",
        lambda url, **_kwargs: (
            {
                "models": [
                    {"name": "model-a", "digest": "sha256:a"},
                    {"name": "model-b", "digest": "sha256:b"},
                ]
            }
            if url.endswith("/api/tags")
            else {"model_info": {"oversized": "x" * 1_000}}
        ),
    )

    result = runtime.list_models(
        {"connection": {"connection_config": {"upstream_base_url": "https://ollama.example"}}}
    )

    assert all(model["metadata"]["model_info"] == {} for model in result["models"])
    assert len(json.dumps(result).encode()) <= runtime.MAX_DISCOVERY_RESULT_BYTES


def test_discovery_paginates_before_show_requests(monkeypatch) -> None:
    runtime = _runtime()
    _allow_public_upstream(monkeypatch, runtime)
    shown = []

    def fake_request(url, **kwargs):
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {"name": f"model-{index}", "digest": f"sha256:{index}"} for index in range(30)
                ]
            }
        shown.append(kwargs["body"]["model"])
        return {"model_info": {}}

    monkeypatch.setattr(runtime, "_request_json", fake_request)
    first = runtime.list_models(
        {
            "connection": {"connection_config": {"upstream_base_url": "https://ollama.example"}},
            "query": {"cursor": "", "limit": 10},
        }
    )
    shown.clear()
    result = runtime.list_models(
        {
            "connection": {"connection_config": {"upstream_base_url": "https://ollama.example"}},
            "query": {"cursor": first["next_cursor"], "limit": 10},
        }
    )

    assert shown == [f"model-{index}" for index in range(10, 20)]
    assert len(result["models"]) == 10
    assert result["next_cursor"].endswith(":20")


def test_discovery_pages_share_one_immutable_tags_snapshot(monkeypatch) -> None:
    runtime = _runtime()
    _allow_public_upstream(monkeypatch, runtime)
    tags_calls = 0
    shown = []

    def fake_request(url, **kwargs):
        nonlocal tags_calls
        if url.endswith("/api/tags"):
            tags_calls += 1
            start = 0 if tags_calls == 1 else 1
            return {
                "models": [
                    {"name": f"model-{index}", "digest": f"sha256:{index}"}
                    for index in range(start, 30)
                ]
            }
        shown.append(kwargs["body"]["model"])
        return {"model_info": {}}

    monkeypatch.setattr(runtime, "_request_json", fake_request)
    request = {
        "connection": {"connection_config": {"upstream_base_url": "https://ollama.example"}},
        "query": {"cursor": "", "limit": 10},
    }

    first = runtime.list_models(request)
    second = runtime.list_models(
        {**request, "query": {"cursor": first["next_cursor"], "limit": 10}}
    )

    assert tags_calls == 1
    assert shown == [f"model-{index}" for index in range(20)]
    assert [model["model_id"] for model in second["models"]] == [
        f"model-{index}" for index in range(10, 20)
    ]


def test_discovery_rejects_new_session_instead_of_evicting_active_cursor(monkeypatch) -> None:
    runtime = _runtime()
    _allow_public_upstream(monkeypatch, runtime)
    monkeypatch.setattr(runtime, "MAX_DISCOVERY_SNAPSHOTS", 4)

    def fake_request(url, **_kwargs):
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {"name": "model-1", "digest": "sha256:1"},
                    {"name": "model-2", "digest": "sha256:2"},
                ]
            }
        return {"model_info": {}}

    monkeypatch.setattr(runtime, "_request_json", fake_request)
    request = {
        "connection": {"connection_config": {"upstream_base_url": "https://ollama.example"}},
        "query": {"cursor": "", "limit": 1},
    }

    cursors = [runtime.list_models(request)["next_cursor"] for _ in range(4)]
    with pytest.raises(RuntimeError, match="rate_limited"):
        runtime.list_models(request)

    resumed = runtime.list_models({**request, "query": {"cursor": cursors[0], "limit": 1}})
    assert [model["model_id"] for model in resumed["models"]] == ["model-2"]


def test_reference_runtime_binds_loopback_by_default(monkeypatch) -> None:
    runtime = _runtime()
    monkeypatch.delenv("FS_EXTENSION_BIND_HOST", raising=False)
    monkeypatch.setenv("FS_EXTENSION_PORT", "8099")

    assert runtime._server_address() == ("127.0.0.1", 8099)


def test_discovery_stops_before_show_after_total_deadline(monkeypatch) -> None:
    runtime = _runtime()
    _allow_public_upstream(monkeypatch, runtime)
    ticks = iter([0.0, 46.0])
    monkeypatch.setattr(runtime.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(
        runtime,
        "_request_json",
        lambda url, **_kwargs: (
            {"models": [{"name": "model-a", "digest": "sha256:a"}]}
            if url.endswith("/api/tags")
            else (_ for _ in ()).throw(AssertionError("show must not start after deadline"))
        ),
    )

    with pytest.raises(RuntimeError, match="timeout"):
        runtime.list_models(
            {"connection": {"connection_config": {"upstream_base_url": "https://ollama.example"}}}
        )


def test_structured_chat_sends_canonical_schema(monkeypatch) -> None:
    runtime = _runtime()
    _allow_public_upstream(monkeypatch, runtime)
    captured = {}

    def fake_request(_url, **kwargs):
        captured["body"] = kwargs["body"]
        return {
            "message": {"content": '{"status":"ok"}'},
            "done": True,
            "done_reason": "stop",
        }

    monkeypatch.setattr(
        runtime,
        "_request_json",
        fake_request,
    )
    runtime.chat(
        {
            "connection": {"connection_config": {"upstream_base_url": "https://ollama.example"}},
            "input": {
                "model_id": "qwen2.5:7b",
                "messages": [],
                "schema": {"type": "object", "additionalProperties": False},
            },
        },
        structured=True,
    )
    assert captured["body"]["format"]["additionalProperties"] is False


def test_chat_translates_canonical_tool_messages_and_assigns_stable_ids(monkeypatch) -> None:
    runtime = _runtime()
    _allow_public_upstream(monkeypatch, runtime)
    captured = {}

    def fake_request(_url, **kwargs):
        captured["body"] = kwargs["body"]
        return {
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "flow_steward_capability_probe",
                            "arguments": {"key": "probe"},
                        }
                    }
                ],
            },
            "done": True,
            "done_reason": "tool_calls",
        }

    monkeypatch.setattr(runtime, "_request_json", fake_request)
    result = runtime.chat(
        {
            "connection": {"connection_config": {"upstream_base_url": "https://ollama.example"}},
            "input": {
                "model_id": "qwen2.5:7b",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "flow_steward_capability_probe",
                            "description": "Return the probe value.",
                            "parameters": {
                                "type": "object",
                                "properties": {"key": {"type": "string"}},
                                "required": ["key"],
                                "additionalProperties": False,
                            },
                        },
                    }
                ],
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"id": "call_previous", "name": "lookup", "arguments": {"q": "x"}}
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_previous", "content": {"value": "ok"}},
                ],
            },
        }
    )

    assert captured["body"]["messages"][0]["tool_calls"] == [
        {"type": "function", "function": {"name": "lookup", "arguments": {"q": "x"}}}
    ]
    assert captured["body"]["messages"][1] == {
        "role": "tool",
        "tool_name": "lookup",
        "content": '{"value":"ok"}',
    }
    assert captured["body"]["tools"][0]["function"]["parameters"]["required"] == ["key"]
    assert result["message"]["tool_calls"] == [
        {
            "id": "ollama_call_0",
            "name": "flow_steward_capability_probe",
            "arguments": {"key": "probe"},
        }
    ]


def test_chat_enforces_tool_choice_none_without_sending_unsupported_field(monkeypatch) -> None:
    runtime = _runtime()
    _allow_public_upstream(monkeypatch, runtime)
    captured = {}

    def fake_request(_url, **kwargs):
        captured["body"] = kwargs["body"]
        return {
            "message": {"content": "final answer"},
            "done": True,
            "done_reason": "stop",
        }

    monkeypatch.setattr(runtime, "_request_json", fake_request)
    runtime.chat(
        {
            "connection": {"connection_config": {"upstream_base_url": "https://ollama.example"}},
            "input": {
                "model_id": "qwen2.5:7b",
                "messages": [],
                "tools": [{"type": "function", "function": {"name": "probe"}}],
                "tool_choice": "none",
            },
        }
    )

    assert "tool_choice" not in captured["body"]
    assert "tools" not in captured["body"]


def test_chat_enforces_tool_choice_required_by_response_outcome(monkeypatch) -> None:
    runtime = _runtime()
    _allow_public_upstream(monkeypatch, runtime)
    captured = {}

    def fake_request(_url, **kwargs):
        captured["body"] = kwargs["body"]
        return {
            "message": {"content": "ignored the required tool"},
            "done": True,
            "done_reason": "stop",
        }

    monkeypatch.setattr(runtime, "_request_json", fake_request)
    with pytest.raises(RuntimeError, match="capability_mismatch"):
        runtime.chat(
            {
                "connection": {
                    "connection_config": {"upstream_base_url": "https://ollama.example"}
                },
                "input": {
                    "model_id": "qwen2.5:7b",
                    "messages": [],
                    "tools": [{"type": "function", "function": {"name": "probe"}}],
                    "tool_choice": "required",
                },
            }
        )

    assert "tool_choice" not in captured["body"]
    assert captured["body"]["tools"]


@pytest.mark.parametrize(
    "response",
    [
        {"done": True},
        {"message": {"content": "ok"}, "done": False},
        {"message": {"content": "ok"}},
    ],
)
def test_chat_rejects_incomplete_non_streaming_response(monkeypatch, response) -> None:
    runtime = _runtime()
    _allow_public_upstream(monkeypatch, runtime)
    monkeypatch.setattr(runtime, "_request_json", lambda *_args, **_kwargs: response)

    with pytest.raises(RuntimeError, match="invalid_response"):
        runtime.chat(
            {
                "connection": {
                    "connection_config": {"upstream_base_url": "https://ollama.example"}
                },
                "input": {"model_id": "qwen2.5:7b", "messages": []},
            }
        )


def test_request_rejects_redirect_status(monkeypatch) -> None:
    runtime = _runtime()
    monkeypatch.setattr(
        runtime, "_public_https_url", lambda _url: ("example.test", "1.2.3.4", 443, "/")
    )

    class _Response:
        status = 302

        def read(self, _size=-1):
            return b"{}"

    class _Connection:
        def __init__(self, *_args, **_kwargs):
            self._create_connection = None

        def request(self, *_args, **_kwargs):
            return None

        def getresponse(self):
            return _Response()

        def close(self):
            return None

    monkeypatch.setattr(runtime.http.client, "HTTPSConnection", _Connection)

    with pytest.raises(RuntimeError, match="upstream_error"):
        runtime._request_json("https://example.test", method="GET", headers={})


def test_private_or_local_upstream_urls_are_rejected() -> None:
    runtime = _runtime()
    for value in ("http://public.example", "https://localhost", "https://host.local"):
        try:
            runtime._public_https_url(value)
        except RuntimeError as exc:
            assert str(exc) == "model_unavailable"
        else:  # pragma: no cover - failure diagnostics
            raise AssertionError("unsafe upstream URL was accepted")


def test_private_dns_answer_is_rejected(monkeypatch) -> None:
    runtime = _runtime()
    monkeypatch.setattr(
        runtime.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))],
    )
    try:
        runtime._public_https_url("https://ollama.example")
    except RuntimeError as exc:
        assert str(exc) == "model_unavailable"
    else:  # pragma: no cover - failure diagnostics
        raise AssertionError("private DNS answer was accepted")


def test_cgnat_dns_answer_is_rejected(monkeypatch) -> None:
    runtime = _runtime()
    monkeypatch.setattr(
        runtime.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("100.64.0.1", 443))
        ],
    )
    try:
        runtime._public_https_url("https://ollama.example")
    except RuntimeError as exc:
        assert str(exc) == "model_unavailable"
    else:  # pragma: no cover - failure diagnostics
        raise AssertionError("CGNAT DNS answer was accepted")


@pytest.mark.parametrize(
    ("address", "family"),
    (
        ("224.0.0.1", socket.AF_INET),
        ("5f00::1", socket.AF_INET6),
        ("fec0::1", socket.AF_INET6),
        ("ff02::1", socket.AF_INET6),
    ),
)
def test_special_use_dns_answer_is_rejected(monkeypatch, address, family) -> None:
    runtime = _runtime()
    monkeypatch.setattr(
        runtime.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(family, socket.SOCK_STREAM, 6, "", (address, 443))],
    )

    with pytest.raises(RuntimeError, match="model_unavailable"):
        runtime._public_https_url("https://ollama.example")


@pytest.mark.parametrize(
    ("unsafe_address", "family"),
    (
        ("127.0.0.1", socket.AF_INET),
        ("224.0.0.1", socket.AF_INET),
        ("5f00::1", socket.AF_INET6),
        ("fec0::1", socket.AF_INET6),
        ("ff02::1", socket.AF_INET6),
    ),
)
def test_mixed_public_and_non_public_dns_answers_are_rejected(
    monkeypatch, unsafe_address, family
) -> None:
    runtime = _runtime()
    monkeypatch.setattr(
        runtime.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443)),
            (family, socket.SOCK_STREAM, 6, "", (unsafe_address, 443)),
        ],
    )

    with pytest.raises(RuntimeError, match="model_unavailable"):
        runtime._public_https_url("https://ollama.example")


def test_public_ipv4_and_ipv6_dns_answers_are_accepted(monkeypatch) -> None:
    runtime = _runtime()
    monkeypatch.setattr(
        runtime.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443)),
            (
                socket.AF_INET6,
                socket.SOCK_STREAM,
                6,
                "",
                ("2606:4700:4700::1111", 443),
            ),
        ],
    )

    target = runtime._public_https_url("https://ollama.example/api/tags")

    assert target == ("ollama.example", "8.8.8.8", 443, "/api/tags")


def test_request_errors_are_normalized_to_safe_categories(monkeypatch) -> None:
    runtime = _runtime()

    monkeypatch.setattr(
        runtime, "_public_https_url", lambda _url: ("example.test", "1.2.3.4", 443, "/")
    )

    class _Connection:
        def __init__(self, *_args, **_kwargs):
            self._create_connection = None

        def request(self, *_args, **_kwargs):
            raise runtime.http.client.HTTPException("401")

        def close(self):
            return None

    monkeypatch.setattr(runtime.http.client, "HTTPSConnection", _Connection)
    try:
        runtime._request_json("https://example.test", method="GET", headers={})
    except RuntimeError as exc:
        assert str(exc) == "authentication_error"
    else:  # pragma: no cover - failure diagnostics
        raise AssertionError("HTTP error was not raised")


def test_https_request_preserves_nondefault_port_in_host_header(monkeypatch) -> None:
    runtime = _runtime()
    captured = {}
    monkeypatch.setattr(
        runtime, "_public_https_url", lambda _url: ("ollama.example", "8.8.8.8", 8443, "/")
    )

    class _Response:
        status = 200

        def read(self, _size=-1):
            return b"{}"

    class _Connection:
        def __init__(self, *_args, **_kwargs):
            self._create_connection = None

        def request(self, _method, _path, **kwargs):
            captured.update(kwargs)

        def getresponse(self):
            return _Response()

        def close(self):
            return None

    monkeypatch.setattr(runtime.http.client, "HTTPSConnection", _Connection)
    runtime._request_json("https://ollama.example:8443", method="GET", headers={})
    assert captured["headers"]["Host"] == "ollama.example:8443"


def test_request_and_upstream_bodies_are_bounded(monkeypatch) -> None:
    import io

    runtime = _runtime()
    oversized = b"x" * (runtime.MAX_RUNTIME_BODY_BYTES + 1)

    try:
        runtime._read_request_body(io.BytesIO(oversized), len(oversized))
    except RuntimeError as exc:
        assert str(exc) == "invalid_response"
    else:  # pragma: no cover - failure diagnostics
        raise AssertionError("oversized runtime request was accepted")

    monkeypatch.setattr(
        runtime, "_public_https_url", lambda _url: ("example.test", "1.2.3.4", 443, "/")
    )

    class _Response:
        status = 200

        def read(self, _size=-1):
            return oversized

    class _Connection:
        def __init__(self, *_args, **_kwargs):
            self._create_connection = None

        def request(self, *_args, **_kwargs):
            return None

        def getresponse(self):
            return _Response()

        def close(self):
            return None

    monkeypatch.setattr(runtime.http.client, "HTTPSConnection", _Connection)
    try:
        runtime._request_json("https://example.test", method="GET", headers={})
    except RuntimeError as exc:
        assert str(exc) == "invalid_response"
    else:  # pragma: no cover - failure diagnostics
        raise AssertionError("oversized upstream response was accepted")
