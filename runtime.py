"""Runnable remote HTTPS runtime for a public Ollama endpoint reference bundle."""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import os
import secrets
import socket
import ssl
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

MAX_RUNTIME_BODY_BYTES = 5 * 1024 * 1024
MAX_DISCOVERY_RESULT_BYTES = 4 * 1024 * 1024
MAX_DISCOVERY_MODELS = 256
MAX_MODEL_INFO_BYTES = 32 * 1024
MAX_MODEL_ID_BYTES = 1_024
MAX_SHOW_MODELS_PER_PAGE = 25
MAX_DISCOVERY_SECONDS = 45
MAX_DISCOVERY_SNAPSHOTS = 4
DISCOVERY_SNAPSHOT_TTL_SECONDS = 90
MAX_ACTION_IDEMPOTENCY_ENTRIES = 256
ACTION_IDEMPOTENCY_TTL_SECONDS = 300
ACTION_IDEMPOTENCY_WAIT_SECONDS = 65
_discovery_snapshots: dict[str, tuple[float, list[Any]]] = {}
_discovery_snapshots_lock = threading.Lock()
_action_idempotency: dict[str, dict[str, Any]] = {}
_action_idempotency_lock = threading.Lock()


def _execute_idempotent_action(
    request: dict[str, Any],
    action_id: str,
    operation: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    key = str(request.get("idempotency_key") or "").strip()
    if not key or len(key) > 512:
        raise RuntimeError("invalid_response")
    cache_key = hashlib.sha256(key.encode()).hexdigest()
    try:
        canonical = json.dumps(
            {"action": action_id, "request": request},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise RuntimeError("invalid_response") from exc
    fingerprint = hashlib.sha256(canonical).hexdigest()
    now = time.monotonic()
    with _action_idempotency_lock:
        for expired_key in [
            candidate
            for candidate, entry in _action_idempotency.items()
            if entry["deadline"] <= now and entry["event"].is_set()
        ]:
            _action_idempotency.pop(expired_key, None)
        entry = _action_idempotency.get(cache_key)
        if entry is not None:
            if entry["fingerprint"] != fingerprint:
                raise RuntimeError("invalid_response")
            owner = False
        else:
            if len(_action_idempotency) >= MAX_ACTION_IDEMPOTENCY_ENTRIES:
                raise RuntimeError("rate_limited")
            entry = {
                "deadline": now + ACTION_IDEMPOTENCY_TTL_SECONDS,
                "event": threading.Event(),
                "fingerprint": fingerprint,
                "result": None,
                "error": "",
            }
            _action_idempotency[cache_key] = entry
            owner = True
    if owner:
        try:
            entry["result"] = operation()
        except Exception as exc:
            entry["error"] = str(exc).split(":", 1)[0] or "upstream_error"
        finally:
            entry["event"].set()
    elif not entry["event"].wait(timeout=ACTION_IDEMPOTENCY_WAIT_SECONDS):
        raise RuntimeError("timeout")
    if entry["error"]:
        raise RuntimeError(entry["error"])
    result = entry["result"]
    if not isinstance(result, dict):
        raise RuntimeError("invalid_response")
    return result


def _read_bounded(stream: Any) -> bytes:
    raw = stream.read(MAX_RUNTIME_BODY_BYTES + 1)
    if len(raw) > MAX_RUNTIME_BODY_BYTES:
        raise RuntimeError("invalid_response")
    return raw


def _read_request_body(stream: Any, content_length: int) -> bytes:
    if content_length < 0 or content_length > MAX_RUNTIME_BODY_BYTES:
        raise RuntimeError("invalid_response")
    raw = stream.read(content_length)
    if len(raw) != content_length:
        raise RuntimeError("invalid_response")
    return raw


def _public_https_url(raw_url: str) -> tuple[str, str, int, str]:
    """Validate an upstream URL and return a DNS-pinned HTTPS request target."""
    parsed = urlsplit(str(raw_url or ""))
    hostname = (parsed.hostname or "").rstrip(".").lower()
    if (
        parsed.scheme != "https"
        or not hostname
        or parsed.username
        or parsed.password
        or hostname == "localhost"
        or hostname.endswith(".local")
    ):
        raise RuntimeError("model_unavailable")
    port = parsed.port or 443
    try:
        addresses = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise RuntimeError("upstream_error") from exc
    public_ips: list[str] = []
    for _family, _socktype, _proto, _canonname, address in addresses:
        try:
            candidate = ipaddress.ip_address(address[0])
        except ValueError as exc:
            raise RuntimeError("model_unavailable") from exc
        if (
            not candidate.is_global
            or candidate.is_private
            or candidate.is_reserved
            or candidate.is_link_local
            or candidate.is_loopback
            or candidate.is_multicast
            or candidate.is_unspecified
            or getattr(candidate, "is_site_local", False)
        ):
            raise RuntimeError("model_unavailable")
        public_ips.append(str(candidate))
    if not public_ips:
        raise RuntimeError("model_unavailable")
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    return hostname, public_ips[0], port, path


def _request_json(
    url: str,
    *,
    method: str,
    headers: dict[str, str],
    body: Any = None,
    timeout_seconds: int | float = 60,
) -> dict[str, Any]:
    hostname, pinned_ip, port, path = _public_https_url(url)
    data = json.dumps(body).encode() if body is not None else None
    connection = http.client.HTTPSConnection(
        hostname, port, timeout=timeout_seconds, context=ssl.create_default_context()
    )
    # Keep the original hostname for TLS SNI/certificate verification while
    # pinning this request's TCP connection to the public address just checked.
    connection._create_connection = lambda _address, timeout, source_address=None: (
        socket.create_connection(  # type: ignore[attr-defined]
            (pinned_ip, port), timeout, source_address
        )
    )
    try:
        host_header = f"[{hostname}]" if ":" in hostname else hostname
        if port != 443:
            host_header = f"{host_header}:{port}"
        connection.request(method, path, body=data, headers={**headers, "Host": host_header})
        response = connection.getresponse()
        raw = _read_bounded(response)
        if response.status < 200 or response.status >= 300:
            raise http.client.HTTPException(str(response.status))
        value = json.loads(raw.decode())
    except http.client.HTTPException as exc:
        status = int(str(exc)) if str(exc).isdigit() else 0
        category = {
            401: "authentication_error",
            403: "authentication_error",
            404: "model_unavailable",
            408: "timeout",
            429: "rate_limited",
        }.get(status, "upstream_error")
        raise RuntimeError(category) from exc
    except OSError as exc:
        raise RuntimeError(
            "timeout" if "timed out" in str(exc).lower() else "upstream_error"
        ) from exc
    except ValueError as exc:
        raise RuntimeError("invalid_response") from exc
    finally:
        connection.close()
    if not isinstance(value, dict):
        raise RuntimeError("invalid_response")
    return value


def _failure_response(code: str, *, request_semantics: str) -> dict[str, Any]:
    """Expose only provider-neutral retry facts to the host orchestrator."""
    safe_code = str(code).split(":", 1)[0]
    retryable_code = safe_code in {"rate_limited", "timeout", "upstream_error"}
    definite_refusal = safe_code in {
        "authentication_error",
        "model_unavailable",
        "rate_limited",
    }
    definitely_no_effect = request_semantics == "safe_read" or definite_refusal
    return {
        "ok": False,
        "errors": [{"code": safe_code}],
        "failure_class": "transient" if safe_code in {"timeout", "upstream_error"} else "provider",
        "retryable": retryable_code and definitely_no_effect,
        "definitely_no_external_effect": definitely_no_effect,
        "external_effect_status": "failed" if definitely_no_effect else "timeout_unknown",
        "provider_error_code": safe_code,
    }


def _connection(request: dict[str, Any]) -> tuple[str, dict[str, str]]:
    connection = request.get("connection") if isinstance(request, dict) else {}
    config = connection.get("connection_config") if isinstance(connection, dict) else {}
    credentials = connection.get("credentials") if isinstance(connection, dict) else {}
    base_url = str((config or {}).get("upstream_base_url") or "").rstrip("/")
    if not base_url:
        raise RuntimeError("model_unavailable")
    _public_https_url(base_url)
    headers = {"Content-Type": "application/json"}
    token = str(
        (credentials or {}).get("bearer_token") or (credentials or {}).get("api_key") or ""
    ).strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return base_url, headers


def _pagination(request: dict[str, Any]) -> tuple[str, int, int]:
    query = request.get("query") if isinstance(request, dict) else {}
    query = query if isinstance(query, dict) else {}
    cursor = str(query.get("cursor") or "").strip()
    token = ""
    offset = 0
    if cursor:
        token, separator, raw_offset = cursor.partition(":")
        if not separator or not token or not raw_offset.isdigit():
            raise RuntimeError("invalid_response")
        offset = int(raw_offset)
    try:
        requested_limit = int(query.get("limit") or MAX_SHOW_MODELS_PER_PAGE)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("invalid_response") from exc
    if offset < 0 or requested_limit < 1:
        raise RuntimeError("invalid_response")
    return token, offset, min(requested_limit, MAX_SHOW_MODELS_PER_PAGE)


def _new_discovery_snapshot(tags: list[Any]) -> str:
    now = time.monotonic()
    token = secrets.token_urlsafe(24)
    with _discovery_snapshots_lock:
        expired = [key for key, (deadline, _) in _discovery_snapshots.items() if deadline <= now]
        for key in expired:
            _discovery_snapshots.pop(key, None)
        if len(_discovery_snapshots) >= MAX_DISCOVERY_SNAPSHOTS:
            raise RuntimeError("rate_limited")
        _discovery_snapshots[token] = (now + DISCOVERY_SNAPSHOT_TTL_SECONDS, list(tags))
    return token


def _read_discovery_snapshot(token: str) -> list[Any]:
    now = time.monotonic()
    with _discovery_snapshots_lock:
        entry = _discovery_snapshots.get(token)
        if entry is None or entry[0] <= now:
            _discovery_snapshots.pop(token, None)
            raise RuntimeError("invalid_response")
        return entry[1]


def _drop_discovery_snapshot(token: str) -> None:
    with _discovery_snapshots_lock:
        _discovery_snapshots.pop(token, None)


def _json_content(value: Any) -> Any:
    return value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))


def _ollama_messages(messages: Any) -> list[dict[str, Any]]:
    """Translate canonical tool calls/results to Ollama's /api/chat message shape."""
    call_names: dict[str, str] = {}
    result: list[dict[str, Any]] = []
    for raw in messages if isinstance(messages, list) else []:
        if not isinstance(raw, dict):
            continue
        role = str(raw.get("role") or "user")
        item: dict[str, Any] = {"role": role, "content": _json_content(raw.get("content", ""))}
        if raw.get("name"):
            item["name"] = str(raw["name"])
        continuation = raw.get("continuation")
        if isinstance(continuation, dict) and "thinking" in continuation:
            item["thinking"] = continuation["thinking"]
        calls: list[dict[str, Any]] = []
        for call in raw.get("tool_calls") if isinstance(raw.get("tool_calls"), list) else []:
            if not isinstance(call, dict):
                continue
            call_id, name = str(call.get("id") or "").strip(), str(call.get("name") or "").strip()
            arguments = call.get("arguments")
            if call_id and name and isinstance(arguments, dict):
                call_names[call_id] = name
                calls.append(
                    {"type": "function", "function": {"name": name, "arguments": arguments}}
                )
        if calls:
            item["tool_calls"] = calls
        if role == "tool":
            item.pop("name", None)
            item["tool_name"] = str(
                raw.get("name") or call_names.get(str(raw.get("tool_call_id") or "")) or ""
            )
        result.append(item)
    return result


def _canonical_message(value: Any) -> dict[str, Any]:
    message = dict(value) if isinstance(value, dict) else {}
    if "thinking" in message:
        message["continuation"] = {"thinking": message.pop("thinking")}
    calls: list[dict[str, Any]] = []
    for index, call in enumerate(
        message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
    ):
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name, arguments = str(function.get("name") or "").strip(), function.get("arguments")
        if name and isinstance(arguments, dict):
            calls.append(
                {
                    "id": str(call.get("id") or f"ollama_call_{index}"),
                    "name": name,
                    "arguments": arguments,
                }
            )
    if calls:
        message["tool_calls"] = calls
    else:
        message.pop("tool_calls", None)
    return message


def _json_size(value: Any) -> int:
    try:
        return len(json.dumps(value, separators=(",", ":")).encode())
    except (TypeError, ValueError) as exc:
        raise RuntimeError("invalid_response") from exc


def _bounded_model_info(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    if _json_size(value) <= MAX_MODEL_INFO_BYTES:
        return value
    bounded: dict[str, Any] = {}
    for raw_key, item in value.items():
        key = str(raw_key)
        if len(key.encode()) > 256:
            continue
        candidate = {**bounded, key: item}
        if _json_size(candidate) <= MAX_MODEL_INFO_BYTES:
            bounded[key] = item
    return bounded


def list_models(request: dict[str, Any]) -> dict[str, Any]:
    base_url, headers = _connection(request)
    deadline = time.monotonic() + MAX_DISCOVERY_SECONDS
    token, offset, limit = _pagination(request)
    if token:
        tags = _read_discovery_snapshot(token)
    else:
        tags = _request_json(
            f"{base_url}/api/tags",
            method="GET",
            headers=headers,
            timeout_seconds=MAX_DISCOVERY_SECONDS,
        ).get("models", [])
        if not isinstance(tags, list) or len(tags) > MAX_DISCOVERY_MODELS:
            raise RuntimeError("invalid_response")
    page_tags = tags[offset : offset + limit]
    models: list[dict[str, Any]] = []
    for item in page_tags:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        if len(name.encode()) > MAX_MODEL_ID_BYTES:
            raise RuntimeError("invalid_response")
        details: dict[str, Any] = {}
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("timeout")
            details = _request_json(
                f"{base_url}/api/show",
                method="POST",
                headers=headers,
                body={"model": name},
                timeout_seconds=remaining,
            )
        except RuntimeError as exc:
            if str(exc) == "timeout" or time.monotonic() >= deadline:
                raise RuntimeError("timeout") from exc
            # Preserve the last complete discovery snapshot on endpoint errors;
            # an individual /show failure is represented as unverified metadata.
            details = {}
        model_info = _bounded_model_info(details.get("model_info"))
        digest = str(item.get("digest") or "")
        modified_at = str(item.get("modified_at") or "")
        if len(digest.encode()) > 512 or len(modified_at.encode()) > 128:
            raise RuntimeError("invalid_response")
        models.append(
            {
                "model_id": name,
                "label": name,
                # Endpoint support is preliminary; the host still gates routing on
                # the mandatory live capability suite for each selected model.
                "capabilities": ["chat", "tools", "structured_output"],
                "metadata": {
                    "digest": digest,
                    "modified_at": modified_at,
                    "model_info": model_info,
                },
                "upstream_revision": digest,
                "upstream_digest": digest,
            }
        )
        candidate = {"models": models, "next_cursor": "", "snapshot_complete": True}
        if _json_size(candidate) > MAX_DISCOVERY_RESULT_BYTES:
            raise RuntimeError("invalid_response")
    next_offset = offset + len(page_tags)
    if next_offset < len(tags):
        token = token or _new_discovery_snapshot(tags)
        next_cursor = f"{token}:{next_offset}"
    else:
        if token:
            _drop_discovery_snapshot(token)
        next_cursor = ""
    return {"models": models, "next_cursor": next_cursor, "snapshot_complete": True}


def chat(request: dict[str, Any], *, structured: bool = False) -> dict[str, Any]:
    payload = dict(request.get("input") or {})
    base_url, headers = _connection(request)
    body: dict[str, Any] = {
        "model": payload.get("model_id"),
        "messages": _ollama_messages(payload.get("messages")),
        "stream": False,
    }
    tool_choice = str(payload.get("tool_choice") or "auto").strip().lower()
    if tool_choice not in {"auto", "none", "required"}:
        raise RuntimeError("capability_mismatch")
    for key in ("tools", "options"):
        if key in payload:
            body[key] = payload[key]
    if tool_choice == "none":
        body.pop("tools", None)
    elif tool_choice == "required" and not body.get("tools"):
        raise RuntimeError("capability_mismatch")
    options = dict(body.get("options") or {})
    if "temperature" in payload:
        options["temperature"] = payload["temperature"]
    if "max_tokens" in payload:
        options["num_predict"] = payload["max_tokens"]
    if options:
        body["options"] = options
    if structured:
        body["format"] = payload.get("schema") or "json"
    response = _request_json(f"{base_url}/api/chat", method="POST", headers=headers, body=body)
    if response.get("done") is not True or not isinstance(response.get("message"), dict):
        raise RuntimeError("invalid_response")
    message = _canonical_message(response["message"])
    if tool_choice == "required" and not message.get("tool_calls"):
        raise RuntimeError("capability_mismatch")
    if tool_choice == "none" and message.get("tool_calls"):
        raise RuntimeError("capability_mismatch")
    usage = {
        "input_tokens": response.get("prompt_eval_count", 0),
        "output_tokens": response.get("eval_count", 0),
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    usage["total_tokens"] = int(usage["input_tokens"] or 0) + int(usage["output_tokens"] or 0)
    result: dict[str, Any] = {
        "message": message,
        "finish_reason": response.get("done_reason") or "stop",
        "usage": usage,
    }
    if structured:
        result["structured_output"] = message.get("content")
    return result


class Handler(BaseHTTPRequestHandler):
    token = os.environ.get("FS_EXTENSION_TOKEN", "")

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode()
        if len(raw) > MAX_RUNTIME_BODY_BYTES:
            status = 502
            raw = b'{"ok":false,"errors":[{"code":"invalid_response"}]}'
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send(
                200, {"ok": True, "status": "ready", "extension_id": "flowsteward.ollama-remote"}
            )
        else:
            self._send(404, {"ok": False, "errors": [{"code": "not_found"}]})

    def do_POST(self) -> None:
        if not self.token:
            self._send(503, {"ok": False, "errors": [{"code": "runtime_auth_not_configured"}]})
            return
        if self.headers.get("Authorization") != f"Bearer {self.token}":
            self._send(401, {"ok": False, "errors": [{"code": "unauthorized"}]})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(_read_request_body(self.rfile, content_length).decode() or "{}")
            if self.path == "/handshake":
                result = {
                    "extension_id": "flowsteward.ollama-remote",
                    "contract_version": "extension_host_v1",
                }
            elif self.path == "/queries/llm.list_models":
                result = list_models(body)
            elif self.path == "/actions/llm.chat":
                result = _execute_idempotent_action(body, "llm.chat", lambda: chat(body))
            elif self.path == "/actions/llm.chat_structured":
                result = _execute_idempotent_action(
                    body, "llm.chat_structured", lambda: chat(body, structured=True)
                )
            elif self.path == "/events":
                result = {"accepted": True}
            else:
                self._send(404, {"ok": False, "errors": [{"code": "not_found"}]})
                return
            self._send(200, {"ok": True, "result": result, "errors": []})
        except Exception as exc:
            semantics = "safe_read" if self.path == "/queries/llm.list_models" else "mutation"
            self._send(502, _failure_response(str(exc), request_semantics=semantics))

    def log_message(self, *_: Any) -> None:
        return


def _server_address() -> tuple[str, int]:
    host = str(os.environ.get("FS_EXTENSION_BIND_HOST") or "127.0.0.1").strip()
    if not host:
        raise RuntimeError("FS_EXTENSION_BIND_HOST must not be empty")
    return host, int(os.environ.get("FS_EXTENSION_PORT", "8090"))


if __name__ == "__main__":
    if not Handler.token:
        raise SystemExit("FS_EXTENSION_TOKEN is required for the Ollama remote runtime")
    ThreadingHTTPServer(_server_address(), Handler).serve_forever()
