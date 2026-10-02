"""Ollama provider operations for the Flow Steward llm_provider contract.

The extension runs as an ordinary Flow Steward extension subprocess (see
``main.py``). Each operation receives the provider's own settings — an optional
base URL (default: Ollama's hosted API) and an optional API key — and talks to
that Ollama endpoint over public HTTPS only.
"""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import socket
import ssl
import time
from typing import Any
from urllib.parse import urlsplit

MAX_RUNTIME_BODY_BYTES = 5 * 1024 * 1024
MAX_DISCOVERY_RESULT_BYTES = 4 * 1024 * 1024
MAX_DISCOVERY_MODELS = 256
MAX_MODEL_INFO_BYTES = 32 * 1024
MAX_MODEL_ID_BYTES = 1_024
MAX_SHOW_MODELS_PER_PAGE = 25
MAX_DISCOVERY_SECONDS = 45
# Ollama's hosted API (https://docs.ollama.com/cloud). A self-hosted Ollama on a
# public HTTPS address can be configured instead.
DEFAULT_BASE_URL = "https://ollama.com"
STRUCTURED_OUTPUT_TOOL = "flow_steward_structured_output"
SAFE_ERROR_CODES = frozenset(
    {
        "authentication_error",
        "billing_quota_exceeded",
        "capability_mismatch",
        "invalid_response",
        "model_unavailable",
        "rate_limited",
        "timeout",
        "upstream_error",
    }
)


def _read_bounded(stream: Any) -> bytes:
    raw = stream.read(MAX_RUNTIME_BODY_BYTES + 1)
    if len(raw) > MAX_RUNTIME_BODY_BYTES:
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
            # ollama.com answers 402 for a model outside the account's plan
            # ("not included in your free usage").
            402: "billing_quota_exceeded",
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
    if safe_code not in SAFE_ERROR_CODES:
        safe_code = "upstream_error"
    retryable_code = safe_code in {"rate_limited", "timeout", "upstream_error"}
    definite_refusal = safe_code in {
        "authentication_error",
        "billing_quota_exceeded",
        "model_unavailable",
        "rate_limited",
    }
    definitely_no_effect = request_semantics == "safe_read" or definite_refusal
    message = f"Ollama request failed: {safe_code}"
    return {
        "ok": False,
        "error_code": safe_code,
        "error": message,
        "errors": [{"code": safe_code, "message": message}],
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
    base_url = str((config or {}).get("upstream_base_url") or "").strip().rstrip("/")
    base_url = base_url or DEFAULT_BASE_URL
    _public_https_url(base_url)
    headers = {"Content-Type": "application/json"}
    # Ollama's hosted API needs a key; a self-hosted endpoint may not.
    token = str((credentials or {}).get("api_key") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return base_url, headers


def _pagination(request: dict[str, Any]) -> tuple[str, int, int]:
    query = request.get("query") if isinstance(request, dict) else {}
    query = query if isinstance(query, dict) else {}
    cursor = str(query.get("cursor") or "").strip()
    fingerprint = ""
    offset = 0
    if cursor:
        fingerprint, separator, raw_offset = cursor.partition(":")
        if not separator or not fingerprint or not raw_offset.isdigit():
            raise RuntimeError("invalid_response")
        offset = int(raw_offset)
    try:
        requested_limit = int(query.get("limit") or MAX_SHOW_MODELS_PER_PAGE)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("invalid_response") from exc
    if offset < 0 or requested_limit < 1:
        raise RuntimeError("invalid_response")
    return fingerprint, offset, min(requested_limit, MAX_SHOW_MODELS_PER_PAGE)


def _tags_fingerprint(tags: list[Any]) -> str:
    """Identify one tag listing so every page of a discovery reads the same list."""
    identity = [
        [str(item.get("name") or ""), str(item.get("digest") or "")]
        for item in tags
        if isinstance(item, dict)
    ]
    raw = json.dumps(identity, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:32]


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
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = None
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


def _discovered_capabilities(details: dict[str, Any]) -> list[str]:
    """Map /api/show capabilities; structured output rides on tool calling.

    Ollama's hosted API ignores the ``format`` schema, so structured output is
    delivered through one forced tool call and needs the ``tools`` capability.
    Without /api/show metadata the model stays a candidate for verification.
    """
    reported = details.get("capabilities")
    if not isinstance(reported, list):
        return ["chat", "tools", "structured_output"]
    names = {str(item).strip() for item in reported}
    if "tools" in names:
        return ["chat", "tools", "structured_output"]
    return ["chat"]


def list_models(request: dict[str, Any]) -> dict[str, Any]:
    base_url, headers = _connection(request)
    deadline = time.monotonic() + MAX_DISCOVERY_SECONDS
    expected_fingerprint, offset, limit = _pagination(request)
    tags = _request_json(
        f"{base_url}/api/tags",
        method="GET",
        headers=headers,
        timeout_seconds=MAX_DISCOVERY_SECONDS,
    ).get("models", [])
    if not isinstance(tags, list) or len(tags) > MAX_DISCOVERY_MODELS:
        raise RuntimeError("invalid_response")
    fingerprint = _tags_fingerprint(tags)
    if expected_fingerprint and expected_fingerprint != fingerprint:
        # The endpoint's model list changed between pages: the host keeps its
        # last complete catalog rather than reconciling a mixed snapshot.
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
                "capabilities": _discovered_capabilities(details),
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
    next_cursor = f"{fingerprint}:{next_offset}" if next_offset < len(tags) else ""
    return {"models": models, "next_cursor": next_cursor, "snapshot_complete": True}


_SCHEMA_NAME_MAPS = frozenset({"properties", "patternProperties", "$defs", "definitions"})
_SCHEMA_LITERALS = frozenset({"const", "enum", "default", "examples"})


def _portable_schema(value: Any) -> Any:
    """Rewrite ``const: x`` as the equivalent ``enum: [x]``, recursively.

    Ollama renders a tool's parameters into the model's prompt template, and
    the templates of several models (gpt-oss, gemma) drop ``const``, so the
    model never sees the constraint. Both keywords mean the same thing in JSON
    Schema, and the host still validates the answer against the schema it sent.
    """
    if isinstance(value, list):
        return [_portable_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    # Beside an explicit ``enum``, ``const`` stays: dropping it would loosen
    # the schema.
    rewrite_const = "const" in value and "enum" not in value
    rewritten: dict[str, Any] = {}
    for key, item in value.items():
        if key == "const" and rewrite_const:
            rewritten["enum"] = [item]
        elif key in _SCHEMA_LITERALS:
            rewritten[key] = item  # JSON values, not schemas
        elif key in _SCHEMA_NAME_MAPS and isinstance(item, dict):
            # Keys here are property/definition names, not keywords: a
            # property may itself be called "const".
            rewritten[key] = {name: _portable_schema(sub) for name, sub in item.items()}
        else:
            rewritten[key] = _portable_schema(item)
    return rewritten


def _structured_output_tool(payload: dict[str, Any]) -> dict[str, Any]:
    schema = payload.get("schema")
    if not isinstance(schema, dict) or not schema:
        raise RuntimeError("capability_mismatch")
    name = str(payload.get("schema_name") or "").strip() or "the requested result"
    return {
        "type": "function",
        "function": {
            "name": STRUCTURED_OUTPUT_TOOL,
            "description": f"Return {name}. Call this exactly once with the complete answer.",
            "parameters": _portable_schema(schema),
        },
    }


def _structured_result(message: dict[str, Any]) -> Any:
    """Take the answer from the forced tool call, else from JSON message content."""
    for call in message.get("tool_calls") or []:
        if call.get("name") == STRUCTURED_OUTPUT_TOOL:
            return call.get("arguments")
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        try:
            return json.loads(content)
        except ValueError as exc:
            raise RuntimeError("capability_mismatch") from exc
    raise RuntimeError("capability_mismatch")


def chat(request: dict[str, Any], *, structured: bool = False) -> dict[str, Any]:
    payload = dict(request.get("input") or {})
    base_url, headers = _connection(request)
    body: dict[str, Any] = {
        "model": payload.get("model_id"),
        "messages": _ollama_messages(payload.get("messages")),
        "stream": False,
    }
    # Ollama has no tool_choice parameter: "none" withholds the tools and
    # "required" is enforced on the response.
    tool_choice = str(payload.get("tool_choice") or "auto").strip().lower()
    if tool_choice not in {"auto", "none", "required"}:
        raise RuntimeError("capability_mismatch")
    for key in ("tools", "options"):
        if key in payload:
            body[key] = payload[key]
    if isinstance(body.get("tools"), list):
        body["tools"] = _portable_schema(body["tools"])
    if tool_choice == "none" or structured:
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
        # A local Ollama honours ``format``; the hosted API does not, so the
        # schema is also offered as the one tool the model must call.
        body["format"] = _portable_schema(payload.get("schema")) or "json"
        body["tools"] = [_structured_output_tool(payload)]
        body["messages"] = [
            *body["messages"],
            {
                "role": "user",
                # The schema is spelled out in the text as well: the tool
                # definition alone reaches the model through its prompt
                # template, which may drop constraints (const, enum, bounds).
                "content": (
                    f"Respond only by calling the {STRUCTURED_OUTPUT_TOOL} tool once "
                    "with the complete answer as its arguments. The arguments must "
                    "be valid against this JSON Schema, including every const and "
                    "enum value: " + json.dumps(payload.get("schema"), separators=(",", ":"))
                ),
            },
        ]
    response = _request_json(f"{base_url}/api/chat", method="POST", headers=headers, body=body)
    if response.get("done") is not True or not isinstance(response.get("message"), dict):
        raise RuntimeError("invalid_response")
    message = _canonical_message(response["message"])
    result: dict[str, Any] = {"finish_reason": response.get("done_reason") or "stop"}
    if structured:
        result["structured_output"] = _structured_result(message)
        message.pop("tool_calls", None)
        message["content"] = json.dumps(result["structured_output"], separators=(",", ":"))
    elif tool_choice == "required" and not message.get("tool_calls"):
        raise RuntimeError("capability_mismatch")
    elif tool_choice == "none" and message.get("tool_calls"):
        raise RuntimeError("capability_mismatch")
    elif payload.get("parallel_tool_calls") is False and len(message.get("tool_calls") or []) > 1:
        # Ollama has no parallel_tool_calls switch; the host's limit of one
        # call per turn is enforced on the response.
        raise RuntimeError("capability_mismatch")
    usage = {
        "input_tokens": response.get("prompt_eval_count", 0),
        "output_tokens": response.get("eval_count", 0),
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    usage["total_tokens"] = int(usage["input_tokens"] or 0) + int(usage["output_tokens"] or 0)
    return {"message": message, **result, "usage": usage}
