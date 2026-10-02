#!/usr/bin/env python3
"""Flow Steward extension subprocess entrypoint for the Ollama LLM provider.

Subprocess contract:
  - reads one JSON request on stdin, writes one JSON response on stdout;
  - exit 0 = success, 2 = a structured provider error.

The host passes the provider's own account-scoped settings as runtime
resources: ``connection_config`` (optional ``upstream_base_url``) and
``provider_secrets`` (optional ``api_key``).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from typing import Any

import runtime
from flowsteward_extension_sdk import RuntimeResources

QUERIES: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "llm.list_models": runtime.list_models,
}
ACTIONS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "llm.chat": runtime.chat,
    "llm.chat_structured": lambda request: runtime.chat(request, structured=True),
}


def _object(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def handle(payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Dispatch one host request to the matching provider operation."""
    resources = RuntimeResources.from_context(_object(payload.get("runtime_context")))
    connection = {
        "connection_config": resources.object("connection_config"),
        "credentials": resources.object("provider_secrets"),
    }
    mode = str(payload.get("mode") or "").strip()
    if mode == "query":
        query = _object(payload.get("query"))
        operation = QUERIES.get(str(query.get("query_id") or "").strip())
        request = {"connection": connection, "query": _object(query.get("params"))}
        semantics = "safe_read"
    elif mode == "action":
        action = _object(payload.get("action"))
        operation = ACTIONS.get(str(action.get("action_id") or "").strip())
        request = {"connection": connection, "input": _object(action.get("input"))}
        semantics = "mutation"
    else:
        operation, request, semantics = None, {}, "safe_read"
    if operation is None:
        return runtime._failure_response("invalid_response", request_semantics="safe_read"), 2
    try:
        return {"ok": True, "result": operation(request), "errors": []}, 0
    except Exception as exc:  # every failure leaves as one safe category
        return runtime._failure_response(str(exc), request_semantics=semantics), 2


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        payload = None
    if not isinstance(payload, dict):
        response = runtime._failure_response("invalid_response", request_semantics="safe_read")
        print(json.dumps(response))
        return 2
    response, exit_code = handle(payload)
    print(json.dumps(response))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
