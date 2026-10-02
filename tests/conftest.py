"""Shared test helpers: the extension's own modules and a fake Ollama upstream.

No test talks to the network. ``FakeOllama`` replaces ``runtime._request_json``
(the single HTTP seam) and answers by path, recording every request, so a test
can assert exactly what would have been sent to Ollama.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
# main.py imports ``runtime`` as a top-level module, exactly as the host runs it
# (``python3 main.py`` from the bundle root).
sys.path.insert(0, str(ROOT))

import main as main_module  # noqa: E402
import runtime as runtime_module  # noqa: E402


@pytest.fixture
def runtime():
    return runtime_module


@pytest.fixture
def main():
    return main_module


class FakeOllama:
    """Answers ``/api/tags``, ``/api/show`` and ``/api/chat`` from canned data."""

    def __init__(self) -> None:
        self.tags: list[dict[str, Any]] = []
        self.show: dict[str, dict[str, Any]] = {}
        self.chat_replies: list[Any] = []
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url: str, *, method: str, headers: dict, body: Any = None, **kwargs):
        self.calls.append(
            {"url": url, "method": method, "headers": dict(headers), "body": body, **kwargs}
        )
        if url.endswith("/api/tags"):
            return {"models": list(self.tags)}
        if url.endswith("/api/show"):
            reply = self.show.get(body["model"], {})
            if isinstance(reply, Exception):
                raise reply
            return reply
        if url.endswith("/api/chat"):
            reply = self.chat_replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply
        raise AssertionError(f"unexpected upstream call {method} {url}")

    def calls_to(self, path: str) -> list[dict[str, Any]]:
        return [call for call in self.calls if call["url"].endswith(path)]


@pytest.fixture
def ollama(monkeypatch, runtime) -> FakeOllama:
    fake = FakeOllama()
    monkeypatch.setattr(runtime, "_request_json", fake)
    # Base URLs are still validated; resolve every host to a public address.
    monkeypatch.setattr(
        runtime,
        "_upstream_target",
        lambda url, **_kwargs: ("https", "ollama.example", "203.0.113.8", 443, "/"),
    )
    return fake


def connection(*, base_url: str = "", api_key: str = "") -> dict[str, Any]:
    """The request's ``connection`` block, as main.py builds it from host resources."""
    config = {"upstream_base_url": base_url} if base_url else {}
    credentials = {"api_key": api_key} if api_key else {}
    return {"connection_config": config, "credentials": credentials}


def chat_reply(message: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """A complete non-streaming /api/chat answer."""
    return {
        "message": {"role": "assistant", **message},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 11,
        "eval_count": 5,
        **extra,
    }
