"""``llm.list_models``: one page of the endpoint's models per call.

Each call is a separate subprocess, so pagination is stateless: the cursor
carries a fingerprint of the tag list plus an offset, and every page re-reads
``/api/tags`` to prove it is still the same list.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .conftest import connection

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _tags(count: int) -> list[dict]:
    return [{"name": f"model-{index}", "digest": f"sha256:{index}"} for index in range(count)]


def _list(runtime, *, cursor: str = "", limit: int = 25, **conn):
    return runtime.list_models(
        {"connection": connection(**conn), "query": {"cursor": cursor, "limit": limit}}
    )


def test_models_come_from_tags_with_details_from_show(runtime, ollama) -> None:
    ollama.tags = [
        {"name": "gpt-oss:20b", "digest": "sha256:a", "modified_at": "2026-06-12T00:00:00Z"}
    ]
    ollama.show = {
        "gpt-oss:20b": {
            "capabilities": ["completion", "tools", "thinking"],
            "model_info": {"general.architecture": "gptoss"},
        }
    }

    result = _list(runtime, api_key="key-1")

    assert result == json.loads((FIXTURES / "llm.list_models.json").read_text())["result"]
    assert [call["url"] for call in ollama.calls] == [
        "https://ollama.com/api/tags",
        "https://ollama.com/api/show",
    ]
    assert all(call["headers"]["Authorization"] == "Bearer key-1" for call in ollama.calls)


def test_a_self_hosted_endpoint_without_a_key_sends_no_authorization(runtime, ollama) -> None:
    ollama.tags = _tags(1)

    _list(runtime, base_url="https://llm.example.com")

    assert ollama.calls[0]["url"] == "https://llm.example.com/api/tags"
    assert all("Authorization" not in call["headers"] for call in ollama.calls)


@pytest.mark.parametrize(
    ("show", "capabilities"),
    [
        ({"capabilities": ["completion", "tools"]}, ["chat", "tools", "structured_output"]),
        # Without tool calling there is no structured output either: it rides
        # on one forced tool call (the hosted API ignores ``format``).
        ({"capabilities": ["completion", "vision"]}, ["chat"]),
        # No metadata: still a candidate; the host's verification decides.
        ({}, ["chat", "tools", "structured_output"]),
    ],
)
def test_capabilities_follow_show_metadata(runtime, ollama, show, capabilities) -> None:
    ollama.tags = _tags(1)
    ollama.show = {"model-0": show}

    assert _list(runtime)["models"][0]["capabilities"] == capabilities


def test_a_failing_show_keeps_the_model_with_unknown_details(runtime, ollama) -> None:
    ollama.tags = _tags(2)
    ollama.show = {"model-0": RuntimeError("model_unavailable"), "model-1": {}}

    result = _list(runtime)

    assert [model["model_id"] for model in result["models"]] == ["model-0", "model-1"]
    assert result["models"][0]["metadata"]["model_info"] == {}


def test_model_ids_are_kept_exactly_and_digest_tracks_revisions(runtime, ollama) -> None:
    ollama.tags = [{"name": "qwen2.5-coder:32b", "digest": "sha256:abc"}]

    model = _list(runtime)["models"][0]

    assert model["model_id"] == model["label"] == "qwen2.5-coder:32b"
    assert model["upstream_digest"] == model["upstream_revision"] == "sha256:abc"


def test_pages_are_stateless_and_bound_to_one_tag_list(runtime, ollama) -> None:
    ollama.tags = _tags(30)

    first = _list(runtime, limit=10)
    shows_before = len(ollama.calls_to("/api/show"))
    second = _list(runtime, cursor=first["next_cursor"], limit=10)
    third = _list(runtime, cursor=second["next_cursor"], limit=10)

    assert [m["model_id"] for m in second["models"]] == [f"model-{i}" for i in range(10, 20)]
    assert len(ollama.calls_to("/api/show")) - shows_before == 20  # only the requested pages
    assert len(ollama.calls_to("/api/tags")) == 3  # every page re-reads the list
    assert first["next_cursor"].endswith(":10")
    assert second["next_cursor"].endswith(":20")
    assert third["next_cursor"] == ""
    assert all(page["snapshot_complete"] is True for page in (first, second, third))


def test_a_list_that_changes_between_pages_is_refused(runtime, ollama) -> None:
    ollama.tags = _tags(30)
    first = _list(runtime, limit=10)
    ollama.tags = _tags(31)

    with pytest.raises(RuntimeError, match="invalid_response"):
        _list(runtime, cursor=first["next_cursor"], limit=10)


def test_the_page_size_is_capped_to_bound_show_requests(runtime, ollama) -> None:
    ollama.tags = _tags(runtime.MAX_SHOW_MODELS_PER_PAGE + 5)

    result = _list(runtime, limit=200)

    assert len(result["models"]) == runtime.MAX_SHOW_MODELS_PER_PAGE
    assert result["next_cursor"].endswith(f":{runtime.MAX_SHOW_MODELS_PER_PAGE}")


@pytest.mark.parametrize("cursor", ["no-separator", ":5", "abc:", "abc:-1", "abc:x"])
def test_a_malformed_cursor_is_invalid(runtime, ollama, cursor) -> None:
    ollama.tags = _tags(3)

    with pytest.raises(RuntimeError, match="invalid_response"):
        _list(runtime, cursor=cursor)


def test_too_many_models_are_refused_before_any_show_request(runtime, ollama) -> None:
    ollama.tags = _tags(runtime.MAX_DISCOVERY_MODELS + 1)

    with pytest.raises(RuntimeError, match="invalid_response"):
        _list(runtime)
    assert ollama.calls_to("/api/show") == []


def test_oversized_model_metadata_is_trimmed(monkeypatch, runtime, ollama) -> None:
    monkeypatch.setattr(runtime, "MAX_MODEL_INFO_BYTES", 80)
    ollama.tags = _tags(1)
    ollama.show = {"model-0": {"model_info": {"small": 1, "oversized": "x" * 1_000}}}

    assert _list(runtime)["models"][0]["metadata"]["model_info"] == {"small": 1}


def test_discovery_stops_at_its_time_budget(monkeypatch, runtime, ollama) -> None:
    ollama.tags = _tags(2)
    ticks = iter([0.0, runtime.MAX_DISCOVERY_SECONDS + 1.0])
    monkeypatch.setattr(runtime.time, "monotonic", lambda: next(ticks))

    with pytest.raises(RuntimeError, match="timeout"):
        _list(runtime)
    assert ollama.calls_to("/api/show") == []


def test_a_tags_answer_without_a_model_list_is_invalid(monkeypatch, runtime, ollama) -> None:
    monkeypatch.setattr(runtime, "_request_json", lambda *_a, **_k: {"models": "nope"})

    with pytest.raises(RuntimeError, match="invalid_response"):
        _list(runtime)
