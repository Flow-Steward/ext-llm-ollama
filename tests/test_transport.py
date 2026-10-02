"""The HTTP layer: public-HTTPS-only targets, DNS pinning, bounds, safe errors.

Every upstream request goes through ``runtime._request_json``. These tests
replace only the socket layer (``http.client.HTTPSConnection`` and DNS), so the
URL checks, status mapping and body bounds under test are the real ones.
"""

from __future__ import annotations

import io
import socket

import pytest


class _Response:
    def __init__(self, status: int = 200, body: bytes = b"{}") -> None:
        self.status = status
        self._body = io.BytesIO(body)

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)


def _connection_class(*, response=None, error: Exception | None = None, captured=None):
    class _Connection:
        def __init__(self, host, port, **kwargs):
            if captured is not None:
                captured.update(host=host, port=port, **kwargs)
            self._create_connection = None

        def request(self, method, path, **kwargs):
            if captured is not None:
                captured.update(method=method, path=path, **kwargs)
                captured["dial"] = self._create_connection
            if error is not None:
                raise error

        def getresponse(self):
            return response or _Response()

        def close(self):
            if captured is not None:
                captured["closed"] = True

    return _Connection


@pytest.fixture
def public_target(monkeypatch, runtime):
    monkeypatch.setattr(
        runtime,
        "_upstream_target",
        lambda _url, **_kwargs: ("https", "ollama.example", "203.0.113.8", 443, "/x"),
    )


# --------------------------------------------------------------------------- #
# URL and DNS policy                                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "url",
    [
        "http://ollama.example",  # plain HTTP
        "https://localhost",
        "https://ollama.local",
        "https://user:pass@ollama.example",  # credentials in the URL
        "ftp://ollama.example",
        "",
    ],
)
def test_non_public_https_urls_are_rejected(runtime, url) -> None:
    with pytest.raises(RuntimeError, match="model_unavailable"):
        runtime._upstream_target(url)


@pytest.mark.parametrize(
    ("address", "family"),
    [
        ("127.0.0.1", socket.AF_INET),  # loopback
        ("10.1.2.3", socket.AF_INET),  # private
        ("100.64.0.1", socket.AF_INET),  # CGNAT
        ("169.254.169.254", socket.AF_INET),  # link-local / cloud metadata
        ("224.0.0.1", socket.AF_INET),  # multicast
        ("::1", socket.AF_INET6),
        ("fec0::1", socket.AF_INET6),  # site-local
        ("ff02::1", socket.AF_INET6),
    ],
)
def test_non_public_dns_answers_are_rejected(monkeypatch, runtime, address, family) -> None:
    monkeypatch.setattr(
        runtime.socket,
        "getaddrinfo",
        lambda *_a, **_k: [(family, socket.SOCK_STREAM, 6, "", (address, 443))],
    )

    with pytest.raises(RuntimeError, match="model_unavailable"):
        runtime._upstream_target("https://ollama.example")


def test_one_non_public_answer_among_public_ones_is_rejected(monkeypatch, runtime) -> None:
    monkeypatch.setattr(
        runtime.socket,
        "getaddrinfo",
        lambda *_a, **_k: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443)),
        ],
    )

    with pytest.raises(RuntimeError, match="model_unavailable"):
        runtime._upstream_target("https://ollama.example")


def test_public_answers_resolve_to_a_pinned_target(monkeypatch, runtime) -> None:
    monkeypatch.setattr(
        runtime.socket,
        "getaddrinfo",
        lambda *_a, **_k: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2606:4700:4700::1111", 443)),
        ],
    )

    assert runtime._upstream_target("https://ollama.example/api/tags?x=1") == (
        "https",
        "ollama.example",
        "8.8.8.8",
        443,
        "/api/tags?x=1",
    )


def test_dns_failure_is_an_upstream_error(monkeypatch, runtime) -> None:
    def fail(*_a, **_k):
        raise socket.gaierror("no such host")

    monkeypatch.setattr(runtime.socket, "getaddrinfo", fail)

    with pytest.raises(RuntimeError, match="upstream_error"):
        runtime._upstream_target("https://ollama.example")


# --------------------------------------------------------------------------- #
# Requests                                                                     #
# --------------------------------------------------------------------------- #


def test_request_keeps_the_hostname_for_tls_and_dials_the_checked_address(
    monkeypatch, runtime, public_target
) -> None:
    captured: dict = {}
    dialed: list = []
    monkeypatch.setattr(
        runtime.http.client, "HTTPSConnection", _connection_class(captured=captured)
    )
    monkeypatch.setattr(
        runtime.socket, "create_connection", lambda address, *_a, **_k: dialed.append(address)
    )

    runtime._request_json("https://ollama.example/x", method="POST", headers={}, body={"a": 1})
    captured["dial"](("ignored.example", 443), 5)

    assert captured["host"] == "ollama.example"  # SNI and certificate check
    assert captured["headers"]["Host"] == "ollama.example"
    assert captured["body"] == b'{"a": 1}'
    assert dialed == [("203.0.113.8", 443)]
    assert captured["closed"] is True


def test_request_names_a_nondefault_port_in_the_host_header(monkeypatch, runtime) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        runtime,
        "_upstream_target",
        lambda _url, **_kwargs: ("https", "ollama.example", "8.8.8.8", 8443, "/"),
    )
    monkeypatch.setattr(
        runtime.http.client, "HTTPSConnection", _connection_class(captured=captured)
    )

    runtime._request_json("https://ollama.example:8443", method="GET", headers={})

    assert captured["headers"]["Host"] == "ollama.example:8443"


@pytest.mark.parametrize(
    ("status", "category"),
    [
        (302, "upstream_error"),  # redirects are never followed
        (400, "upstream_error"),
        (401, "authentication_error"),
        (402, "billing_quota_exceeded"),  # ollama.com: model outside the plan
        (403, "authentication_error"),
        (404, "model_unavailable"),
        (408, "timeout"),
        (429, "rate_limited"),
        (500, "upstream_error"),
        (503, "upstream_error"),
    ],
)
def test_http_status_maps_to_one_safe_category(
    monkeypatch, runtime, public_target, status, category
) -> None:
    monkeypatch.setattr(
        runtime.http.client,
        "HTTPSConnection",
        _connection_class(response=_Response(status, b'{"error":"secret detail"}')),
    )

    with pytest.raises(RuntimeError) as caught:
        runtime._request_json("https://ollama.example", method="GET", headers={})

    assert str(caught.value) == category


@pytest.mark.parametrize(
    ("error", "category"),
    [
        (TimeoutError("timed out"), "timeout"),
        (ConnectionRefusedError("refused"), "upstream_error"),
        (OSError("network is unreachable"), "upstream_error"),
    ],
)
def test_network_failures_map_to_safe_categories(
    monkeypatch, runtime, public_target, error, category
) -> None:
    monkeypatch.setattr(runtime.http.client, "HTTPSConnection", _connection_class(error=error))

    with pytest.raises(RuntimeError) as caught:
        runtime._request_json("https://ollama.example", method="GET", headers={})

    assert str(caught.value) == category


@pytest.mark.parametrize("body", [b"not json", b"[1, 2]", b'"text"'])
def test_a_body_that_is_not_a_json_object_is_invalid(
    monkeypatch, runtime, public_target, body
) -> None:
    monkeypatch.setattr(
        runtime.http.client, "HTTPSConnection", _connection_class(response=_Response(200, body))
    )

    with pytest.raises(RuntimeError, match="invalid_response"):
        runtime._request_json("https://ollama.example", method="GET", headers={})


def test_an_oversized_body_is_refused_before_parsing(monkeypatch, runtime, public_target) -> None:
    oversized = b"x" * (runtime.MAX_RUNTIME_BODY_BYTES + 1)
    monkeypatch.setattr(
        runtime.http.client,
        "HTTPSConnection",
        _connection_class(response=_Response(200, oversized)),
    )

    with pytest.raises(RuntimeError, match="invalid_response"):
        runtime._request_json("https://ollama.example", method="GET", headers={})


# --------------------------------------------------------------------------- #
# Connection settings                                                          #
# --------------------------------------------------------------------------- #


def test_no_base_url_means_ollamas_hosted_api(monkeypatch, runtime) -> None:
    checked: list[str] = []
    monkeypatch.setattr(
        runtime, "_upstream_target", lambda url, **kwargs: checked.append((url, kwargs))
    )

    base_url, headers, allow_private = runtime._connection({"connection": {}})

    assert base_url == "https://ollama.com"
    assert checked == [("https://ollama.com", {"allow_private": False})]
    assert allow_private is False
    assert "Authorization" not in headers


def test_a_custom_base_url_and_key_are_used(monkeypatch, runtime) -> None:
    monkeypatch.setattr(runtime, "_upstream_target", lambda _url, **_kwargs: None)

    base_url, headers, _ = runtime._connection(
        {
            "connection": {
                "connection_config": {"upstream_base_url": " https://llm.example.com/ "},
                "credentials": {"api_key": " key-1 "},
            }
        }
    )

    assert base_url == "https://llm.example.com"
    assert headers["Authorization"] == "Bearer key-1"


def test_an_unsafe_custom_base_url_is_refused_before_any_request(runtime) -> None:
    with pytest.raises(RuntimeError, match="model_unavailable"):
        runtime._connection(
            {"connection": {"connection_config": {"upstream_base_url": "http://10.0.0.5"}}}
        )


# --------------------------------------------------------------------------- #
# Self-hosted Ollama on a private network                                      #
# --------------------------------------------------------------------------- #


def _resolve_to(monkeypatch, runtime, *addresses: str) -> None:
    monkeypatch.setattr(
        runtime.socket,
        "getaddrinfo",
        lambda *_a, **_k: [
            (
                socket.AF_INET6 if ":" in address else socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                (address, 11434),
            )
            for address in addresses
        ],
    )


@pytest.mark.parametrize(
    ("url", "address"),
    [
        ("http://localhost:11434", "127.0.0.1"),
        ("http://192.168.1.20:11434", "192.168.1.20"),
        ("http://ollama.lan:11434", "10.0.0.7"),
        ("https://ollama.lan", "172.16.4.2"),
        # Docker Desktop's name for the machine running the Compact container.
        ("http://host.docker.internal:11434", "fdc4:f303:9324::254"),
    ],
)
def test_private_endpoints_are_refused_unless_the_host_allows_them(
    monkeypatch, runtime, url, address
) -> None:
    _resolve_to(monkeypatch, runtime, address)

    with pytest.raises(RuntimeError, match="model_unavailable"):
        runtime._upstream_target(url)
    scheme, host, pinned, _port, _path = runtime._upstream_target(url, allow_private=True)

    assert (scheme, pinned) == (url.split(":", 1)[0], address)


@pytest.mark.parametrize("allow_private", [False, True])
def test_a_public_endpoint_always_needs_https(monkeypatch, runtime, allow_private) -> None:
    _resolve_to(monkeypatch, runtime, "8.8.8.8")

    with pytest.raises(RuntimeError, match="model_unavailable"):
        runtime._upstream_target("http://ollama.example.com", allow_private=allow_private)


def test_plain_http_needs_every_answer_to_be_private(monkeypatch, runtime) -> None:
    _resolve_to(monkeypatch, runtime, "10.0.0.7", "8.8.8.8")

    with pytest.raises(RuntimeError, match="model_unavailable"):
        runtime._upstream_target("http://split.example.com", allow_private=True)


@pytest.mark.parametrize("allow_private", [False, True])
@pytest.mark.parametrize(
    "address",
    ["169.254.169.254", "169.254.1.1", "fe80::1", "100.64.0.1", "0.0.0.0", "224.0.0.1"],
)
def test_metadata_and_link_local_addresses_are_refused_either_way(
    monkeypatch, runtime, address, allow_private
) -> None:
    _resolve_to(monkeypatch, runtime, address)

    with pytest.raises(RuntimeError, match="model_unavailable"):
        runtime._upstream_target("http://ollama.lan:11434", allow_private=allow_private)


def test_a_private_endpoint_is_reached_over_plain_http_pinned_to_its_address(
    monkeypatch, runtime
) -> None:
    captured: dict = {}
    dialed: list = []
    _resolve_to(monkeypatch, runtime, "192.168.1.20")

    class _Plain:
        def __init__(self, host, port, **kwargs):
            captured.update(host=host, port=port, **kwargs)
            self._create_connection = None

        def request(self, method, path, **kwargs):
            captured.update(method=method, path=path, **kwargs)
            self._create_connection(("ignored", 0), 5)

        def getresponse(self):
            return _Response(200, b'{"models": []}')

        def close(self):
            return None

    monkeypatch.setattr(runtime.http.client, "HTTPConnection", _Plain)
    monkeypatch.setattr(
        runtime.http.client,
        "HTTPSConnection",
        lambda *_a, **_k: pytest.fail("a plain-http endpoint must not use TLS"),
    )
    monkeypatch.setattr(
        runtime.socket, "create_connection", lambda address, *_a, **_k: dialed.append(address)
    )

    result = runtime._request_json(
        "http://ollama.lan:11434/api/tags", method="GET", headers={}, allow_private=True
    )

    assert result == {"models": []}
    assert captured["headers"]["Host"] == "ollama.lan:11434"
    assert dialed == [("192.168.1.20", 11434)]  # resolved once, then pinned


def test_the_host_policy_reaches_the_url_guard(monkeypatch, runtime) -> None:
    seen: list = []
    monkeypatch.setattr(
        runtime, "_upstream_target", lambda url, **kwargs: seen.append((url, kwargs))
    )

    _, _, allow_private = runtime._connection(
        {
            "network_policy": {"allow_private_addresses": True},
            "connection": {"connection_config": {"upstream_base_url": "http://ollama.lan:11434"}},
        }
    )

    assert allow_private is True
    assert seen == [("http://ollama.lan:11434", {"allow_private": True})]


@pytest.mark.parametrize("policy", [None, {}, {"allow_private_addresses": "yes"}, "on"])
def test_anything_but_an_explicit_true_keeps_private_hosts_refused(runtime, policy) -> None:
    with pytest.raises(RuntimeError, match="model_unavailable"):
        runtime._connection(
            {
                "network_policy": policy,
                "connection": {
                    "connection_config": {"upstream_base_url": "http://127.0.0.1:11434"}
                },
            }
        )
