# Ollama LLM Provider for Flow Steward

Use Ollama models in Flow Steward agents. The extension connects either to
**Ollama's hosted API** (`https://ollama.com`, needs an API key) or to **your own
Ollama server**: on a public HTTPS address, or on your machine or local network
once an administrator allows private addresses (see
[Self-hosted Ollama](#self-hosted-ollama)). Models you add appear in Flow
Steward's model catalog: you can pick them for an agent or make one a project's
default model.

- Extension id: `flowsteward.ollama-remote`
- Kind: `tool_provider` with the `llm_provider` contract (`llm_provider_extension_v1`)
- Runs as an ordinary Flow Steward extension subprocess. There is no separate
  service to deploy, no runtime URL and no runtime token.
- Needs Flow Steward's extension host contract 1.1.0 or newer
  (`runtime.compatibility.platform_min: 1.1.0`); an older host marks the bundle
  incompatible.

## Setup

1. Install the extension from the Marketplace (or **Install extension from
   file** with the release ZIP).
2. Open the extension page. The **LLM provider** panel asks for:

   | Field | Required | Meaning |
   | --- | --- | --- |
   | **Base URL** (`upstream_base_url`) | no | Leave empty for Ollama's hosted API, `https://ollama.com`. For your own server, its address, e.g. `https://ollama.example.com`, or `http://host.docker.internal:11434` for an Ollama on the machine running Flow Steward (see [Self-hosted Ollama](#self-hosted-ollama)). |
   | **API key** (`api_key`) | no | Your key from [ollama.com/settings/keys](https://ollama.com/settings/keys). The hosted API needs one; a self-hosted server usually does not. |

   The key is stored encrypted, scoped to your Flow Steward account, and only
   reaches this extension when it calls Ollama. It is sent as
   `Authorization: Bearer <key>`.
3. Click **Discover models** to load the model list.
4. Tick the models you want. Flow Steward verifies each one (see below) and
   enables it once it passes.
5. Choose a verified model on an agent, or set it as a project's default model.

## How models are listed

`llm.list_models` reads `GET /api/tags`, then `POST /api/show` for each model on
the page, to learn its capabilities:

- a model whose `/api/show` lists `tools` is offered with `chat`, `tools` and
  `structured_output`;
- a model without `tools` is offered with `chat` only, and Flow Steward will not
  enable it (it needs all three);
- a model whose `/api/show` fails is still listed, with no metadata, and
  verification decides.

Model ids are kept exactly as Ollama returns them (`gpt-oss:20b`). The `digest`
is the model's revision: when it changes, Flow Steward marks the model's
verification stale.

Pages are at most 25 models, and a whole discovery has a 45-second budget.
Every page is a separate subprocess, so the cursor holds a fingerprint of the
tag list plus an offset. If the list changes between pages, discovery fails
with `invalid_response` and Flow Steward keeps the previous catalog. Run
**Discover models** again.

## How models are verified

Flow Steward, not this extension, verifies a model before it can be enabled. It
runs a fixed probe: a plain chat, a forced tool call, a tool-result replay, and
two structured-output answers checked against a JSON Schema with exact
(`const`) values. A model that fails any step stays disabled and shows the
reason. A verification lasts 30 days.

## Ollama limits this extension works around

- **The hosted API ignores `format`.** Ollama's docs say:
  *"Ollama's Cloud currently does not support structured outputs."* So
  `llm.chat_structured` asks for the answer as **one forced tool call** whose
  parameters are the requested schema. It also sends `format`, which a local
  Ollama does enforce, and states the schema in the final instruction, because
  a model's prompt template can drop schema details. If the model answers in
  plain JSON content instead of a tool call, that JSON is used.
- **Some model templates drop `const`.** `const: x` is rewritten as the
  equivalent `enum: [x]` in tool parameters. A `const` beside an explicit
  `enum` is left alone so the schema is never loosened. Flow Steward still
  validates the answer against the original schema.
- **No `tool_choice` or `parallel_tool_calls`.** `tool_choice: none` withholds
  the tools. `required` and "one call at a time" are checked on the response:
  if the model ignores them, the call fails with `capability_mismatch`.
- **Free-tier refusals.** ollama.com answers `402` for models outside your
  plan ("not included in your free usage"). They are listed, but verifying or
  using them fails with `billing_quota_exceeded`. On the free plan,
  `gpt-oss:20b`, `gpt-oss:120b` and `gemma4:31b` passed verification; others
  (for example `glm-5.3-flash`, `deepseek-v4.1-flash`) were refused.
- **`/api/tags` is public on ollama.com.** The model list loads even with a
  wrong key. A wrong key shows up at verification as `authentication_error`.
- **Verification can be flaky for some models.** Verification runs a real
  model, so its result can vary. `gpt-oss:120b` passed most runs but about one
  in five failed the tool-result step (`capability_mismatch`), because it
  summarised the tool call without repeating its result. **Reverify** repeats
  the check; the check itself is not relaxed.

## Self-hosted Ollama

An Ollama on a public HTTPS address works with no extra setup: put its URL in
**Base URL**, and an API key only if your server checks one.

An Ollama on your own machine or local network (`localhost`, `127.0.0.1`,
`192.168.x.x`, `10.x.x.x`, `172.16–31.x.x`, `fc00::/7`) is refused by default.
A Flow Steward administrator can allow it with the server setting
`FS_ALLOW_PRIVATE_REMOTE_URLS=1` (in the installation's `.env`, then restart
Flow Steward). Until then, saving such a URL is rejected with a message that
names this setting. With the setting on:

- private and loopback addresses may use `http://` or `https://`;
- public addresses still need `https://`;
- link-local addresses, including the cloud metadata service `169.254.169.254`
  and `fe80::/10`, are refused anyway, as are CGNAT, multicast, reserved and
  unspecified addresses.

The setting is the host's decision. Flow Steward passes it to the extension as
`runtime_context.network_policy.allow_private_addresses`, and the extension never
reads host environment for it.

**`localhost` is the Flow Steward container, not your computer.** Flow Steward
(the Compact install in particular) runs in Docker, so `http://localhost:11434`
reaches the container itself, where no Ollama runs. To reach an Ollama on the
machine running Docker:

- Docker Desktop (macOS, Windows): use `http://host.docker.internal:11434`.
- Linux: add `extra_hosts: ["host.docker.internal:host-gateway"]` to the Flow
  Steward service, or use the host's LAN address, e.g. `http://192.168.1.20:11434`.
- Start Ollama so it listens beyond its own loopback: `OLLAMA_HOST=0.0.0.0 ollama serve`.

## Network safety

By default the base URL must be public HTTPS. `http://`, `localhost`,
`*.local`, URLs with credentials, and hosts that resolve to any loopback,
private, link-local, CGNAT, multicast or reserved address are refused,
including when only one of several DNS answers is unsafe. With private
addresses allowed, the rules above apply instead. Every name is resolved once
and each request is pinned to the address that was checked, so DNS cannot
rebind it elsewhere, and TLS still verifies the hostname. Redirects are never
followed, so the key cannot be forwarded elsewhere. Upstream bodies are capped
at 5 MiB.

## Errors

Every failure leaves the extension as one of these categories, never as
upstream text: `authentication_error`, `billing_quota_exceeded`,
`model_unavailable`, `rate_limited`, `timeout`, `capability_mismatch`,
`invalid_response`, `upstream_error`. Model discovery can be retried safely. A
failed chat is not retried automatically, because it may already have run
upstream.

## Files

| File | Purpose |
| --- | --- |
| `extension.yaml` | Manifest: subprocess entrypoint and the `llm_provider` contract |
| `main.py` | Subprocess entrypoint: reads the host request on stdin and dispatches `llm.list_models`, `llm.chat`, `llm.chat_structured` |
| `runtime.py` | Ollama HTTP client and the three operations |
| `health.py` | Health command run by the host |
| `fixtures/` | Canonical results of each operation; the tests compare against them |
| `ui/` | Extension page shown in Flow Steward |

### Host contract in one paragraph

The host runs `python3 main.py` with one JSON request on stdin.
`mode` is `query` or `action`, the operation is in `query.query_id` /
`action.action_id`, and its arguments are in `query.params` / `action.input`. The
provider settings arrive as runtime resources:
`runtime_context.resources.connection_config` and
`runtime_context.resources.provider_secrets`. The extension prints
`{"ok": true, "result": …}` and exits 0. On failure it prints
`{"ok": false, "error_code", "error", "errors": [{"code", "message"}]}` and exits
2, which the host treats as a provider error rather than a crash.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install pytest==8.3.5 PyYAML==6.0.3 dev-wheels/flowsteward_extension_sdk-*.whl
python -m pytest -q                                    # every handler, with mocked HTTP
python .github/scripts/package_extension.py            # dist/flowsteward.ollama-remote-<version>.zip
python .github/scripts/catalog_check.py dist/*.zip     # what the catalog would say
```

These are the same steps CI runs. No test touches the network. Each test
replaces `runtime._request_json`, or the socket layer for the transport tests.
To validate the bundle with Flow Steward itself:

```bash
flow-steward extensions validate flowsteward.ollama-remote --root <directory containing this bundle>
```

Releases: bump `version` in `extension.yaml`, then push a matching tag
(`v1.1.0`). The release workflow publishes the archive CI built.
