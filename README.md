# Remote Ollama LLM Provider — reference extension

Status: active reference bundle. It includes a runnable stdlib remote runtime,
canonical wire handlers, fixtures, and contract tests. Hosting the runtime is
an operator responsibility and is intentionally outside this bundle.

## Install boundary

The manifest deliberately contains no default runtime URL. Installation requires
an explicit public HTTPS `remote_base_url` for the extension runtime. The
account-scoped public HTTPS Ollama `upstream_base_url` is passed as extension
configuration; a bearer token is optional and uses the existing encrypted
`secret_refs` flow. Private IPs, LAN/localhost, `.local`, and HTTP endpoints
are intentionally unsupported.

The bundled stdlib process is an origin server and binds to `127.0.0.1:8090`
by default. Production hosting must place it behind a reverse proxy or load
balancer that terminates TLS and exposes the configured public HTTPS
`remote_base_url`. Set `FS_EXTENSION_BIND_HOST` only for an explicitly isolated
deployment network; never expose the plain HTTP origin directly.

The reference runtime must be started with `FS_EXTENSION_TOKEN`; configure the
matching required `runtime_bearer_token` through the same encrypted credentials
flow. It authenticates host-to-runtime calls only and is never forwarded to
Ollama. The upstream Ollama `bearer_token` remains optional.

A remote HTTPS LLM provider extension that fronts a **remote** Ollama endpoint
and exposes its models to the Flow Steward model catalog through the
`llm_provider` extension contract (schema `llm_provider_extension_v1`).

## Contract

- **supplier_key / supplier_label**: `ollama_remote` / `Ollama`
- **runtime_mode**: `remote_https`. Local/private/non-HTTPS Ollama endpoints are
  out of scope for v1 and are rejected by the platform URL-safety policy
  (`assert_safe_remote_http_url`) — only public HTTPS runtime base URLs are
  accepted.
- **credentials**: account-scoped, via the existing provider-secret flow;
  `runtime_bearer_token` is required and upstream `bearer_token` is optional.
- **operations**: `llm.list_models`, `llm.chat`, `llm.chat_structured`.
- **action idempotency**: chat actions require the host `idempotency_key`. The
  single-process reference runtime coalesces concurrent calls and retains the
  completed result or safe error for five minutes in a bounded in-memory cache,
  preventing a transport retry from duplicating an accepted upstream request
  while that runtime process remains alive.
- **model_grouping**: `endpoint` — all models from one connection form a single
  provider group labelled `"Ollama: <connection display name>"` (the install's
  display name), composed from descriptor metadata.

## Naming

`model_id` is preserved exactly as the Ollama endpoint returns it:

| `model_id` (preserved exactly) | Provider group label             |
| ------------------------------ | -------------------------------- |
| `llama3.3:70b`                 | `Ollama: <connection display name>` |
| `qwen2.5-coder:32b`            | `Ollama: <connection display name>` |

## Fixtures

`fixtures/` contains canonical `llm.list_models`, `llm.chat`, and
`llm.chat_structured` responses, exercised by
`tests/unit/test_llm_provider_extensions.py`.
