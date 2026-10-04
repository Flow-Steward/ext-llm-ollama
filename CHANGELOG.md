# Changelog

## 1.1.2

- Accepts a Base URL pasted with an Ollama API path (`/api`, `/api/chat`,
  `/api/tags`, `/api/show`, `/api/generate`, `/v1`, `/v1/chat/completions`):
  the path is stripped and the server address is used.
- Checks the API key before listing models: **Discover models** fails with
  "authentication failed" when Ollama rejects the key (for example, only the
  part before the dot was copied), instead of loading the public model list and
  failing every model at verification. Uses `POST /api/me`, which runs no model;
  a self-hosted server without that endpoint is not affected.
- README: "Which Base URL to enter", with the right value for each setup and
  the endpoint addresses that are not a Base URL.

## 1.1.1

- Declares the default Base URL, `https://ollama.com`, as
  `connection_config_defaults`. Flow Steward compares effective settings, so
  emptying the Base URL while it holds the default (or saving the same value
  again) no longer makes verified models stale.
- Needs extension host contract 1.2.0 (`platform_min: 1.2.0`), the first that
  reads `connection_config_defaults`.

## 1.1.0

- Runs as an ordinary Flow Steward extension subprocess and asks only for
  Ollama's own settings: an optional API key and an optional Base URL.
- Needs extension host contract 1.1.0.
