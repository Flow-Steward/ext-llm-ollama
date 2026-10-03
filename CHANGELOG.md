# Changelog

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
