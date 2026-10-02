"""Contract tests: the manifest, the entrypoint and the code agree.

Flow Steward reads ``extension.yaml``; these tests fail when the manifest
promises something the code does not do (or the other way round).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = yaml.safe_load((ROOT / "extension.yaml").read_text())
CONTRACT = MANIFEST["runtime"]["extension_contract_v2"]["llm_provider"]


def test_it_runs_as_an_ordinary_extension_subprocess() -> None:
    entrypoint = MANIFEST["entrypoint"]

    assert entrypoint["mode"] == "subprocess"
    assert entrypoint["command"] == ["python3", "main.py"]
    assert (ROOT / "main.py").is_file()
    assert CONTRACT["runtime_mode"] == "subprocess"
    # No separately hosted runtime: nothing to point at, nothing to protect.
    assert "remote" not in MANIFEST["runtime"]
    assert "remote" not in entrypoint


def test_it_asks_only_for_the_providers_own_settings() -> None:
    assert CONTRACT["required_provider_secrets"] == []
    assert CONTRACT["optional_provider_secrets"] == ["api_key"]
    assert CONTRACT["required_connection_config"] == []
    assert CONTRACT["optional_connection_config"] == ["upstream_base_url"]


def test_the_code_reads_exactly_the_declared_settings() -> None:
    source = (ROOT / "runtime.py").read_text()
    read_settings = set(re.findall(r'\.get\("(api_key|upstream_base_url|\w+_token)"\)', source))

    assert read_settings == {"api_key", "upstream_base_url"}


def test_every_bound_operation_has_a_handler(main) -> None:
    operations = CONTRACT["operations"]

    assert set(main.QUERIES) == {operations["list_models"]["query_id"]}
    assert set(main.ACTIONS) == {
        operations["chat"]["action_id"],
        operations["structured_chat"]["action_id"],
    }


def test_enablement_requires_the_capabilities_the_host_verifies() -> None:
    assert set(CONTRACT["required_capabilities_for_enablement"]) == {
        "chat",
        "tools",
        "structured_output",
    }
    assert CONTRACT["schema_version"] == "llm_provider_extension_v1"


def test_the_chat_timeout_outlasts_the_upstream_request(runtime) -> None:
    # The subprocess must not be killed while a 60 s upstream call is open.
    assert MANIFEST["entrypoint"]["timeout_seconds"] > 60


@pytest.mark.parametrize("path", ["health.py", "ui/ui_manifest.yaml", "README.md"])
def test_referenced_files_exist(path) -> None:
    assert (ROOT / path).is_file()


def test_ui_pages_listed_in_the_ui_manifest_exist() -> None:
    ui = yaml.safe_load((ROOT / "ui" / "ui_manifest.yaml").read_text())

    for page in ui["pages"]:
        assert (ROOT / "ui" / page["ref"]).is_file()


def test_version_is_semver() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", MANIFEST["version"])
