from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_health_contract() -> None:
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parents[1] / "health.py")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == '{"status": "healthy"}'
