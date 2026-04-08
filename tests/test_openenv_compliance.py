from __future__ import annotations

import subprocess
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_server_script_entrypoint_targets_main() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))
    scripts = pyproject["project"]["scripts"]
    assert scripts["server"].endswith(":main")


def test_root_server_app_exposes_callable_main() -> None:
    app_text = (REPO_ROOT / "server" / "app.py").read_text("utf-8")
    assert "def main(" in app_text
    assert "if __name__ == \"__main__\":" in app_text
    assert "main(" in app_text


def test_openenv_local_validate_passes() -> None:
    result = subprocess.run(
        ["openenv", "validate"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "openenv validate failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}\n"
    )
