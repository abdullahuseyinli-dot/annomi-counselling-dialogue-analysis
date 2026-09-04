from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _venv_python(environment: Path) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def main() -> None:
    parser = argparse.ArgumentParser(description="Install and smoke-test the built wheel")
    parser.add_argument("--dist-dir", type=Path, default=ROOT / "dist")
    args = parser.parse_args()

    wheels = sorted(args.dist_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("Expected exactly one wheel")
    wheel = wheels[0].resolve()
    expected_version = str(
        tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    )
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required for the isolated wheel smoke test")

    with tempfile.TemporaryDirectory(prefix="annomi-wheel-") as directory:
        environment = Path(directory) / "venv"
        subprocess.run(
            [uv, "venv", "--python", sys.executable, str(environment)],
            check=True,
            cwd=directory,
        )
        python = _venv_python(environment)
        subprocess.run(
            [uv, "pip", "install", "--python", str(python), str(wheel)],
            check=True,
            cwd=directory,
        )
        version = subprocess.run(
            [python, "-c", "import annomi_research; print(annomi_research.__version__)"],
            check=True,
            capture_output=True,
            text=True,
            cwd=directory,
            env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        ).stdout.strip()
        if version != expected_version:
            raise ValueError(f"Installed version mismatch: {version} != {expected_version}")
        subprocess.run(
            [python, "-m", "annomi_research", "--help"],
            check=True,
            capture_output=True,
            text=True,
            cwd=directory,
            env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        )

    print(f"PASS  isolated wheel import and command help ({expected_version})")


if __name__ == "__main__":
    main()
