from __future__ import annotations

import argparse
import tarfile
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def _project() -> dict[str, object]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def _safe_members(names: list[str], archive: str) -> None:
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe path in {archive}: {name}")


def validate_wheel(path: Path, project: dict[str, object]) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        _safe_members(names, path.name)
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        entry_point_names = [name for name in names if name.endswith(".dist-info/entry_points.txt")]
        if len(metadata_names) != 1 or len(entry_point_names) != 1:
            raise ValueError("Wheel must contain one metadata file and one entry-point file")

        metadata = Parser().parsestr(archive.read(metadata_names[0]).decode("utf-8"))
        if metadata["Name"] != project["name"]:
            raise ValueError("Wheel project name does not match pyproject.toml")
        if metadata["Version"] != project["version"]:
            raise ValueError("Wheel version does not match pyproject.toml")

        entry_points = archive.read(entry_point_names[0]).decode("utf-8")
        expected_entry = "annomi-research = annomi_research.cli:main"
        if expected_entry not in entry_points:
            raise ValueError("Wheel does not expose the expected command")

        dist_info = metadata_names[0].split("/", 1)[0]
        allowed = ("annomi_research/", "annomi_portfolio/", f"{dist_info}/")
        unexpected = [name for name in names if not name.startswith(allowed)]
        if unexpected:
            raise ValueError(f"Unexpected wheel contents: {unexpected[:5]}")


def validate_sdist(path: Path) -> None:
    with tarfile.open(path, "r:gz") as archive:
        names = archive.getnames()
        _safe_members(names, path.name)
        roots = {PurePosixPath(name).parts[0] for name in names if PurePosixPath(name).parts}
        if len(roots) != 1:
            raise ValueError("Source distribution must have one top-level directory")
        root = next(iter(roots))
        required = {
            f"{root}/CHANGELOG.md",
            f"{root}/CITATION.cff",
            f"{root}/CONTRIBUTING.md",
            f"{root}/LICENSE",
            f"{root}/README.md",
            f"{root}/SECURITY.md",
            f"{root}/THIRD_PARTY_NOTICES.md",
            f"{root}/pyproject.toml",
        }
        missing = sorted(required - set(names))
        if missing:
            raise ValueError(f"Source distribution is missing release documents: {missing}")
        forbidden = {"artifacts", "data", "results", "checkpoints", "models", "outputs"}
        for name in names:
            parts = PurePosixPath(name).parts
            if len(parts) > 1 and parts[1] in forbidden:
                raise ValueError(f"Research evidence leaked into source distribution: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the built package boundary")
    parser.add_argument("--dist-dir", type=Path, default=ROOT / "dist")
    args = parser.parse_args()

    wheels = sorted(args.dist_dir.glob("*.whl"))
    sdists = sorted(args.dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("Expected exactly one wheel and one source distribution")

    project = _project()
    validate_wheel(wheels[0], project)
    validate_sdist(sdists[0])
    print(f"PASS  wheel boundary: {wheels[0].name}")
    print(f"PASS  source-distribution boundary: {sdists[0].name}")


if __name__ == "__main__":
    main()
