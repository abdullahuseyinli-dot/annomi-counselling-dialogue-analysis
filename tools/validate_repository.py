from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from annomi_portfolio.evidence import validate_evidence  # noqa: E402


REQUIRED = {
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    "annomi_counselling_dialogue_analysis.ipynb",
    "experiments/pipeline_source.ipynb",
    "data/source_manifest.json",
    "results/provenance.json",
    "results/main/model_comparison.csv",
    "results/protocol/official_split.json",
}
TEXT_SUFFIXES = {".md", ".py", ".toml", ".yml", ".yaml", ".json", ".csv", ".txt", ".ipynb"}
release_hygiene_terms = [
    "assess" + "ment",
    "course" + "work",
    r"student\s+" + "number",
    r"assign" + r"ment\s+sub" + "mission",
]
attribution_terms = [
    "chat" + "gpt",
    r"as\s+an\s+" + "ai",
    r"generated\s+by\s+(?:an?\s+)?" + "ai",
    "copi" + "lot",
    "clau" + "de",
    "gem" + "ini",
]
FORBIDDEN = {
    "machine-specific user path": re.compile(r"[A-Za-z]:\\Users\\", re.IGNORECASE),
    "local file URI": re.compile("file:" + "/" * 2, re.IGNORECASE),
    "course identifier": re.compile(r"\bcs\s*552j\b", re.IGNORECASE),
    "personal identifier": re.compile(r"\b" + "5253" + "3844" + r"\b"),
    "academic framing": re.compile(r"\b(" + "|".join(release_hygiene_terms) + r")\b", re.IGNORECASE),
    "assistant attribution": re.compile(
        r"\b(" + "|".join(attribution_terms) + r")\b", re.IGNORECASE
    ),
}
HEAVY_SUFFIXES = {".joblib", ".pkl", ".pt", ".pth", ".ckpt", ".safetensors", ".npy", ".npz"}


def repository_files() -> list[Path]:
    if (ROOT / ".git").exists():
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        return [ROOT / raw.decode() for raw in result.stdout.split(b"\0") if raw]
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
        and ".ruff_cache" not in path.parts
    ]


def validate_notebooks() -> list[str]:
    portfolio = json.loads(
        (ROOT / "annomi_counselling_dialogue_analysis.ipynb").read_text(encoding="utf-8")
    )
    pipeline = json.loads(
        (ROOT / "experiments" / "pipeline_source.ipynb").read_text(encoding="utf-8")
    )
    for name, notebook in (("portfolio", portfolio), ("pipeline", pipeline)):
        if notebook.get("nbformat") != 4:
            raise ValueError(f"{name} notebook does not use nbformat 4")
        if notebook.get("metadata", {}).get("kernelspec", {}).get("name") != "python3":
            raise ValueError(f"{name} notebook does not use the portable python3 kernel")

    portfolio_code = [cell for cell in portfolio["cells"] if cell["cell_type"] == "code"]
    if not portfolio_code or any(cell["execution_count"] is None for cell in portfolio_code):
        raise ValueError("Portfolio notebook is not fully executed")
    if any(
        output.get("output_type") == "error"
        for cell in portfolio_code
        for output in cell.get("outputs", [])
    ):
        raise ValueError("Portfolio notebook contains an error output")
    if any(cell["cell_type"] != "code" for cell in pipeline["cells"]):
        raise ValueError("Full pipeline must remain code-only")
    if any(cell["execution_count"] is not None for cell in pipeline["cells"]):
        raise ValueError("Full pipeline must not contain execution counts")
    if any(cell.get("outputs") for cell in pipeline["cells"]):
        raise ValueError("Full pipeline must not contain outputs")
    return ["portfolio notebook is executed and error-free", "full pipeline is code-only and clean"]


def validate_links() -> str:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", readme)
    missing = []
    for target in targets:
        target = target.strip().split("#", 1)[0]
        if not target or re.match(r"(?:https?|mailto):", target):
            continue
        if not (ROOT / target).exists():
            missing.append(target)
    if missing:
        raise ValueError(f"README has missing local links: {missing}")
    return "README local links resolve"


def main() -> None:
    files = repository_files()
    relative = {path.relative_to(ROOT).as_posix() for path in files}
    missing = sorted(REQUIRED - relative)
    if missing:
        raise ValueError(f"Missing required files: {missing}")

    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if path.stat().st_size > 10 * 1024 * 1024:
            raise ValueError(f"Tracked file exceeds 10 MiB: {rel}")
        if path.suffix.lower() in HEAVY_SUFFIXES:
            raise ValueError(f"Model or binary experiment artifact is tracked: {rel}")
        if rel.startswith("data/raw/"):
            raise ValueError(f"Raw dataset file is tracked: {rel}")
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {".gitignore", ".gitattributes"}:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            for label, pattern in FORBIDDEN.items():
                if pattern.search(text):
                    raise ValueError(f"Found {label} in {rel}")

    checks = validate_evidence(ROOT)
    checks.extend(validate_notebooks())
    checks.append(validate_links())
    checks.append(f"{len(files)} repository files passed hygiene checks")
    for check in checks:
        print(f"PASS  {check}")


if __name__ == "__main__":
    main()
