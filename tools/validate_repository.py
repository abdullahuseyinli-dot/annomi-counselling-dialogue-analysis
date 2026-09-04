from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tomllib
from html import unescape
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


REQUIRED = {
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "CITATION.cff",
    "MANIFEST.in",
    ".zenodo.json",
    ".gitleaks.toml",
    "pyproject.toml",
    "CHANGELOG.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    ".mailmap",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/research_change.yml",
    ".github/ISSUE_TEMPLATE/config.yml",
    ".github/dependabot.yml",
    ".github/workflows/ci.yml",
    ".github/workflows/evidence.yml",
    ".github/workflows/security.yml",
    "docs/README.md",
    "docs/BENCHMARK_CARD.md",
    "docs/DATA_CARD.md",
    "docs/REPRODUCIBILITY.md",
    "docs/ARTIFACTS.md",
    "docs/HARDWARE.md",
    "docs/PROJECT_STATUS.md",
    "docs/RELEASE_EVIDENCE_GATE.md",
    "docs/VERSIONING.md",
    "docs/LIMITATIONS.md",
    "docs/LITERATURE_MATRIX.md",
    "paper/README.md",
    "paper/OUTLINE.md",
    "paper/CLAIM_EVIDENCE_CROSSWALK.md",
    "paper/references.bib",
    "annomi_counselling_dialogue_analysis.ipynb",
    "experiments/pipeline_source.ipynb",
    "data/source_manifest.json",
    "results/provenance.json",
    "results/main/model_comparison.csv",
    "results/protocol/official_split.json",
    "results/research/publication_v1/manifest.json",
    "results/research/publication_v1/classification_summary.csv",
    "results/research/publication_v1/multiannotator_summary.csv",
    "results/research/publication_v1/registered_inference_summary.csv",
    "configs/research/protocol_ac_v1.json",
    "configs/research/qtrace_mi_v1.json",
    "docs/research/QTRACE_MI_REGISTRATION_V1.md",
    "docs/research/QTRACE_MI_V1_RESULT.md",
    "results/research/gate1/qtrace_mi_smoke_v1.json",
    "results/research/ac_v1/baselines/summary.json",
    "results/research/ac_v1/baselines/task_a_predictions.csv",
    "results/research/ac_v1/baselines/task_c_predictions.csv",
    "results/research/ac_v1/qtrace_mi/summary.json",
    "results/research/ac_v1/qtrace_mi/bootstrap_draws.csv",
    "results/research/ac_v1/qtrace_mi/calibration.json",
    "results/research/ac_v1/qtrace_mi/partitions.json",
    "results/research/ac_v1/qtrace_mi/selection.json",
    "results/research/ac_v1/qtrace_mi/task_a_predictions_by_seed.csv",
    "results/research/ac_v1/qtrace_mi/task_a_predictions_seed_ensemble.csv",
    "results/research/ac_v1/qtrace_mi/task_c_predictions_by_seed.csv",
    "results/research/ac_v1/qtrace_mi/task_c_predictions_seed_ensemble.csv",
    "results/research/publication_ac_v1/manifest.json",
    "results/research/publication_ac_v1/task_a_summary.csv",
    "results/research/publication_ac_v1/task_c_summary.csv",
    "results/research/publication_ac_v1/registered_inference_summary.csv",
    "results/research/publication_ac_v1/prediction_set_summary.csv",
    "configs/research/protocol_safe_mi_v2.json",
    "configs/research/safe_mi_v2.json",
    "configs/research/protocol_safe_mi_v2_1.json",
    "configs/research/protocol_mi_tags_external_v1.json",
    "docs/research/SAFE_MI_V2_RESULT.md",
    "results/research/gate1/safe_mi_smoke_v2.json",
    "results/research/safe_mi_v2/summary.json",
    "results/research/safe_mi_v2/bootstrap_draws.csv",
    "results/research/safe_mi_v2/selection.json",
    "results/research/safe_mi_v2_1/summary.json",
    "results/research/safe_mi_v2_1/bootstrap_draws.csv",
    "results/research/safe_mi_v2_1/task_c_crossfit_prediction_sets.csv",
    "results/research/mi_tags_external_v1/sample_overlap_audit.json",
    "results/research/publication_safe_mi_v2/manifest.json",
    "results/research/publication_safe_mi_v2/task_a_t10_summary.csv",
    "results/research/publication_safe_mi_v2/task_c_summary.csv",
    "results/research/publication_safe_mi_v2/screen_summary.csv",
    "results/research/publication_safe_mi_v2/paired_inference_summary.csv",
    "results/research/publication_safe_mi_v2/prediction_set_summary.csv",
    "results/research/publication_safe_mi_v2/external_overlap_summary.csv",
    "assets/research/research_overview.png",
    "assets/research/research_overview.svg",
    "assets/research/registered_effect_intervals.png",
    "assets/research/registered_effect_intervals.svg",
    "assets/research/qtrace_ac_results.png",
    "assets/research/qtrace_ac_results.svg",
    "assets/research/qtrace_ac_intervals.png",
    "assets/research/qtrace_ac_intervals.svg",
    "assets/research/safe_mi_results.png",
    "assets/research/safe_mi_results.svg",
    "assets/research/safe_mi_effect_intervals.png",
    "assets/research/safe_mi_effect_intervals.svg",
    "tools/build_ac_assets.py",
    "tools/build_safe_mi_assets.py",
    "tools/validate_distribution.py",
    "tools/smoke_install_distribution.py",
}
TEXT_SUFFIXES = {
    ".bib",
    ".cff",
    ".csv",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
FORBIDDEN = {
    "machine-specific user path": re.compile(r"[A-Za-z]:\\Users\\", re.IGNORECASE),
    "local file URI": re.compile("file:" + "/" * 2, re.IGNORECASE),
    "unresolved merge marker": re.compile(r"^(?:<{7}|={7}|>{7})(?: |$)", re.MULTILINE),
    "unresolved editorial marker": re.compile(r"\b(?:TODO|FIXME|TBD|lorem ipsum)\b", re.IGNORECASE),
}
HEAVY_SUFFIXES = {".joblib", ".pkl", ".pt", ".pth", ".ckpt", ".safetensors", ".npy", ".npz"}
MAX_TRACKED_BYTES = 10 * 1024 * 1024
LARGE_EVIDENCE_SHA256 = {
    "results/research/neural_v1/dash_mi/predictions_by_seed.csv": (
        "f503530b1048206ff9ba15fabaffe944f858dda0e3006e4694e92f629ceb5a86"
    ),
    "results/research/ac_v1/qtrace_mi/task_c_predictions_by_seed.csv": (
        "815dbc04ba88f3cd8702eaae8718fae3372e44df6fd0b5df65198957575b5709"
    ),
    "results/research/safe_mi_v2/final_task_c_predictions_by_seed.csv": (
        "4586ccc631dec2eb419753b680365c10f7c662e875096c72337453a4e629c602"
    ),
    "results/research/safe_mi_v2/screen_task_c_predictions_by_seed.csv": (
        "58ba4bf6a82a9331ed3f8ba0b94c6b4cde4112aa2d14924bd6363f651040c374"
    ),
    "results/research/safe_mi_v2/screen_task_c_predictions_seed_ensemble.csv": (
        "0adc5787ad0028eb86ea428a42215bb382dd6e57185e2180d2c2bd41626a174f"
    ),
    "results/research/safe_mi_v2_1/task_c_predictions_by_seed.csv": (
        "1ce27bef109a42853a30f1e51d835899f427b07e6bb55451194e03e591b51801"
    ),
}


def repository_files() -> list[Path]:
    if (ROOT / ".git").exists():
        result = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
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


def validate_links(files: list[Path]) -> str:
    missing: list[str] = []
    markdown_files = [path for path in files if path.suffix.lower() == ".md"]
    for path in markdown_files:
        content = path.read_text(encoding="utf-8-sig")
        targets = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", content)
        for raw_target in targets:
            target = unescape(raw_target.strip())
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1]
            else:
                target = target.split(maxsplit=1)[0]
            target = unquote(target).split("#", 1)[0]
            if not target or re.match(r"(?:https?|mailto):", target, re.IGNORECASE):
                continue
            candidate = (
                ROOT / target.lstrip("/") if target.startswith("/") else path.parent / target
            )
            if not candidate.exists():
                rel = path.relative_to(ROOT).as_posix()
                missing.append(f"{rel} -> {target}")
    if missing:
        raise ValueError(f"Markdown has missing local links: {missing}")
    return f"local links resolve across {len(markdown_files)} Markdown files"


def validate_release_metadata() -> str:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    expected = str(project["version"])

    package_text = (ROOT / "src" / "annomi_research" / "__init__.py").read_text(encoding="utf-8")
    fallback_match = re.search(
        r'^\s*__version__\s*=\s*["\']([^"\']+)["\']', package_text, re.MULTILINE
    )
    if fallback_match is None or fallback_match.group(1) != expected:
        raise ValueError("Package fallback version does not match pyproject.toml")

    citation_text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    citation_match = re.search(r"^version:\s*[\"']?([^\"'\s]+)", citation_text, re.MULTILINE)
    if citation_match is None or citation_match.group(1) != expected:
        raise ValueError("CITATION.cff version does not match pyproject.toml")
    if ".dev" in expected and re.search(r"^date-released:", citation_text, re.MULTILINE):
        raise ValueError("Development citation metadata must not claim a release date")

    zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    if str(zenodo.get("version")) != expected:
        raise ValueError("Zenodo metadata version does not match pyproject.toml")
    if zenodo.get("doi"):
        raise ValueError("Preparatory Zenodo metadata must not claim an unissued DOI")
    return f"development release metadata is consistent at version {expected}"


def validate_research_publication_assets() -> list[str]:
    publication_root = ROOT / "results" / "research" / "publication_v1"
    manifest = json.loads((publication_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("manifest_id") != "annomi-research-publication-assets-v1":
        raise ValueError("Unexpected research-publication manifest ID")
    for relative, expected in manifest["source_sha256"].items():
        path = ROOT / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Research-publication source hash mismatch: {relative}")
    for filename, expected in manifest["table_sha256"].items():
        path = publication_root / filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Research-publication table hash mismatch: {filename}")
    for stem, formats in manifest["figure_sha256"].items():
        for extension, expected in formats.items():
            path = ROOT / "assets" / "research" / f"{stem}.{extension}"
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Research-publication figure hash mismatch: {path.name}")
    builder = ROOT / "tools" / "build_research_assets.py"
    if hashlib.sha256(builder.read_bytes()).hexdigest() != manifest["builder_sha256"]:
        raise ValueError("Research-publication builder hash mismatch")
    return [
        "research publication inputs and tables match manifest",
        "research publication figures and builder match manifest",
    ]


def validate_ac_publication_assets() -> list[str]:
    publication_root = ROOT / "results" / "research" / "publication_ac_v1"
    manifest = json.loads((publication_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("manifest_id") != "annomi-qtrace-ac-publication-assets-v1":
        raise ValueError("Unexpected Task A/C publication manifest ID")
    for relative, expected in manifest["source_sha256"].items():
        path = ROOT / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Task A/C publication source hash mismatch: {relative}")
    for filename, expected in manifest["table_sha256"].items():
        path = publication_root / filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Task A/C publication table hash mismatch: {filename}")
    for stem, formats in manifest["figure_sha256"].items():
        for extension, expected in formats.items():
            path = ROOT / "assets" / "research" / f"{stem}.{extension}"
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Task A/C publication figure hash mismatch: {path.name}")
    builder = ROOT / "tools" / "build_ac_assets.py"
    if hashlib.sha256(builder.read_bytes()).hexdigest() != manifest["builder_sha256"]:
        raise ValueError("Task A/C publication builder hash mismatch")
    return [
        "Task A/C publication inputs and tables match manifest",
        "Task A/C publication figures and builder match manifest",
    ]


def validate_safe_mi_publication_assets() -> list[str]:
    publication_root = ROOT / "results" / "research" / "publication_safe_mi_v2"
    manifest = json.loads((publication_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("manifest_id") != "annomi-safe-mi-publication-assets-v2":
        raise ValueError("Unexpected SAFE-MI publication manifest ID")
    for relative, expected in manifest["source_sha256"].items():
        path = ROOT / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"SAFE-MI publication source hash mismatch: {relative}")
    for filename, expected in manifest["table_sha256"].items():
        path = publication_root / filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"SAFE-MI publication table hash mismatch: {filename}")
    for stem, formats in manifest["figure_sha256"].items():
        for extension, expected in formats.items():
            path = ROOT / "assets" / "research" / f"{stem}.{extension}"
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"SAFE-MI publication figure hash mismatch: {path.name}")
    builder = ROOT / "tools" / "build_safe_mi_assets.py"
    if hashlib.sha256(builder.read_bytes()).hexdigest() != manifest["builder_sha256"]:
        raise ValueError("SAFE-MI publication builder hash mismatch")
    return [
        "SAFE-MI publication inputs and tables match manifest",
        "SAFE-MI publication figures and builder match manifest",
    ]


def main() -> None:
    from annomi_portfolio.evidence import validate_evidence

    files = repository_files()
    relative = {path.relative_to(ROOT).as_posix() for path in files}
    missing = sorted(REQUIRED - relative)
    if missing:
        raise ValueError(f"Missing required files: {missing}")

    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if path.stat().st_size > MAX_TRACKED_BYTES:
            expected_hash = LARGE_EVIDENCE_SHA256.get(rel)
            if expected_hash is None:
                raise ValueError(f"Tracked file exceeds 10 MiB: {rel}")
            observed_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if observed_hash != expected_hash:
                raise ValueError(f"Large evidence hash mismatch: {rel}")
        if path.suffix.lower() in HEAVY_SUFFIXES:
            raise ValueError(f"Model or binary experiment artifact is tracked: {rel}")
        if rel.startswith("data/raw/"):
            raise ValueError(f"Raw dataset file is tracked: {rel}")
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {
            ".gitattributes",
            ".gitignore",
            ".mailmap",
        }:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            for label, pattern in FORBIDDEN.items():
                if rel == "tools/validate_repository.py" and label == "unresolved editorial marker":
                    continue
                if pattern.search(text):
                    raise ValueError(f"Found {label} in {rel}")

    checks = validate_evidence(ROOT)
    checks.extend(validate_notebooks())
    checks.append(validate_links(files))
    checks.append(validate_release_metadata())
    checks.extend(validate_research_publication_assets())
    checks.extend(validate_ac_publication_assets())
    checks.extend(validate_safe_mi_publication_assets())
    checks.append(f"{len(files)} repository files passed hygiene checks")
    for check in checks:
        print(f"PASS  {check}")


if __name__ == "__main__":
    main()
