from __future__ import annotations

from pathlib import Path
from typing import Any

from .baselines import validate_baseline_evidence
from .constants import (
    FULL_DATA,
    FULL_MANIFEST,
    LABELS,
    PROTOCOL,
    RESEARCH_RESULTS,
    ROOT,
    SIMPLE_DATA,
    SIMPLE_MANIFEST,
)
from .data import Corpus
from .io import git_commit, read_json, sha256_file
from .splits import validate_source_folds


def legacy_inventory() -> dict[str, Any]:
    files: dict[str, str] = {}
    results_root = ROOT / "results"
    for path in sorted(results_root.rglob("*")):
        if not path.is_file() or RESEARCH_RESULTS in path.parents:
            continue
        files[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    return {
        "inventory_id": "portfolio-v0.1.0-development-consumed",
        "preserved_commit": "e3ff100b3866a283146e4af41596f5a837153818",
        "classification": "development_consumed",
        "files": files,
    }


def _validate_dataset(path: Path, manifest_path: Path) -> None:
    manifest = read_json(manifest_path)
    if sha256_file(path) != manifest["sha256"]:
        raise ValueError(f"Dataset hash mismatch: {path}")


def validate_research(
    corpus: Corpus,
    split_manifest_path: Path = RESEARCH_RESULTS / "gate0" / "source_folds_v1.json",
    inventory_path: Path = RESEARCH_RESULTS / "gate0" / "legacy_inventory.json",
) -> list[str]:
    checks: list[str] = []
    _validate_dataset(SIMPLE_DATA, SIMPLE_MANIFEST)
    _validate_dataset(FULL_DATA, FULL_MANIFEST)
    checks.append("pinned simple and full dataset hashes")

    protocol = read_json(PROTOCOL)
    if tuple(protocol["primary_task"]["labels"]) != LABELS:
        raise ValueError("Protocol label order differs from executable label order")
    if protocol["status"] != "locked_before_new_model_evaluation":
        raise ValueError("Research protocol is not locked")
    checks.append("machine-readable protocol and label order")

    split_manifest = read_json(split_manifest_path)
    validate_source_folds(corpus, split_manifest)
    checks.append("source-disjoint exhaustive outer folds")

    recorded_inventory = read_json(inventory_path)
    if recorded_inventory != legacy_inventory():
        raise ValueError("Legacy evidence inventory changed")
    checks.append("legacy development evidence remains byte-identical")

    baseline_dir = RESEARCH_RESULTS / "baseline_v1"
    if baseline_dir.exists():
        validate_baseline_evidence(baseline_dir)
        checks.append("baseline metrics reconstruct from row-level predictions")

    neural_root = RESEARCH_RESULTS / "neural_v1"
    if neural_root.exists():
        from .neural import validate_neural_evidence

        for result_dir in sorted(path.parent for path in neural_root.glob("*/summary.json")):
            if result_dir.name == "dash_mi":
                from .dash import validate_dash_evidence

                validate_dash_evidence(result_dir)
            else:
                validate_neural_evidence(result_dir)
            checks.append(f"neural metrics reconstruct for {result_dir.name}")

    comparison_root = RESEARCH_RESULTS / "comparisons"
    if comparison_root.exists():
        from .inference import validate_comparison_evidence

        for result_dir in sorted(path.parent for path in comparison_root.glob("*/summary.json")):
            validate_comparison_evidence(result_dir)
            checks.append(f"paired-source inference reconstructs for {result_dir.name}")

    panel_root = RESEARCH_RESULTS / "multiannotator_v1"
    if panel_root.exists():
        from .panel import validate_panel_evidence

        for result_dir in sorted(path.parent for path in panel_root.glob("*/summary.json")):
            validate_panel_evidence(result_dir)
            checks.append(f"multi-annotator evidence reconstructs for {result_dir.name}")
    return checks


def current_code_state() -> dict[str, str]:
    return {"git_commit": git_commit(ROOT)}
