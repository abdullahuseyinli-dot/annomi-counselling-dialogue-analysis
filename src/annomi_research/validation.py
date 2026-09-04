from __future__ import annotations

from pathlib import Path
from typing import Any

from .ac_baselines import validate_ac_baseline_evidence
from .ac_data import build_task_a_examples, build_task_c_examples
from .baselines import validate_baseline_evidence
from .constants import (
    AC_PROTOCOL,
    FULL_DATA,
    FULL_MANIFEST,
    LABELS,
    MI_TAGS_EXTERNAL_PROTOCOL,
    PROTOCOL,
    QTRACE_CONFIG,
    RESEARCH_RESULTS,
    ROOT,
    SAFE_MI_CONFIG,
    SAFE_MI_EXTENSION_PROTOCOL,
    SAFE_MI_PROTOCOL,
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

    ac_protocol = read_json(AC_PROTOCOL)
    qtrace_config = read_json(QTRACE_CONFIG)
    if ac_protocol["status"] != "locked_before_task_ac_evaluation":
        raise ValueError("Task A/C protocol is not locked")
    if qtrace_config["status"] != "registered_before_qtrace_neural_evaluation":
        raise ValueError("Q-TRACE configuration is not registered")
    if qtrace_config["protocol_id"] != ac_protocol["protocol_id"]:
        raise ValueError("Task A/C protocol and Q-TRACE configuration disagree")
    task_a = build_task_a_examples(corpus, tuple(ac_protocol["task_a"]["therapist_turn_budgets"]))
    task_c = build_task_c_examples(
        corpus, int(ac_protocol["task_c"]["context_turns_for_flat_baseline"])
    )
    if task_a[task_a["checkpoint"].eq("full")]["transcript_id"].nunique() != 133:
        raise ValueError("Task A endpoint does not cover 133 transcripts")
    if len(task_c) != int(ac_protocol["task_c"]["expected_decisions"]):
        raise ValueError("Task C handoff count differs from registration")
    checks.append("Task A absolute prefixes and Task C strict handoffs")

    safe_protocol = read_json(SAFE_MI_PROTOCOL)
    safe_config = read_json(SAFE_MI_CONFIG)
    if safe_protocol["status"] != "registered_exploratory_after_qtrace_v1":
        raise ValueError("SAFE-MI protocol lost its exploratory designation")
    if safe_config["status"] != "registered_exploratory_before_safe_mi_execution":
        raise ValueError("SAFE-MI configuration is not registered")
    if safe_config["protocol_id"] != safe_protocol["protocol_id"]:
        raise ValueError("SAFE-MI protocol and configuration disagree")
    checks.append("SAFE-MI v2 is explicitly registered as post-Q-TRACE exploratory work")

    external_protocol = read_json(MI_TAGS_EXTERNAL_PROTOCOL)
    if external_protocol["status"] != "registered_before_mi_tags_sample_or_full_data_access":
        raise ValueError("MI-TAGS external protocol was not locked before data access")
    if external_protocol["sample_boundary"]["performance_claim_permitted"]:
        raise ValueError("MI-TAGS public samples cannot support a performance claim")
    if len(external_protocol["task_c_mapping"]) != 11:
        raise ValueError("MI-TAGS to AnnoMI mapping is incomplete")
    checks.append("MI-TAGS external protocol, mapping, and sample boundary")

    extension_protocol = read_json(SAFE_MI_EXTENSION_PROTOCOL)
    if extension_protocol["status"] != "registered_posthoc_after_safe_mi_v2_outcomes":
        raise ValueError("SAFE-MI v2.1 extension lost its posthoc designation")
    if extension_protocol["claim_boundary"]["confirmatory_claim_permitted"]:
        raise ValueError("SAFE-MI v2.1 cannot support a confirmatory claim")
    checks.append("SAFE-MI v2.1 posthoc extension and stopping rule")

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

    ac_baseline_dir = RESEARCH_RESULTS / "ac_v1" / "baselines"
    if ac_baseline_dir.exists():
        validate_ac_baseline_evidence(ac_baseline_dir)
        checks.append("Task A/C baseline metrics reconstruct from prediction ledgers")

    qtrace_dir = RESEARCH_RESULTS / "ac_v1" / "qtrace_mi"
    if qtrace_dir.exists():
        from .qtrace import validate_qtrace_evidence

        validate_qtrace_evidence(qtrace_dir)
        checks.append("Q-TRACE Task A/C metrics reconstruct from prediction ledgers")

    safe_dir = RESEARCH_RESULTS / "safe_mi_v2"
    if safe_dir.exists():
        from .safe_mi import validate_safe_mi_evidence

        validate_safe_mi_evidence(safe_dir)
        checks.append("SAFE-MI exploratory metrics reconstruct from prediction ledgers")

    safe_extension_dir = RESEARCH_RESULTS / "safe_mi_v2_1"
    if safe_extension_dir.exists():
        from .safe_mi_extension import validate_safe_mi_extension

        validate_safe_mi_extension(safe_extension_dir)
        checks.append("SAFE-MI v2.1 posthoc metrics reconstruct from prediction ledgers")

    from .mi_tags_external import validate_mi_tags_sample_audit

    validate_mi_tags_sample_audit(corpus)
    if (RESEARCH_RESULTS / "mi_tags_external_v1" / "sample_overlap_audit.json").exists():
        checks.append("MI-TAGS public-sample overlap evidence and claim boundary")

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
