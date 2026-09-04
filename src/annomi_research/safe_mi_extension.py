from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ac_data import build_session_turns
from .ac_metrics import (
    add_prediction_sets,
    aps_scores,
    evaluate_action_predictions,
    evaluate_prediction_sets,
    evaluate_quality_predictions,
    source_crc_threshold,
)
from .constants import (
    LABELS,
    RESEARCH_RESULTS,
    ROOT,
    SAFE_MI_CONFIG,
    SAFE_MI_EXTENSION_PROTOCOL,
)
from .data import Corpus
from .io import canonical_json_bytes, git_commit, read_json, sha256_file, write_create_only
from .qtrace import _csv_payload, _git_is_clean, extract_turn_embeddings
from .safe_mi import (
    SafeFitResult,
    _aggregate_results,
    _execute_modes,
    _load_fit_cache,
    _paired_bootstrap,
    _partitions_by_fold,
    _seed_deltas,
)
from .safe_mi_model import SafeMIMode, mode_from_config
from .splits import fold_lookup, validate_source_folds

SAFE_MI_EXTENSION_RESULTS = RESEARCH_RESULTS / "safe_mi_v2_1"


def _base_cache_directory(
    summary: dict[str, Any],
    config_sha256: str,
    split_sha256: str,
) -> Path:
    directory = (
        ROOT
        / "artifacts"
        / "safe_mi_v2"
        / (f"runs_{summary['code_commit'][:12]}_{config_sha256[:12]}_{split_sha256[:12]}")
    )
    if not directory.exists():
        raise FileNotFoundError(f"Missing registered SAFE-MI v2 fit cache: {directory}")
    return directory


def _extension_cache_directory(config_sha256: str, split_sha256: str) -> Path:
    return (
        ROOT
        / "artifacts"
        / "safe_mi_v2_1"
        / (f"runs_{git_commit(ROOT)[:12]}_{config_sha256[:12]}_{split_sha256[:12]}")
    )


def _load_registered_fits(
    cache_directory: Path,
    modes: dict[str, SafeMIMode],
    folds: list[int],
    seeds: list[int],
    screen_seed: int,
) -> dict[tuple[str, int, int], SafeFitResult]:
    registry: dict[tuple[str, int, int], SafeFitResult] = {}
    for fold in folds:
        for seed in seeds:
            for name in ("c0_frozen_gru", "c1_adapted_gru"):
                result = _load_fit_cache(cache_directory, modes[name], fold, seed)
                if result is None:
                    raise FileNotFoundError(
                        f"Missing registered fit cache for {name}/fold={fold}/seed={seed}"
                    )
                registry[(name, fold, seed)] = result
        result = _load_fit_cache(cache_directory, modes["m2_oneway"], fold, screen_seed)
        if result is None:
            raise FileNotFoundError(
                f"Missing registered m2 screen fit for fold={fold}/seed={screen_seed}"
            )
        registry[("m2_oneway", fold, screen_seed)] = result
    return registry


def crossfit_prediction_sets(
    predictions: pd.DataFrame,
    alpha: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Calibrate each outer fold from all other out-of-fold source predictions.

    This is a post-hoc cross-fitted sensitivity analysis.  Every calibration row
    was itself predicted by a model that excluded its source, and the target fold
    never supplies labels to its own threshold.
    """

    probability_columns = [f"prob_{label}" for label in LABELS]
    outputs: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    drop_columns = [
        "prediction_set",
        "prediction_set_size",
        "set_covered",
        "prediction_set_threshold",
    ]
    for (model, fold), test_fold in predictions.groupby(["model", "outer_fold"], sort=True):
        calibration = predictions[
            predictions["model"].eq(model) & ~predictions["outer_fold"].eq(fold)
        ].copy()
        if set(calibration["source_id"].astype(str)) & set(test_fold["source_id"].astype(str)):
            raise ValueError("Cross-fitted prediction-set sources overlap")
        probabilities = calibration[probability_columns].to_numpy(dtype=float)
        scores = aps_scores(probabilities)
        targets = np.asarray(
            [LABELS.index(str(value)) for value in calibration["label"]], dtype=int
        )
        risk = source_crc_threshold(
            scores[np.arange(len(scores)), targets],
            calibration["source_id"].astype(str).to_numpy(),
            alpha,
        )
        values = test_fold.drop(columns=drop_columns, errors="ignore").copy()
        values["prediction_set_threshold"] = float(risk["threshold"])
        values = add_prediction_sets(values, float(risk["threshold"]))
        values["prediction_set_method"] = "outer-crossfit-source-crc"
        outputs.append(values)
        records.append(
            {
                "model": str(model),
                "outer_fold": int(fold),
                "calibration_outer_folds": sorted(
                    int(value) for value in calibration["outer_fold"].unique()
                ),
                "calibration_rows": len(calibration),
                **risk,
            }
        )
    return pd.concat(outputs, ignore_index=True), records


def _posthoc_gates(
    aggregate: dict[str, Any],
    baseline_a_metrics: dict[str, Any],
    seed_deltas: dict[str, Any],
    crossfit_metrics: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    gates = protocol["descriptive_gates"]
    baseline_c = aggregate["task_c_metrics"]["c0_frozen_gru"]
    result: dict[str, Any] = {}
    for model in ("c1_adapted_gru", "m2_oneway"):
        candidate_c = aggregate["task_c_metrics"][model]
        c_f1_delta = float(
            candidate_c["source_balanced_macro_f1"] - baseline_c["source_balanced_macro_f1"]
        )
        c_brier_delta = float(
            candidate_c["source_balanced_brier"] - baseline_c["source_balanced_brier"]
        )
        components: dict[str, bool] = {
            "task_c_noninferiority": c_f1_delta >= float(gates["task_c_noninferiority_margin"]),
            "task_c_brier": c_brier_delta <= float(gates["task_c_maximum_brier_degradation"]),
            "crossfit_prediction_set_coverage": float(
                crossfit_metrics[model]["source_balanced_coverage"]
            )
            >= float(gates["crossfit_prediction_set_minimum_coverage"]),
            "crossfit_prediction_set_efficiency": float(
                crossfit_metrics[model]["source_balanced_mean_set_size"]
            )
            <= float(gates["crossfit_prediction_set_maximum_mean_size"]),
        }
        record: dict[str, Any] = {
            "model": model,
            "posthoc_not_confirmatory": True,
            "task_c_macro_f1_delta": c_f1_delta,
            "task_c_brier_delta": c_brier_delta,
        }
        if model == "m2_oneway":
            candidate_a = aggregate["task_a_metrics"][model]["t10"]
            baseline_a = baseline_a_metrics["t10"]
            a_gain = float(
                candidate_a["source_balanced_balanced_accuracy"]
                - baseline_a["source_balanced_balanced_accuracy"]
            )
            a_brier = float(
                candidate_a["source_balanced_brier"] - baseline_a["source_balanced_brier"]
            )
            components.update(
                {
                    "task_a_minimum_gain": a_gain
                    >= float(gates["task_a_minimum_t10_balanced_accuracy_gain"]),
                    "task_a_brier": a_brier <= float(gates["task_a_maximum_t10_brier_degradation"]),
                    "task_a_positive_seeds": int(seed_deltas[model]["task_a_positive_seed_count"])
                    >= int(gates["task_a_minimum_positive_seed_count"]),
                }
            )
            record["task_a_t10_balanced_accuracy_delta"] = a_gain
            record["task_a_t10_brier_delta"] = a_brier
        record["components"] = components
        record["pass"] = all(components.values())
        result[model] = record
    return result


def run_safe_mi_extension(
    corpus: Corpus,
    split_manifest: dict[str, Any],
    output_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir = output_dir or SAFE_MI_EXTENSION_RESULTS
    if (output_dir / "summary.json").exists():
        validate_safe_mi_extension(output_dir)
        return read_json(output_dir / "summary.json")
    if not _git_is_clean():
        raise RuntimeError("Commit the SAFE-MI v2.1 protocol/code before generating evidence")
    started = time.perf_counter()
    protocol = read_json(SAFE_MI_EXTENSION_PROTOCOL)
    config = read_json(SAFE_MI_CONFIG)
    if protocol["status"] != "registered_posthoc_after_safe_mi_v2_outcomes":
        raise ValueError("SAFE-MI v2.1 is not explicitly posthoc")
    validate_source_folds(corpus, split_manifest)
    base_summary = read_json(RESEARCH_RESULTS / "safe_mi_v2" / "summary.json")
    config_sha256 = sha256_file(SAFE_MI_CONFIG)
    split_sha256 = split_manifest["manifest_sha256"]
    base_cache = _base_cache_directory(base_summary, config_sha256, split_sha256)
    modes = {value["model"]: mode_from_config(config, value["model"]) for value in config["models"]}
    sessions = build_session_turns(corpus, (3, 5, 10, 20))
    partitions, partition_records = _partitions_by_fold(
        sessions,
        fold_lookup(split_manifest),
        config,
        len(split_manifest["folds"]),
    )
    seeds = [int(value) for value in protocol["frozen_inputs"]["final_seeds"]]
    screen_seed = int(protocol["frozen_inputs"]["screen_seed"])
    registry = _load_registered_fits(
        base_cache,
        modes,
        sorted(partitions),
        seeds,
        screen_seed,
    )
    extension_selections: list[dict[str, Any]] = []
    frozen_embeddings = extract_turn_embeddings(corpus, config)
    extension_cache = _extension_cache_directory(config_sha256, split_sha256)
    missing_seeds = [int(value) for value in protocol["execution"]["run_missing_m2_seeds"]]
    _execute_modes(
        corpus,
        partitions,
        [modes["m2_oneway"]],
        missing_seeds,
        frozen_embeddings,
        config,
        extension_cache,
        registry,
        extension_selections,
    )
    model_names = {"c0_frozen_gru", "c1_adapted_gru", "m2_oneway"}
    aggregate = _aggregate_results(
        registry,
        model_names,
        float(config["calibration"]["prediction_set_alpha"]),
    )
    crossfit, crossfit_records = crossfit_prediction_sets(
        aggregate["task_c"],
        float(protocol["execution"]["crossfit_prediction_set_alpha"]),
    )
    crossfit_metrics = {
        model: evaluate_prediction_sets(frame.reset_index(drop=True))
        for model, frame in crossfit.groupby("model", sort=True)
    }
    qtrace_root = RESEARCH_RESULTS / "ac_v1" / "qtrace_mi"
    baseline_a = pd.read_csv(
        qtrace_root / "task_a_predictions_seed_ensemble.csv", dtype={"source_id": str}
    )
    baseline_a = baseline_a[baseline_a["model"].eq("a_only")].reset_index(drop=True)
    baseline_a_by_seed = pd.read_csv(
        qtrace_root / "task_a_predictions_by_seed.csv", dtype={"source_id": str}
    )
    baseline_a_metrics = evaluate_quality_predictions(baseline_a)
    inference: dict[str, Any] = {}
    bootstrap_frames: list[pd.DataFrame] = []
    seed_delta_records: dict[str, Any] = {}
    for model in ("c1_adapted_gru", "m2_oneway"):
        draws, record = _paired_bootstrap(
            model,
            aggregate["task_a"],
            aggregate["task_c"],
            baseline_a,
            "c0_frozen_gru",
            int(protocol["execution"]["bootstrap_resamples"]),
            int(protocol["execution"]["bootstrap_seed"]),
        )
        bootstrap_frames.append(draws)
        inference[model] = record
        seed_delta_records[model] = _seed_deltas(
            model,
            aggregate["task_a_by_seed"],
            aggregate["task_c_by_seed"],
            baseline_a_by_seed,
            "c0_frozen_gru",
        )
    gates = _posthoc_gates(
        aggregate,
        baseline_a_metrics,
        seed_delta_records,
        crossfit_metrics,
        protocol,
    )
    payloads = {
        "task_a_predictions_by_seed.csv": _csv_payload(aggregate["task_a_by_seed"]),
        "task_c_predictions_by_seed.csv": _csv_payload(aggregate["task_c_by_seed"]),
        "task_a_predictions_seed_ensemble.csv": _csv_payload(aggregate["task_a"]),
        "task_c_predictions_seed_ensemble.csv": _csv_payload(aggregate["task_c"]),
        "task_c_crossfit_prediction_sets.csv": _csv_payload(crossfit),
        "bootstrap_draws.csv": _csv_payload(pd.concat(bootstrap_frames, ignore_index=True)),
        "selection.json": canonical_json_bytes(
            {
                "registered_seed17_and_c1_fits_loaded_from": str(base_cache.relative_to(ROOT)),
                "new_m2_fits": extension_selections,
                "crossfit_calibration": crossfit_records,
            }
        ),
        "partitions.json": canonical_json_bytes({"partitions": partition_records}),
        "calibration.json": canonical_json_bytes(aggregate["calibration_records"]),
    }
    hashes = {
        name: write_create_only(output_dir / name, payload) for name, payload in payloads.items()
    }
    summary = {
        "result_id": "annomi-safe-mi-posthoc-extension-v2.1",
        "status": "complete_posthoc_exploratory_not_confirmatory",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(SAFE_MI_EXTENSION_PROTOCOL),
        "code_commit": git_commit(ROOT),
        "base_safe_mi_code_commit": base_summary["code_commit"],
        "config_sha256": config_sha256,
        "split_manifest_sha256": split_sha256,
        "models": sorted(model_names),
        "task_a_metrics": aggregate["task_a_metrics"],
        "task_c_metrics": aggregate["task_c_metrics"],
        "split_prediction_set_metrics": aggregate["prediction_set_metrics"],
        "crossfit_prediction_set_metrics": crossfit_metrics,
        "seed_deltas": seed_delta_records,
        "paired_source_bootstrap": inference,
        "descriptive_gates": gates,
        "stopping_rule": protocol["stopping_rule"],
        "evidence_sha256": hashes,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_create_only(output_dir / "summary.json", canonical_json_bytes(summary))
    validate_safe_mi_extension(output_dir)
    return summary


def _assert_close(actual: float, expected: float, name: str) -> None:
    if not np.isclose(actual, expected, atol=1e-8):
        raise ValueError(f"SAFE-MI v2.1 reconstruction mismatch: {name}")


def validate_safe_mi_extension(output_dir: Path = SAFE_MI_EXTENSION_RESULTS) -> None:
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        return
    summary = read_json(summary_path)
    if summary["status"] != "complete_posthoc_exploratory_not_confirmatory":
        raise ValueError("SAFE-MI v2.1 result is missing its post-hoc status")
    for name, expected in summary["evidence_sha256"].items():
        if sha256_file(output_dir / name) != expected:
            raise ValueError(f"SAFE-MI v2.1 evidence hash mismatch: {name}")
    task_a = pd.read_csv(
        output_dir / "task_a_predictions_seed_ensemble.csv", dtype={"source_id": str}
    )
    task_c = pd.read_csv(
        output_dir / "task_c_predictions_seed_ensemble.csv", dtype={"source_id": str}
    )
    crossfit = pd.read_csv(
        output_dir / "task_c_crossfit_prediction_sets.csv", dtype={"source_id": str}
    )
    for model, expected in summary["task_a_metrics"].items():
        actual = evaluate_quality_predictions(task_a[task_a["model"].eq(model)])
        for checkpoint, metrics in expected.items():
            for name in (
                "source_balanced_balanced_accuracy",
                "source_balanced_brier",
                "source_balanced_log_loss",
            ):
                _assert_close(float(actual[checkpoint][name]), float(metrics[name]), name)
    for model, expected in summary["task_c_metrics"].items():
        actual = evaluate_action_predictions(task_c[task_c["model"].eq(model)])
        for name in (
            "source_balanced_macro_f1",
            "source_balanced_brier",
            "source_balanced_log_loss",
        ):
            _assert_close(float(actual[name]), float(expected[name]), name)
        actual_sets = evaluate_prediction_sets(crossfit[crossfit["model"].eq(model)])
        expected_sets = summary["crossfit_prediction_set_metrics"][model]
        for name in (
            "source_balanced_coverage",
            "source_balanced_mean_set_size",
            "source_balanced_singleton_rate",
        ):
            _assert_close(float(actual_sets[name]), float(expected_sets[name]), name)
