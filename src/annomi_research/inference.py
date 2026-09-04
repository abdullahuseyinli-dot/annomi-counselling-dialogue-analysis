from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import LABELS, PROTOCOL, RESEARCH_RESULTS, ROOT
from .io import canonical_json_bytes, git_commit, read_json, sha256_file, write_create_only
from .metrics import evaluate_predictions

DEFAULT_CONFIG = ROOT / "configs" / "research" / "inference_v1.json"
DEFAULT_OUTPUT = RESEARCH_RESULTS / "comparisons" / "roberta_utterance_vs_tfidf_v1"
KEYS = ["transcript_id", "utterance_id", "source_id", "label"]


def _read_model_ledger(path: Path, spec: dict[str, Any]) -> pd.DataFrame:
    model = str(spec["model"])
    frame = pd.read_csv(path, dtype={"source_id": str})
    if "model" in frame and frame["model"].nunique() > 1:
        frame = frame[frame["model"].eq(model)].reset_index(drop=True)
    elif "model" in frame and not frame["model"].eq(model).all():
        raise ValueError(f"Expected model {model} in {path}")
    if frame.empty:
        raise ValueError(f"No rows for model {model} in {path}")
    if frame.duplicated(KEYS[:-1]).any():
        raise ValueError(f"Duplicate prediction keys in {path}")

    prediction_column = str(spec.get("prediction_column", "prediction"))
    probability_prefix = str(spec.get("probability_prefix", "prob_"))
    selected_columns = {
        prediction_column,
        *{f"{probability_prefix}{label}" for label in LABELS},
    }
    missing = selected_columns - set(frame.columns)
    if missing:
        raise ValueError(
            f"Configured prediction view is missing columns in {path}: {sorted(missing)}"
        )
    frame["prediction"] = frame[prediction_column]
    for label in LABELS:
        frame[f"prob_{label}"] = frame[f"{probability_prefix}{label}"]
    return frame


def _align_ledgers(
    baseline: pd.DataFrame, candidate: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = baseline.sort_values(KEYS, kind="stable").reset_index(drop=True)
    candidate = candidate.sort_values(KEYS, kind="stable").reset_index(drop=True)
    if not baseline[KEYS].equals(candidate[KEYS]):
        raise ValueError("Baseline and candidate ledgers do not describe identical examples")
    return baseline, candidate


def _macro_f1_from_confusions(confusions: np.ndarray) -> np.ndarray:
    true_totals = confusions.sum(axis=-1)
    predicted_totals = confusions.sum(axis=-2)
    true_positives = np.diagonal(confusions, axis1=-2, axis2=-1)
    denominator = true_totals + predicted_totals
    class_f1 = np.divide(
        2.0 * true_positives,
        denominator,
        out=np.zeros_like(true_positives, dtype=float),
        where=denominator > 0,
    )
    return class_f1.mean(axis=-1)


def _source_components(frame: pd.DataFrame) -> dict[str, Any]:
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    probability_columns = [f"prob_{label}" for label in LABELS]
    source_values = sorted(frame["source_id"].unique())
    confusions = np.zeros((len(source_values), len(LABELS), len(LABELS)), dtype=float)
    brier = np.zeros(len(source_values), dtype=float)
    log_loss = np.zeros(len(source_values), dtype=float)
    accuracy = np.zeros(len(source_values), dtype=float)
    support = np.zeros(len(source_values), dtype=int)
    for source_index, source in enumerate(source_values):
        group = frame[frame["source_id"].eq(source)]
        true_indices = np.asarray(
            [label_to_index[value] for value in group["label"]], dtype=np.intp
        )
        predicted_indices = np.asarray(
            [label_to_index[value] for value in group["prediction"]], dtype=np.intp
        )
        probabilities = group[probability_columns].to_numpy(dtype=float)
        one_hot = np.eye(len(LABELS), dtype=float)[true_indices]
        count = len(group)
        support[source_index] = count
        for true_index, predicted_index in zip(true_indices, predicted_indices, strict=True):
            confusions[source_index, true_index, predicted_index] += 1.0 / count
        row_brier = np.square(probabilities - one_hot).sum(axis=1)
        clipped = np.clip(probabilities, 1e-12, 1.0)
        brier[source_index] = float(row_brier.mean())
        log_loss[source_index] = float((-np.log(clipped[np.arange(count), true_indices])).mean())
        accuracy[source_index] = float((true_indices == predicted_indices).mean())
    return {
        "sources": source_values,
        "confusions": confusions,
        "brier": brier,
        "log_loss": log_loss,
        "accuracy": accuracy,
        "support": support,
    }


def _bootstrap_draws(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    n_resamples: int,
    seed: int,
) -> pd.DataFrame:
    if baseline["sources"] != candidate["sources"]:
        raise ValueError("Models do not cover the same source IDs")
    n_sources = len(baseline["sources"])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, n_sources, size=(n_resamples, n_sources))
    baseline_confusions = baseline["confusions"][sampled].mean(axis=1)
    candidate_confusions = candidate["confusions"][sampled].mean(axis=1)
    return pd.DataFrame(
        {
            "draw": np.arange(n_resamples),
            "delta_source_balanced_macro_f1": (
                _macro_f1_from_confusions(candidate_confusions)
                - _macro_f1_from_confusions(baseline_confusions)
            ),
            "delta_source_balanced_brier": (
                candidate["brier"][sampled].mean(axis=1) - baseline["brier"][sampled].mean(axis=1)
            ),
            "delta_source_balanced_log_loss": (
                candidate["log_loss"][sampled].mean(axis=1)
                - baseline["log_loss"][sampled].mean(axis=1)
            ),
        }
    )


def _interval(values: pd.Series, confidence: float) -> dict[str, float]:
    alpha = (1.0 - confidence) / 2.0
    return {
        "low": float(values.quantile(alpha, interpolation="linear")),
        "median": float(values.quantile(0.5, interpolation="linear")),
        "high": float(values.quantile(1.0 - alpha, interpolation="linear")),
    }


def _per_source_table(baseline: dict[str, Any], candidate: dict[str, Any]) -> pd.DataFrame:
    baseline_f1 = _macro_f1_from_confusions(baseline["confusions"])
    candidate_f1 = _macro_f1_from_confusions(candidate["confusions"])
    return pd.DataFrame(
        {
            "source_id": baseline["sources"],
            "support": baseline["support"],
            "baseline_macro_f1": baseline_f1,
            "candidate_macro_f1": candidate_f1,
            "delta_macro_f1": candidate_f1 - baseline_f1,
            "baseline_accuracy": baseline["accuracy"],
            "candidate_accuracy": candidate["accuracy"],
            "delta_accuracy": candidate["accuracy"] - baseline["accuracy"],
            "baseline_brier": baseline["brier"],
            "candidate_brier": candidate["brier"],
            "delta_brier": candidate["brier"] - baseline["brier"],
        }
    )


def _per_seed_f1_deltas(
    config: dict[str, Any],
    baseline_f1: float,
) -> tuple[dict[str, float], str]:
    candidate_summary = read_json(ROOT / config["candidate"]["per_seed_summary"])
    candidate_metrics_key = config["candidate"].get("per_seed_metrics_key", "per_seed")
    candidate_seeds = candidate_summary["metrics"][candidate_metrics_key]
    baseline_summary_path = config["baseline"].get("per_seed_summary")
    if baseline_summary_path is None:
        return (
            {
                seed: metrics["source_balanced_macro_f1"] - baseline_f1
                for seed, metrics in candidate_seeds.items()
            },
            "candidate seed minus baseline ensemble",
        )

    baseline_summary = read_json(ROOT / baseline_summary_path)
    baseline_metrics_key = config["baseline"].get("per_seed_metrics_key", "per_seed")
    baseline_seeds = baseline_summary["metrics"][baseline_metrics_key]
    if set(candidate_seeds) != set(baseline_seeds):
        raise ValueError("Candidate and baseline summaries do not contain matching seeds")
    return (
        {
            seed: candidate_seeds[seed]["source_balanced_macro_f1"]
            - baseline_seeds[seed]["source_balanced_macro_f1"]
            for seed in sorted(candidate_seeds, key=int)
        },
        "candidate seed minus matching baseline seed",
    )


def run_comparison(
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    if (output_dir / "summary.json").exists():
        validate_comparison_evidence(output_dir)
        return read_json(output_dir / "summary.json")
    config = read_json(config_path)
    protocol = read_json(PROTOCOL)
    if config["status"] != "registered_before_bootstrap_computation":
        raise ValueError("Inference configuration is not registered")
    if config["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("Inference and research protocols disagree")
    if config["bootstrap_resamples"] != protocol["inference"]["bootstrap_resamples"]:
        raise ValueError("Inference and research protocols disagree on bootstrap count")
    baseline_path = ROOT / config["baseline"]["ledger"]
    candidate_path = ROOT / config["candidate"]["ledger"]
    baseline, candidate = _align_ledgers(
        _read_model_ledger(baseline_path, config["baseline"]),
        _read_model_ledger(candidate_path, config["candidate"]),
    )
    baseline_metrics = evaluate_predictions(baseline)
    candidate_metrics = evaluate_predictions(candidate)
    baseline_components = _source_components(baseline)
    candidate_components = _source_components(candidate)
    draws = _bootstrap_draws(
        baseline_components,
        candidate_components,
        int(config["bootstrap_resamples"]),
        int(config["bootstrap_seed"]),
    )
    per_source = _per_source_table(baseline_components, candidate_components)
    confidence = float(config["confidence_level"])
    intervals = {
        column.removeprefix("delta_"): _interval(draws[column], confidence)
        for column in draws.columns
        if column.startswith("delta_")
    }
    baseline_f1 = baseline_metrics["source_balanced_macro_f1"]
    per_seed_deltas, per_seed_contrast = _per_seed_f1_deltas(config, baseline_f1)
    class_collapse = any(candidate_metrics["per_class"][label]["f1"] == 0.0 for label in LABELS)
    point_deltas = {
        metric: candidate_metrics[metric] - baseline_metrics[metric]
        for metric in (
            "source_balanced_macro_f1",
            "source_balanced_brier",
            "source_balanced_log_loss",
        )
    }
    gate = protocol["candidate_success_gate"]
    success_checks = {
        "minimum_f1_delta": (
            point_deltas["source_balanced_macro_f1"]
            >= float(gate["minimum_source_balanced_macro_f1_delta"])
        ),
        "f1_ci_excludes_zero": intervals["source_balanced_macro_f1"]["low"] > 0.0,
        "positive_seed_count": (
            sum(delta > 0.0 for delta in per_seed_deltas.values())
            >= int(gate["minimum_positive_seed_count"])
        ),
        "brier_degradation_within_limit": (
            point_deltas["source_balanced_brier"]
            <= float(gate["maximum_source_balanced_brier_degradation"])
        ),
        "no_class_collapse": not class_collapse,
    }
    gate_applies = bool(config.get("apply_candidate_success_gate", True))

    draw_buffer = io.StringIO()
    draws.to_csv(draw_buffer, index=False, lineterminator="\n", float_format="%.12g")
    source_buffer = io.StringIO()
    per_source.to_csv(source_buffer, index=False, lineterminator="\n", float_format="%.12g")
    draw_hash = write_create_only(
        output_dir / "bootstrap_draws.csv", draw_buffer.getvalue().encode("utf-8")
    )
    source_hash = write_create_only(
        output_dir / "per_source.csv", source_buffer.getvalue().encode("utf-8")
    )
    summary = {
        "result_id": config.get("result_id", "annomi-roberta-utterance-vs-tfidf-paired-source-v1"),
        "comparison_role": config.get("comparison_role", "primary_success_gate"),
        "protocol_id": protocol["protocol_id"],
        "code_commit": git_commit(ROOT),
        "config_sha256": sha256_file(config_path),
        "baseline_ledger_sha256": sha256_file(baseline_path),
        "candidate_ledger_sha256": sha256_file(candidate_path),
        "n_examples": len(baseline),
        "n_sources": len(baseline_components["sources"]),
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "point_deltas_candidate_minus_baseline": point_deltas,
        "bootstrap": {
            "sampling_unit": config["sampling_unit"],
            "paired": config["paired"],
            "resamples": config["bootstrap_resamples"],
            "seed": config["bootstrap_seed"],
            "confidence_level": confidence,
            "intervals": intervals,
            "draw_ledger_sha256": draw_hash,
        },
        "per_seed_f1_deltas": per_seed_deltas,
        "per_seed_contrast": per_seed_contrast,
        "positive_seed_count": sum(delta > 0.0 for delta in per_seed_deltas.values()),
        "per_source_descriptive": {
            "positive_macro_f1_sources": int((per_source["delta_macro_f1"] > 0).sum()),
            "tied_macro_f1_sources": int((per_source["delta_macro_f1"] == 0).sum()),
            "negative_macro_f1_sources": int((per_source["delta_macro_f1"] < 0).sum()),
            "median_macro_f1_delta": float(per_source["delta_macro_f1"].median()),
            "source_ledger_sha256": source_hash,
        },
        "candidate_success_gate": {
            "applies": gate_applies,
            "checks": success_checks,
            "pass": all(success_checks.values()) if gate_applies else None,
        },
    }
    write_create_only(output_dir / "summary.json", canonical_json_bytes(summary))
    validate_comparison_evidence(output_dir)
    return summary


def validate_comparison_evidence(output_dir: Path = DEFAULT_OUTPUT) -> None:
    summary = read_json(output_dir / "summary.json")
    draws_path = output_dir / "bootstrap_draws.csv"
    sources_path = output_dir / "per_source.csv"
    if sha256_file(draws_path) != summary["bootstrap"]["draw_ledger_sha256"]:
        raise ValueError("Bootstrap draw-ledger hash mismatch")
    if sha256_file(sources_path) != summary["per_source_descriptive"]["source_ledger_sha256"]:
        raise ValueError("Per-source comparison-ledger hash mismatch")
    draws = pd.read_csv(draws_path)
    confidence = float(summary["bootstrap"]["confidence_level"])
    for metric, recorded in summary["bootstrap"]["intervals"].items():
        rebuilt = _interval(draws[f"delta_{metric}"], confidence)
        for endpoint in ("low", "median", "high"):
            if not np.isclose(rebuilt[endpoint], recorded[endpoint], atol=1e-10):
                raise ValueError(f"Bootstrap interval mismatch for {metric}/{endpoint}")
