from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from .constants import LABELS
from .metrics import evaluate_predictions, source_balanced_weights


def _binary_ece(
    targets: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    n_bins: int = 10,
) -> float:
    order = np.argsort(probabilities, kind="stable")
    bins = np.array_split(order, min(n_bins, len(order)))
    total = float(weights.sum())
    return float(
        sum(
            float(weights[index].sum())
            / total
            * abs(
                float(np.average(targets[index], weights=weights[index]))
                - float(np.average(probabilities[index], weights=weights[index]))
            )
            for index in bins
            if len(index)
        )
    )


def evaluate_quality_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    required = {"checkpoint", "label", "source_id", "prob_low"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Task A ledger is missing columns: {sorted(missing)}")
    result: dict[str, Any] = {}
    checkpoint_order = ["t3", "t5", "t10", "t20", "full"]
    for checkpoint in checkpoint_order:
        frame = predictions[predictions["checkpoint"].eq(checkpoint)].reset_index(drop=True)
        if frame.empty:
            continue
        probabilities = frame["prob_low"].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError("Task A probabilities must be finite and in [0, 1]")
        targets = frame["label"].eq("low").to_numpy(dtype=int)
        predicted = np.where(probabilities >= 0.5, "low", "high")
        weights = source_balanced_weights(frame["source_id"])
        clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
        row_log_loss = -(targets * np.log(clipped) + (1 - targets) * np.log(1 - clipped))
        result[checkpoint] = {
            "n_transcripts": len(frame),
            "n_sources": int(frame["source_id"].nunique()),
            "low_transcripts": int(targets.sum()),
            "source_balanced_accuracy": float(np.average(predicted == frame["label"], weights=weights)),
            "source_balanced_balanced_accuracy": float(
                balanced_accuracy_score(frame["label"], predicted, sample_weight=weights)
            ),
            "source_balanced_macro_f1": float(
                f1_score(
                    frame["label"],
                    predicted,
                    labels=["high", "low"],
                    average="macro",
                    sample_weight=weights,
                    zero_division=0,
                )
            ),
            "source_balanced_low_f1": float(
                f1_score(
                    frame["label"],
                    predicted,
                    labels=["low"],
                    average="macro",
                    sample_weight=weights,
                    zero_division=0,
                )
            ),
            "source_balanced_low_auprc": (
                float(average_precision_score(targets, probabilities, sample_weight=weights))
                if len(np.unique(targets)) == 2
                else None
            ),
            "source_balanced_roc_auc": (
                float(roc_auc_score(targets, probabilities, sample_weight=weights))
                if len(np.unique(targets)) == 2
                else None
            ),
            "source_balanced_brier": float(
                np.average(np.square(probabilities - targets), weights=weights)
            ),
            "source_balanced_log_loss": float(np.average(row_log_loss, weights=weights)),
            "equal_frequency_ece_10": _binary_ece(targets, probabilities, weights),
        }
    return result


def evaluate_action_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    frame = predictions.copy()
    if "seen_text_in_outer_train" not in frame:
        frame["seen_text_in_outer_train"] = False
    return evaluate_predictions(frame)


def _temperature_probabilities(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def fit_multiclass_temperature(
    probabilities: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
) -> float:
    label_indices = np.asarray([LABELS.index(str(label)) for label in labels], dtype=int)
    weights = source_balanced_weights(sources)

    def objective(log_temperature: float) -> float:
        calibrated = _temperature_probabilities(probabilities, np.exp(log_temperature))
        losses = -np.log(np.clip(calibrated[np.arange(len(labels)), label_indices], 1e-12, 1.0))
        return float(np.average(losses, weights=weights))

    result = minimize_scalar(objective, bounds=(np.log(0.25), np.log(4.0)), method="bounded")
    if not result.success:
        raise RuntimeError("Multiclass temperature fitting failed")
    return float(np.exp(result.x))


def apply_multiclass_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    return _temperature_probabilities(probabilities, temperature)


def fit_binary_temperature(
    probabilities: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
) -> float:
    targets = np.asarray(labels == "low", dtype=int)
    weights = source_balanced_weights(sources)
    logits = np.log(np.clip(probabilities, 1e-12, 1 - 1e-12)) - np.log(
        np.clip(1 - probabilities, 1e-12, 1 - 1e-12)
    )

    def objective(log_temperature: float) -> float:
        scaled = logits / np.exp(log_temperature)
        calibrated = 1.0 / (1.0 + np.exp(-np.clip(scaled, -40, 40)))
        calibrated = np.clip(calibrated, 1e-12, 1 - 1e-12)
        losses = -(targets * np.log(calibrated) + (1 - targets) * np.log(1 - calibrated))
        return float(np.average(losses, weights=weights))

    result = minimize_scalar(objective, bounds=(np.log(0.25), np.log(4.0)), method="bounded")
    if not result.success:
        raise RuntimeError("Binary temperature fitting failed")
    return float(np.exp(result.x))


def apply_binary_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-12, 1 - 1e-12)) - np.log(
        np.clip(1 - probabilities, 1e-12, 1 - 1e-12)
    )
    return 1.0 / (1.0 + np.exp(-np.clip(logits / temperature, -40, 40)))


def aps_scores(probabilities: np.ndarray) -> np.ndarray:
    """Return deterministic APS candidate scores for every row and class."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(LABELS):
        raise ValueError("APS expects one probability column per therapist label")
    order = np.argsort(-probabilities, axis=1, kind="stable")
    sorted_probabilities = np.take_along_axis(probabilities, order, axis=1)
    cumulative = np.cumsum(sorted_probabilities, axis=1)
    scores = np.empty_like(cumulative)
    np.put_along_axis(scores, order, cumulative, axis=1)
    return scores


def source_crc_threshold(
    true_scores: np.ndarray,
    sources: np.ndarray,
    alpha: float,
) -> dict[str, float | int]:
    """Calibrate expected within-source miscoverage with sources as exchangeable units."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    frame = pd.DataFrame({"source_id": sources, "score": true_scores})
    n_sources = int(frame["source_id"].nunique())
    candidates = np.unique(np.concatenate(([0.0], true_scores, [1.0])))
    selected = 1.0
    selected_empirical_risk = 0.0
    selected_bound = 1.0 / (n_sources + 1)
    for threshold in candidates:
        source_losses = frame.assign(missed=frame["score"].gt(threshold)).groupby(
            "source_id", sort=False
        )["missed"].mean()
        empirical_risk = float(source_losses.mean())
        risk_bound = n_sources / (n_sources + 1) * empirical_risk + 1.0 / (n_sources + 1)
        if risk_bound <= alpha + 1e-12:
            selected = float(threshold)
            selected_empirical_risk = empirical_risk
            selected_bound = risk_bound
            break
    return {
        "threshold": selected,
        "alpha": float(alpha),
        "calibration_sources": n_sources,
        "empirical_source_mean_miscoverage": selected_empirical_risk,
        "finite_sample_risk_bound": selected_bound,
    }


def add_prediction_sets(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    frame = predictions.copy()
    probabilities = frame[[f"prob_{label}" for label in LABELS]].to_numpy(dtype=float)
    scores = aps_scores(probabilities)
    included = scores <= float(threshold) + 1e-12
    included[np.arange(len(included)), probabilities.argmax(axis=1)] = True
    sets: list[str] = []
    for row in range(len(frame)):
        labels = [LABELS[index] for index in range(len(LABELS)) if included[row, index]]
        sets.append("|".join(labels))
    frame["prediction_set"] = sets
    frame["prediction_set_size"] = included.sum(axis=1).astype(int)
    frame["set_covered"] = [
        str(label) in value.split("|") for label, value in zip(frame["label"], sets, strict=True)
    ]
    return frame


def evaluate_prediction_sets(predictions: pd.DataFrame) -> dict[str, Any]:
    required = {"source_id", "label", "prediction_set", "prediction_set_size", "set_covered"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction-set ledger is missing columns: {sorted(missing)}")
    weights = source_balanced_weights(predictions["source_id"])
    result: dict[str, Any] = {
        "n_decisions": len(predictions),
        "n_sources": int(predictions["source_id"].nunique()),
        "source_balanced_coverage": float(
            np.average(predictions["set_covered"].astype(bool), weights=weights)
        ),
        "source_balanced_mean_set_size": float(
            np.average(predictions["prediction_set_size"], weights=weights)
        ),
        "source_balanced_singleton_rate": float(
            np.average(predictions["prediction_set_size"].eq(1), weights=weights)
        ),
        "per_class_coverage": {},
    }
    for label in LABELS:
        mask = predictions["label"].eq(label).to_numpy()
        result["per_class_coverage"][label] = float(
            np.average(predictions.loc[mask, "set_covered"].astype(bool), weights=weights[mask])
        )
    return result
