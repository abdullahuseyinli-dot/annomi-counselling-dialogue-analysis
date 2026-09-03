from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from .constants import LABELS


def source_balanced_weights(sources: Iterable[str]) -> np.ndarray:
    series = pd.Series(list(sources), dtype="string")
    counts = series.value_counts()
    weights = series.map(lambda value: 1.0 / float(counts[value])).to_numpy(dtype=float)
    return weights / weights.mean()


def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray, weights: np.ndarray) -> float:
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    one_hot = np.zeros_like(probabilities, dtype=float)
    for row, label in enumerate(y_true):
        one_hot[row, label_to_index[str(label)]] = 1.0
    row_losses = np.square(probabilities - one_hot).sum(axis=1)
    return float(np.average(row_losses, weights=weights))


def equal_frequency_ece(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    n_bins: int = 10,
) -> float:
    predicted_index = probabilities.argmax(axis=1)
    predicted = np.asarray(LABELS, dtype=object)[predicted_index]
    confidence = probabilities.max(axis=1)
    correct = predicted == y_true
    order = np.argsort(confidence, kind="stable")
    bins = np.array_split(order, n_bins)
    total_weight = float(weights.sum())
    result = 0.0
    for indices in bins:
        if len(indices) == 0:
            continue
        bin_weight = float(weights[indices].sum())
        accuracy = float(np.average(correct[indices], weights=weights[indices]))
        mean_confidence = float(np.average(confidence[indices], weights=weights[indices]))
        result += bin_weight / total_weight * abs(accuracy - mean_confidence)
    return result


def _safe_macro_f1(y_true: pd.Series, y_pred: pd.Series) -> float | None:
    if y_true.empty:
        return None
    return float(f1_score(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0))


def evaluate_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    required = {
        "label",
        "prediction",
        "source_id",
        "seen_text_in_outer_train",
        *{f"prob_{label}" for label in LABELS},
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction ledger is missing columns: {sorted(missing)}")
    y_true = predictions["label"].to_numpy(dtype=object)
    y_pred = predictions["prediction"].to_numpy(dtype=object)
    probabilities = predictions[[f"prob_{label}" for label in LABELS]].to_numpy(dtype=float)
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Prediction probabilities do not sum to one")
    weights = source_balanced_weights(predictions["source_id"])

    ordinary_f1 = float(
        f1_score(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0)
    )
    balanced_f1 = float(
        f1_score(
            y_true,
            y_pred,
            labels=list(LABELS),
            average="macro",
            sample_weight=weights,
            zero_division=0,
        )
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(LABELS),
        zero_division=0,
    )
    clipped = np.clip(probabilities, 1e-12, 1.0)
    true_indices = np.asarray([LABELS.index(str(label)) for label in y_true], dtype=int)
    row_log_loss = -np.log(clipped[np.arange(len(clipped)), true_indices])
    source_losses = pd.DataFrame(
        {"source_id": predictions["source_id"].to_numpy(), "loss": row_log_loss}
    ).groupby("source_id", sort=False)["loss"].mean()
    tail_count = max(1, math.ceil(0.2 * len(source_losses)))
    confusion = confusion_matrix(y_true, y_pred, labels=list(LABELS)).astype(int)

    seen = predictions["seen_text_in_outer_train"].astype(bool)
    return {
        "n_predictions": len(predictions),
        "n_sources": int(predictions["source_id"].nunique()),
        "utterance_macro_f1": ordinary_f1,
        "source_balanced_macro_f1": balanced_f1,
        "source_balanced_brier": multiclass_brier(y_true, probabilities, weights),
        "source_balanced_log_loss": float(np.average(row_log_loss, weights=weights)),
        "equal_frequency_ece_10": equal_frequency_ece(y_true, clipped, weights),
        "worst_20pct_source_log_loss_cvar": float(
            source_losses.nlargest(tail_count).mean()
        ),
        "seen_text_macro_f1": _safe_macro_f1(
            predictions.loc[seen, "label"], predictions.loc[seen, "prediction"]
        ),
        "unseen_text_macro_f1": _safe_macro_f1(
            predictions.loc[~seen, "label"], predictions.loc[~seen, "prediction"]
        ),
        "seen_text_rows": int(seen.sum()),
        "unseen_text_rows": int((~seen).sum()),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
        "confusion_matrix": {
            "labels": list(LABELS),
            "rows_true_columns_predicted": confusion.tolist(),
        },
    }
