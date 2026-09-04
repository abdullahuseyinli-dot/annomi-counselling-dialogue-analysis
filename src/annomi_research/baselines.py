from __future__ import annotations

import io
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion

from .constants import LABELS, PROTOCOL, RESEARCH_RESULTS, ROOT
from .data import Corpus, build_therapist_examples
from .io import canonical_json_bytes, read_json, sha256_file, write_create_only
from .metrics import evaluate_predictions, source_balanced_weights
from .splits import fold_lookup


@dataclass(frozen=True)
class Recipe:
    recipe_id: str
    alpha: float
    l1_ratio: float


RECIPES = (
    Recipe("alpha1e-4_l2", 1e-4, 0.0),
    Recipe("alpha3e-5_l2", 3e-5, 0.0),
    Recipe("alpha1e-5_l2", 1e-5, 0.0),
    Recipe("alpha3e-5_elastic15", 3e-5, 0.15),
)


def _features() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=50_000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    lowercase=True,
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=50_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def _sample_weights(frame: pd.DataFrame) -> np.ndarray:
    group_weights = source_balanced_weights(frame["source_id"])
    class_counts = frame["label"].value_counts()
    class_weights = (
        frame["label"]
        .map(lambda label: len(frame) / (len(LABELS) * class_counts[label]))
        .to_numpy(dtype=float)
    )
    weights = group_weights * class_weights
    return weights / weights.mean()


def _fit(
    train: pd.DataFrame,
    text_column: str,
    recipe: Recipe,
    seed: int,
) -> tuple[FeatureUnion, SGDClassifier]:
    features = _features()
    matrix = features.fit_transform(train[text_column])
    classifier = SGDClassifier(
        loss="log_loss",
        alpha=recipe.alpha,
        penalty="elasticnet",
        l1_ratio=recipe.l1_ratio,
        max_iter=3_000,
        tol=1e-4,
        average=True,
        random_state=seed,
    )
    classifier.fit(matrix, train["label"], sample_weight=_sample_weights(train))
    return features, classifier


def _predict(
    features: FeatureUnion,
    classifier: SGDClassifier,
    frame: pd.DataFrame,
    text_column: str,
) -> np.ndarray:
    raw = classifier.predict_proba(features.transform(frame[text_column]))
    aligned = np.zeros((len(frame), len(LABELS)), dtype=float)
    for source_index, label in enumerate(classifier.classes_):
        aligned[:, LABELS.index(str(label))] = raw[:, source_index]
    return aligned


def _inner_score(
    train: pd.DataFrame,
    text_column: str,
    recipe: Recipe,
    seed: int,
    n_splits: int,
) -> float:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores: list[float] = []
    indices = np.arange(len(train))
    for inner_train_idx, validation_idx in splitter.split(
        indices, train["label"], groups=train["source_id"]
    ):
        inner_train = train.iloc[inner_train_idx]
        validation = train.iloc[validation_idx]
        features, classifier = _fit(inner_train, text_column, recipe, seed)
        probabilities = _predict(features, classifier, validation, text_column)
        predicted = np.asarray(LABELS, dtype=object)[probabilities.argmax(axis=1)]
        scores.append(
            float(
                f1_score(
                    validation["label"],
                    predicted,
                    labels=list(LABELS),
                    average="macro",
                    sample_weight=source_balanced_weights(validation["source_id"]),
                    zero_division=0,
                )
            )
        )
    return float(np.mean(scores))


def _ledger_rows(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    model: str,
    fold: int,
    seed: int,
    selected_recipe: str,
    train_texts: set[str],
) -> pd.DataFrame:
    predicted = np.asarray(LABELS, dtype=object)[probabilities.argmax(axis=1)]
    ledger = frame[["transcript_id", "utterance_id", "source_id", "label"]].copy()
    ledger.insert(0, "model", model)
    ledger.insert(1, "seed", seed)
    ledger.insert(2, "outer_fold", fold)
    ledger["prediction"] = predicted
    for index, label in enumerate(LABELS):
        ledger[f"prob_{label}"] = probabilities[:, index]
    ledger["seen_text_in_outer_train"] = frame["normalized_text"].isin(train_texts).to_numpy()
    ledger["normalized_text_sha256"] = frame["normalized_text"].map(
        lambda value: __import__("hashlib").sha256(value.encode("utf-8")).hexdigest()
    )
    ledger["selected_recipe"] = selected_recipe
    return ledger


def _prior_probabilities(train: pd.DataFrame, test_size: int) -> np.ndarray:
    counts = train["label"].value_counts()
    values = np.asarray([float(counts.get(label, 0)) for label in LABELS], dtype=float)
    values /= values.sum()
    return np.repeat(values[None, :], test_size, axis=0)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run_baselines(
    corpus: Corpus,
    split_manifest: dict[str, Any],
    output_dir: Path = RESEARCH_RESULTS / "baseline_v1",
) -> dict[str, Any]:
    protocol = read_json(PROTOCOL)
    seed = int(protocol["development"]["seeds"][0])
    inner_folds = int(protocol["development"]["inner_folds"])
    examples = build_therapist_examples(
        corpus, context_turns=int(protocol["data"]["context_turns"])
    )
    examples["outer_fold"] = examples["source_id"].map(fold_lookup(split_manifest))
    if examples["outer_fold"].isna().any():
        raise ValueError("At least one source lacks an outer-fold assignment")

    ledgers: list[pd.DataFrame] = []
    selections: list[dict[str, Any]] = []
    model_columns = {
        "tfidf_elasticnet_utterance": "utterance_text",
        "tfidf_elasticnet_causal10": "context_text",
    }
    for fold in range(int(split_manifest["n_splits"])):
        train = examples[examples["outer_fold"].ne(fold)].reset_index(drop=True)
        test = examples[examples["outer_fold"].eq(fold)].reset_index(drop=True)
        train_texts = set(train["normalized_text"])

        prior = _prior_probabilities(train, len(test))
        ledgers.append(
            _ledger_rows(test, prior, "class_prior", fold, seed, "train_prior", train_texts)
        )

        for model, text_column in model_columns.items():
            candidate_scores = [
                {
                    "recipe_id": recipe.recipe_id,
                    "mean_inner_source_balanced_macro_f1": _inner_score(
                        train,
                        text_column,
                        recipe,
                        seed + fold,
                        inner_folds,
                    ),
                }
                for recipe in RECIPES
            ]
            best = max(
                candidate_scores,
                key=lambda item: (item["mean_inner_source_balanced_macro_f1"], item["recipe_id"]),
            )
            recipe = next(value for value in RECIPES if value.recipe_id == best["recipe_id"])
            features, classifier = _fit(train, text_column, recipe, seed + fold)
            probabilities = _predict(features, classifier, test, text_column)
            ledgers.append(
                _ledger_rows(test, probabilities, model, fold, seed, recipe.recipe_id, train_texts)
            )
            selections.append(
                {
                    "model": model,
                    "outer_fold": fold,
                    "selected_recipe": recipe.recipe_id,
                    "candidate_scores": candidate_scores,
                }
            )

    ledger = pd.concat(ledgers, ignore_index=True)
    ledger = ledger.sort_values(
        ["model", "outer_fold", "transcript_id", "utterance_id"], kind="stable"
    ).reset_index(drop=True)
    if ledger.duplicated(["model", "transcript_id", "utterance_id"]).any():
        raise ValueError("A model produced more than one out-of-fold prediction per utterance")
    expected = len(examples)
    counts = ledger.groupby("model").size()
    if not counts.eq(expected).all():
        raise ValueError(f"Incomplete out-of-fold coverage: {counts.to_dict()}")

    metrics = {
        model: evaluate_predictions(group.reset_index(drop=True))
        for model, group in ledger.groupby("model", sort=True)
    }
    metadata = {
        "result_id": "annomi-source-grouped-baselines-v1",
        "protocol_id": protocol["protocol_id"],
        "code_commit": _git_commit(),
        "split_manifest_sha256": split_manifest["manifest_sha256"],
        "dataset_sha256": read_json(ROOT / protocol["data"]["simple_manifest"])["sha256"],
        "seed": seed,
        "models": sorted(metrics),
        "metrics": metrics,
        "selections": selections,
    }

    buffer = io.StringIO()
    ledger.to_csv(buffer, index=False, lineterminator="\n", float_format="%.10g")
    ledger_payload = buffer.getvalue().encode("utf-8")
    ledger_digest = write_create_only(output_dir / "predictions.csv", ledger_payload)
    metadata["prediction_ledger_sha256"] = ledger_digest
    write_create_only(output_dir / "summary.json", canonical_json_bytes(metadata))
    write_create_only(
        output_dir / "selection.json", canonical_json_bytes({"selections": selections})
    )
    return metadata


def validate_baseline_evidence(output_dir: Path = RESEARCH_RESULTS / "baseline_v1") -> None:
    summary_path = output_dir / "summary.json"
    ledger_path = output_dir / "predictions.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if sha256_file(ledger_path) != summary["prediction_ledger_sha256"]:
        raise ValueError("Baseline prediction-ledger hash mismatch")
    ledger = pd.read_csv(ledger_path, dtype={"source_id": str})
    for model, group in ledger.groupby("model", sort=True):
        rebuilt = evaluate_predictions(group.reset_index(drop=True))
        recorded = summary["metrics"][model]
        for metric in (
            "utterance_macro_f1",
            "source_balanced_macro_f1",
            "source_balanced_brier",
            "source_balanced_log_loss",
        ):
            if not np.isclose(rebuilt[metric], recorded[metric], atol=1e-10):
                raise ValueError(f"Reconstruction mismatch for {model}/{metric}")
