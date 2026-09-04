from __future__ import annotations

import io
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .ac_data import build_task_a_examples, build_task_c_examples
from .ac_metrics import evaluate_action_predictions, evaluate_quality_predictions
from .constants import (
    AC_PROTOCOL,
    CLIENT_LABELS,
    FULL_DATA,
    FULL_MANIFEST,
    LABELS,
    QTRACE_CONFIG,
    RESEARCH_RESULTS,
    ROOT,
    SIMPLE_DATA,
    SIMPLE_MANIFEST,
)
from .data import Corpus
from .io import canonical_json_bytes, git_commit, read_json, sha256_file, write_create_only
from .metrics import source_balanced_weights
from .splits import fold_lookup


def _git_is_clean() -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not completed.stdout.strip()


def _class_source_weights(frame: pd.DataFrame, labels: tuple[str, ...]) -> np.ndarray:
    source = source_balanced_weights(frame["source_id"])
    counts = frame["label"].value_counts()
    class_weights = frame["label"].map(
        lambda value: len(frame) / (len(labels) * float(counts[value]))
    ).to_numpy(dtype=float)
    values = source * class_weights
    return values / values.mean()


def _quality_prior(train: pd.DataFrame) -> float:
    weights = source_balanced_weights(train["source_id"])
    return float(np.average(train["label"].eq("low"), weights=weights))


def _action_prior(train: pd.DataFrame) -> np.ndarray:
    weights = source_balanced_weights(train["source_id"])
    values = np.asarray(
        [weights[train["label"].eq(label).to_numpy()].sum() for label in LABELS], dtype=float
    )
    values += 1.0
    return values / values.sum()


def _quality_ledger(
    test: pd.DataFrame,
    probabilities: np.ndarray,
    model: str,
    fold: int,
) -> pd.DataFrame:
    ledger = test[["transcript_id", "source_id", "checkpoint", "label"]].copy()
    ledger.insert(0, "model", model)
    ledger.insert(1, "outer_fold", fold)
    ledger["prob_low"] = probabilities
    ledger["prob_high"] = 1.0 - probabilities
    ledger["prediction"] = np.where(probabilities >= 0.5, "low", "high")
    return ledger


def _action_ledger(
    test: pd.DataFrame,
    probabilities: np.ndarray,
    model: str,
    fold: int,
) -> pd.DataFrame:
    ledger = test[
        ["transcript_id", "decision_utterance_id", "target_utterance_id", "source_id", "label"]
    ].copy()
    ledger.insert(0, "model", model)
    ledger.insert(1, "outer_fold", fold)
    ledger["prediction"] = np.asarray(LABELS, dtype=object)[probabilities.argmax(axis=1)]
    for index, label in enumerate(LABELS):
        ledger[f"prob_{label}"] = probabilities[:, index]
    ledger["seen_text_in_outer_train"] = False
    return ledger


def _fit_tfidf(
    train_text: pd.Series,
    train_labels: pd.Series,
    train_weights: np.ndarray,
    test_text: pd.Series,
    labels: tuple[str, ...],
) -> np.ndarray:
    word = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_features=40000,
        sublinear_tf=True,
    )
    character = TfidfVectorizer(
        analyzer="char_wb",
        lowercase=True,
        ngram_range=(3, 5),
        min_df=3,
        max_features=30000,
        sublinear_tf=True,
    )
    train_word = word.fit_transform(train_text.astype(str))
    train_character = character.fit_transform(train_text.astype(str))
    test_matrix = hstack(
        [word.transform(test_text.astype(str)), character.transform(test_text.astype(str))],
        format="csr",
    )
    train_matrix = hstack([train_word, train_character], format="csr")
    model = LogisticRegression(
        C=1.0,
        solver="liblinear" if len(labels) == 2 else "lbfgs",
        max_iter=2000,
        random_state=17,
    )
    model.fit(train_matrix, train_labels, sample_weight=train_weights)
    raw = model.predict_proba(test_matrix)
    order = [list(model.classes_).index(label) for label in labels]
    return raw[:, order]


def _fit_numeric_quality(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
) -> np.ndarray:
    scaler = StandardScaler()
    train_matrix = scaler.fit_transform(train[columns].fillna(0.0))
    test_matrix = scaler.transform(test[columns].fillna(0.0))
    model = LogisticRegression(C=1.0, solver="liblinear", max_iter=2000, random_state=17)
    model.fit(
        train_matrix,
        train["label"],
        sample_weight=_class_source_weights(train, ("high", "low")),
    )
    return model.predict_proba(test_matrix)[:, list(model.classes_).index("low")]


def _oracle_markov(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    global_prior = _action_prior(train)
    counts: dict[tuple[str, str], np.ndarray] = defaultdict(lambda: np.ones(len(LABELS)))
    for row in train.itertuples(index=False):
        previous = str(row.oracle_previous_therapist_label)
        client = str(row.oracle_current_client_label)
        if previous in LABELS and client in CLIENT_LABELS:
            counts[(previous, client)][LABELS.index(str(row.label))] += 1.0
    probabilities: list[np.ndarray] = []
    for row in test.itertuples(index=False):
        key = (str(row.oracle_previous_therapist_label), str(row.oracle_current_client_label))
        values = counts.get(key)
        probabilities.append(global_prior if values is None else values / values.sum())
    return np.stack(probabilities)


def run_ac_baselines(
    corpus: Corpus,
    split_manifest: dict[str, Any],
    output_dir: Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = output_dir or RESEARCH_RESULTS / "ac_v1" / "baselines"
    if (output_dir / "summary.json").exists():
        validate_ac_baseline_evidence(output_dir)
        return read_json(output_dir / "summary.json")
    protocol = read_json(AC_PROTOCOL)
    config = read_json(QTRACE_CONFIG)
    if protocol["status"] != "locked_before_task_ac_evaluation":
        raise ValueError("Task A/C protocol is not locked")
    if config["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("Task A/C protocol and Q-TRACE configuration disagree")
    for data_path, manifest_path in (
        (SIMPLE_DATA, SIMPLE_MANIFEST),
        (FULL_DATA, FULL_MANIFEST),
    ):
        if sha256_file(data_path) != read_json(manifest_path)["sha256"]:
            raise ValueError(f"Dataset hash mismatch: {data_path}")
    if RESEARCH_RESULTS in output_dir.parents and not _git_is_clean():
        raise RuntimeError("Commit tracked code and configuration before baseline evidence")
    lookup = fold_lookup(split_manifest)
    task_a = build_task_a_examples(corpus, tuple(protocol["task_a"]["therapist_turn_budgets"]))
    task_c = build_task_c_examples(corpus, int(protocol["task_c"]["context_turns_for_flat_baseline"]))
    task_a["outer_fold"] = task_a["source_id"].map(lookup)
    task_c["outer_fold"] = task_c["source_id"].map(lookup)
    a_ledgers: list[pd.DataFrame] = []
    c_ledgers: list[pd.DataFrame] = []
    structure_columns = [
        "observed_turns",
        "observed_words",
        "mean_words",
        "client_turns",
        "role_switch_rate",
    ]
    oracle_columns = [
        *structure_columns,
        *[f"oracle_therapist_prop_{label}" for label in LABELS],
        *[f"oracle_client_prop_{label}" for label in CLIENT_LABELS],
    ]

    for fold in sorted(task_a["outer_fold"].unique()):
        a_train = task_a[task_a["outer_fold"].ne(fold)].reset_index(drop=True)
        a_test = task_a[task_a["outer_fold"].eq(fold)].reset_index(drop=True)
        for checkpoint in a_test["checkpoint"].unique():
            train = a_train[a_train["checkpoint"].eq(checkpoint)].reset_index(drop=True)
            test = a_test[a_test["checkpoint"].eq(checkpoint)].reset_index(drop=True)
            prior = np.full(len(test), _quality_prior(train))
            a_ledgers.append(_quality_ledger(test, prior, "class_prior", int(fold)))
            a_ledgers.append(
                _quality_ledger(
                    test,
                    _fit_numeric_quality(train, test, structure_columns),
                    "structure_only",
                    int(fold),
                )
            )
            raw = _fit_tfidf(
                train["prefix_text"],
                train["label"],
                _class_source_weights(train, ("high", "low")),
                test["prefix_text"],
                ("high", "low"),
            )
            a_ledgers.append(
                _quality_ledger(test, raw[:, 1], "tfidf_raw_prefix", int(fold))
            )
            a_ledgers.append(
                _quality_ledger(
                    test,
                    _fit_numeric_quality(train, test, oracle_columns),
                    "oracle_gold_codes",
                    int(fold),
                )
            )

        c_train = task_c[task_c["outer_fold"].ne(fold)].reset_index(drop=True)
        c_test = task_c[task_c["outer_fold"].eq(fold)].reset_index(drop=True)
        c_ledgers.append(
            _action_ledger(
                c_test,
                np.tile(_action_prior(c_train), (len(c_test), 1)),
                "class_prior",
                int(fold),
            )
        )
        tfidf = _fit_tfidf(
            c_train["context_text"],
            c_train["label"],
            _class_source_weights(c_train, LABELS),
            c_test["context_text"],
            LABELS,
        )
        c_ledgers.append(_action_ledger(c_test, tfidf, "tfidf_causal10", int(fold)))
        c_ledgers.append(
            _action_ledger(c_test, _oracle_markov(c_train, c_test), "oracle_gold_markov", int(fold))
        )

    a_ledger = pd.concat(a_ledgers, ignore_index=True)
    c_ledger = pd.concat(c_ledgers, ignore_index=True)
    a_metrics = {
        model: evaluate_quality_predictions(frame.reset_index(drop=True))
        for model, frame in a_ledger.groupby("model", sort=True)
    }
    c_metrics = {
        model: evaluate_action_predictions(frame.reset_index(drop=True))
        for model, frame in c_ledger.groupby("model", sort=True)
    }
    a_buffer = io.StringIO()
    a_ledger.to_csv(a_buffer, index=False, lineterminator="\n", float_format="%.10g")
    c_buffer = io.StringIO()
    c_ledger.to_csv(c_buffer, index=False, lineterminator="\n", float_format="%.10g")
    a_hash = write_create_only(
        output_dir / "task_a_predictions.csv", a_buffer.getvalue().encode("utf-8")
    )
    c_hash = write_create_only(
        output_dir / "task_c_predictions.csv", c_buffer.getvalue().encode("utf-8")
    )
    summary = {
        "result_id": "annomi-task-ac-source-baselines-v1",
        "protocol_id": protocol["protocol_id"],
        "code_commit": git_commit(ROOT),
        "protocol_sha256": sha256_file(AC_PROTOCOL),
        "config_sha256": sha256_file(QTRACE_CONFIG),
        "split_manifest_sha256": split_manifest["manifest_sha256"],
        "dataset_sha256": {
            "simple": sha256_file(SIMPLE_DATA),
            "full": sha256_file(FULL_DATA),
        },
        "task_a_metrics": a_metrics,
        "task_c_metrics": c_metrics,
        "task_a_prediction_sha256": a_hash,
        "task_c_prediction_sha256": c_hash,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_create_only(output_dir / "summary.json", canonical_json_bytes(summary))
    validate_ac_baseline_evidence(output_dir)
    return summary


def validate_ac_baseline_evidence(output_dir: Path) -> None:
    summary = read_json(output_dir / "summary.json")
    task_a_path = output_dir / "task_a_predictions.csv"
    task_c_path = output_dir / "task_c_predictions.csv"
    if sha256_file(task_a_path) != summary["task_a_prediction_sha256"]:
        raise ValueError("Task A baseline prediction hash mismatch")
    if sha256_file(task_c_path) != summary["task_c_prediction_sha256"]:
        raise ValueError("Task C baseline prediction hash mismatch")
    task_a = pd.read_csv(task_a_path, dtype={"source_id": str})
    task_c = pd.read_csv(task_c_path, dtype={"source_id": str})
    if task_a.duplicated(["model", "transcript_id", "checkpoint"]).any():
        raise ValueError("Task A baseline ledger contains duplicate predictions")
    if task_c.duplicated(["model", "transcript_id", "target_utterance_id"]).any():
        raise ValueError("Task C baseline ledger contains duplicate predictions")
    for model, frame in task_a.groupby("model", sort=True):
        rebuilt = evaluate_quality_predictions(frame.reset_index(drop=True))
        for checkpoint, metrics in rebuilt.items():
            recorded = summary["task_a_metrics"][model][checkpoint]
            for metric in (
                "source_balanced_balanced_accuracy",
                "source_balanced_macro_f1",
                "source_balanced_brier",
            ):
                if not np.isclose(metrics[metric], recorded[metric], atol=1e-8):
                    raise ValueError(
                        f"Task A baseline reconstruction mismatch: {model}/{checkpoint}/{metric}"
                    )
    for model, frame in task_c.groupby("model", sort=True):
        rebuilt = evaluate_action_predictions(frame.reset_index(drop=True))
        recorded = summary["task_c_metrics"][model]
        for metric in (
            "source_balanced_macro_f1",
            "source_balanced_brier",
            "source_balanced_log_loss",
        ):
            if not np.isclose(rebuilt[metric], recorded[metric], atol=1e-8):
                raise ValueError(f"Task C baseline reconstruction mismatch: {model}/{metric}")
