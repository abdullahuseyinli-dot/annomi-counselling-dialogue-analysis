from __future__ import annotations

import gc
import hashlib
import json
import math
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors
from torch import nn

from .ac_data import SessionTurns, build_session_turns
from .ac_metrics import (
    evaluate_action_predictions,
    evaluate_prediction_sets,
    evaluate_quality_predictions,
)
from .constants import (
    CLIENT_LABELS,
    FULL_DATA,
    FULL_MANIFEST,
    LABELS,
    RESEARCH_RESULTS,
    ROOT,
    SAFE_MI_CONFIG,
    SAFE_MI_PROTOCOL,
    SIMPLE_DATA,
    SIMPLE_MANIFEST,
)
from .data import Corpus, normalize_text
from .io import canonical_json_bytes, git_commit, read_json, sha256_file, write_create_only
from .qtrace import (
    _calibrate_action,
    _calibrate_quality,
    _class_weights,
    _csv_payload,
    _ensemble_action,
    _ensemble_quality,
    _git_is_clean,
    _inner_partitions,
    _loader,
    _move_batch,
    _require_device,
    _runtime_environment,
    _seed_everything,
    _source_bootstrap_weights,
    _source_weighted_low_prior,
    extract_turn_embeddings,
)
from .safe_mi_encoder import train_and_extract_adapted_embeddings
from .safe_mi_model import SafeMIMode, SafeMIModel, mode_from_config, safe_mi_loss
from .splits import fold_lookup, validate_source_folds


@dataclass(frozen=True)
class SafeFitResult:
    best_epoch: int
    best_score: float
    epochs_completed: int
    peak_memory_bytes: int
    validation_metrics: dict[str, Any]
    validation_a: pd.DataFrame
    validation_c: pd.DataFrame
    calibration_a: pd.DataFrame
    calibration_c: pd.DataFrame
    test_a: pd.DataFrame
    test_c: pd.DataFrame


def estimate_global_transitions(
    sessions: list[SessionTurns],
    dirichlet_strength: float,
) -> np.ndarray:
    """Estimate P(next therapist act | previous therapist act, current client code)."""

    counts = np.zeros((len(LABELS), len(CLIENT_LABELS), len(LABELS)), dtype=float)
    global_counts = np.ones(len(LABELS), dtype=float)
    for session in sessions:
        previous_therapist = -100
        for position, role in enumerate(session.roles):
            if role == 1:
                previous_therapist = int(session.therapist_labels[position])
                continue
            target = int(session.next_action_targets[position])
            client = int(session.client_labels[position])
            if target < 0:
                continue
            global_counts[target] += 1.0
            if previous_therapist >= 0 and client >= 0:
                counts[previous_therapist, client, target] += 1.0
    global_prior = global_counts / global_counts.sum()
    smoothed = counts + float(dirichlet_strength) * global_prior[None, None, :]
    probabilities = smoothed / smoothed.sum(axis=-1, keepdims=True)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=-1), 1.0):
        raise AssertionError("SAFE-MI transition probabilities are invalid")
    return probabilities


def _action_class_prior(sessions: list[SessionTurns]) -> np.ndarray:
    counts = np.ones(len(LABELS), dtype=float)
    for session in sessions:
        values = session.next_action_targets[session.next_action_targets >= 0]
        counts += np.bincount(values, minlength=len(LABELS))
    return counts / counts.sum()


def _safe_class_weights(
    sessions: list[SessionTurns],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    result = _class_weights(sessions, device)
    result["action_log_prior"] = torch.log(
        torch.tensor(_action_class_prior(sessions), dtype=torch.float32, device=device)
    )
    return result


def _require_registered_state(
    corpus: Corpus,
    split_manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = read_json(SAFE_MI_PROTOCOL)
    config = read_json(SAFE_MI_CONFIG)
    if protocol["status"] != "registered_exploratory_after_qtrace_v1":
        raise ValueError("SAFE-MI protocol does not retain its exploratory status")
    if config["status"] != "registered_exploratory_before_safe_mi_execution":
        raise ValueError("SAFE-MI model configuration is not registered")
    if config["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("SAFE-MI protocol and configuration disagree")
    validate_source_folds(corpus, split_manifest)
    for data_path, manifest_path in (
        (SIMPLE_DATA, SIMPLE_MANIFEST),
        (FULL_DATA, FULL_MANIFEST),
    ):
        if sha256_file(data_path) != read_json(manifest_path)["sha256"]:
            raise ValueError(f"Dataset hash mismatch: {data_path}")
    if not _git_is_clean():
        raise RuntimeError("Commit tracked SAFE-MI code/configuration before generating evidence")
    return protocol, config


def _make_model(
    sessions: list[SessionTurns],
    embeddings: dict[tuple[int, int], np.ndarray],
    config: dict[str, Any],
    mode: SafeMIMode,
    device: torch.device,
) -> SafeMIModel:
    transition = estimate_global_transitions(
        sessions,
        float(config["architecture"]["transition_dirichlet_strength"]),
    )
    return SafeMIModel(
        embedding_size=len(next(iter(embeddings.values()))),
        transition_probabilities=transition,
        low_quality_prior=_source_weighted_low_prior(sessions),
        action_class_prior=_action_class_prior(sessions),
        architecture=config["architecture"],
        mode=mode,
    ).to(device)


@torch.inference_mode()
def _predict_sessions(
    model: SafeMIModel,
    sessions: list[SessionTurns],
    embeddings: dict[tuple[int, int], np.ndarray],
    device: torch.device,
    batch_size: int,
    fold: int,
    seed: int,
    fit_texts: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model.eval()
    quality_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    for batch in _loader(sessions, embeddings, batch_size, shuffle=False, seed=seed):
        batch_on_device = _move_batch(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(
                batch_on_device["embeddings"],
                batch_on_device["roles"],
                batch_on_device["lengths"],
            )
        quality_probabilities = output["online_quality_probabilities"].float().cpu().numpy()
        action_probabilities = output["action_probabilities"].float().cpu().numpy()
        transition_gates = output["transition_gate"].float().cpu().numpy()
        transition_strength = float(output["transition_strength"].detach().cpu())
        text_evidence = output["text_quality_evidence"].float().cpu().numpy()
        action_evidence = output["action_quality_evidence"].float().cpu().numpy()
        for row, session in enumerate(batch["sessions"]):
            if model.mode.task_a_loss:
                for checkpoint, position in session.quality_positions.items():
                    probability_low = float(quality_probabilities[row, position, 1])
                    quality_rows.append(
                        {
                            "model": model.mode.name,
                            "seed": seed,
                            "outer_fold": fold,
                            "transcript_id": session.transcript_id,
                            "source_id": session.source_id,
                            "checkpoint": checkpoint,
                            "last_utterance_id": int(session.utterance_ids[position]),
                            "label": ("high", "low")[session.quality],
                            "prob_high": 1.0 - probability_low,
                            "prob_low": probability_low,
                            "prediction": "low" if probability_low >= 0.5 else "high",
                            "cumulative_text_evidence": float(
                                text_evidence[row, : position + 1].sum()
                            ),
                            "cumulative_action_evidence": float(
                                action_evidence[row, : position + 1].sum()
                            ),
                        }
                    )
            if not model.mode.task_c_loss:
                continue
            for position in np.flatnonzero(session.next_action_targets >= 0):
                probabilities = action_probabilities[row, position]
                target_position = position + 1
                record: dict[str, Any] = {
                    "model": model.mode.name,
                    "seed": seed,
                    "outer_fold": fold,
                    "transcript_id": session.transcript_id,
                    "decision_utterance_id": int(session.utterance_ids[position]),
                    "target_utterance_id": int(session.utterance_ids[target_position]),
                    "source_id": session.source_id,
                    "label": LABELS[int(session.next_action_targets[position])],
                    "prediction": LABELS[int(probabilities.argmax())],
                    "seen_text_in_outer_train": normalize_text(session.texts[position])
                    in fit_texts,
                    "predicted_low_quality_probability": (
                        float(quality_probabilities[row, position, 1])
                        if model.mode.task_a_loss
                        else math.nan
                    ),
                    "transition_gate_high": float(transition_gates[row, position]),
                    "transition_gate_low": float(transition_gates[row, position]),
                    "transition_residual_strength": transition_strength,
                }
                for index, label in enumerate(LABELS):
                    record[f"prob_{label}"] = float(probabilities[index])
                action_rows.append(record)
    return pd.DataFrame(quality_rows), pd.DataFrame(action_rows)


def _selection_score(
    quality: pd.DataFrame,
    action: pd.DataFrame,
    mode: SafeMIMode,
    training: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    metrics: dict[str, Any] = {}
    score = 0.0
    if mode.task_c_loss:
        metrics["task_c"] = evaluate_action_predictions(action)
        score = float(metrics["task_c"]["source_balanced_macro_f1"])
        score -= float(training["selection_brier_weight"]) * float(
            metrics["task_c"]["source_balanced_brier"]
        )
    if mode.task_a_loss:
        metrics["task_a"] = evaluate_quality_predictions(quality)
        a_metrics = metrics["task_a"]["t10"]
        if mode.task_c_loss:
            score -= float(training["selection_task_a_log_loss_weight"]) * float(
                a_metrics["source_balanced_log_loss"]
            )
        else:
            score = -float(a_metrics["source_balanced_log_loss"])
    return score, metrics


def _train_epoch(
    model: SafeMIModel,
    loader: Iterable[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    class_weights: dict[str, torch.Tensor],
    config: dict[str, Any],
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        batch_on_device = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(
                batch_on_device["embeddings"],
                batch_on_device["roles"],
                batch_on_device["lengths"],
            )
            loss, _ = safe_mi_loss(
                output,
                batch_on_device,
                model.mode,
                config["training"],
                class_weights,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("SAFE-MI training produced a non-finite loss")
        loss.backward()
        nn.utils.clip_grad_norm_(
            model.parameters(),
            float(config["training"]["maximum_gradient_norm"]),
        )
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def _optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )


def _fit_one(
    fit: list[SessionTurns],
    validation: list[SessionTurns],
    calibration: list[SessionTurns],
    test: list[SessionTurns],
    selection_embeddings: dict[tuple[int, int], np.ndarray],
    refit_embeddings: dict[tuple[int, int], np.ndarray],
    config: dict[str, Any],
    mode: SafeMIMode,
    fold: int,
    seed: int,
) -> SafeFitResult:
    device = _require_device()
    training = config["training"]
    batch_size = int(training["batch_size_sessions"])
    _seed_everything(seed)
    model = _make_model(fit, selection_embeddings, config, mode, device)
    optimizer = _optimizer(model, config)
    class_weights = _safe_class_weights(fit, device)
    train_loader = _loader(fit, selection_embeddings, batch_size, shuffle=True, seed=seed)
    fit_texts = {normalize_text(text) for session in fit for text in session.texts}
    best_score = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] = {}
    best_validation_a = pd.DataFrame()
    best_validation_c = pd.DataFrame()
    stale = 0
    epochs_completed = 0
    peak_memory = 0
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for epoch in range(1, int(training["maximum_epochs"]) + 1):
            mean_loss = _train_epoch(
                model,
                train_loader,
                optimizer,
                class_weights,
                config,
                device,
            )
            epochs_completed = epoch
            validation_a, validation_c = _predict_sessions(
                model,
                validation,
                selection_embeddings,
                device,
                batch_size,
                fold,
                seed,
                fit_texts,
            )
            score, metrics = _selection_score(
                validation_a,
                validation_c,
                mode,
                training,
            )
            if score > best_score + 1e-8:
                best_score = score
                best_epoch = epoch
                best_metrics = metrics
                best_validation_a = validation_a
                best_validation_c = validation_c
                best_state = {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                }
                stale = 0
                print(
                    f"SAFE-MI {mode.name}/fold={fold}/seed={seed}: "
                    f"epoch={epoch}, selection={score:.4f}, loss={mean_loss:.4f}",
                    flush=True,
                )
            else:
                stale += 1
            if epoch >= int(training["minimum_epochs"]) and stale >= int(
                training["early_stopping_patience"]
            ):
                break
        if best_state is None:
            raise AssertionError("SAFE-MI training never selected an epoch")
        peak_memory = max(peak_memory, int(torch.cuda.max_memory_allocated(device)))
    finally:
        del model
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()

    refit = fit + validation
    _seed_everything(seed)
    model = _make_model(refit, refit_embeddings, config, mode, device)
    optimizer = _optimizer(model, config)
    class_weights = _safe_class_weights(refit, device)
    refit_loader = _loader(refit, refit_embeddings, batch_size, shuffle=True, seed=seed)
    refit_texts = {normalize_text(text) for session in refit for text in session.texts}
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for _ in range(best_epoch):
            _train_epoch(
                model,
                refit_loader,
                optimizer,
                class_weights,
                config,
                device,
            )
        calibration_a, calibration_c = _predict_sessions(
            model,
            calibration,
            refit_embeddings,
            device,
            batch_size,
            fold,
            seed,
            refit_texts,
        )
        test_a, test_c = _predict_sessions(
            model,
            test,
            refit_embeddings,
            device,
            batch_size,
            fold,
            seed,
            refit_texts,
        )
        peak_memory = max(peak_memory, int(torch.cuda.max_memory_allocated(device)))
    finally:
        del model
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()
    return SafeFitResult(
        best_epoch=best_epoch,
        best_score=best_score,
        epochs_completed=epochs_completed,
        peak_memory_bytes=peak_memory,
        validation_metrics=best_metrics,
        validation_a=best_validation_a,
        validation_c=best_validation_c,
        calibration_a=calibration_a,
        calibration_c=calibration_c,
        test_a=test_a,
        test_c=test_c,
    )


def _cache_directory(config_sha256: str, split_sha256: str) -> Path:
    return (
        ROOT
        / "artifacts"
        / "safe_mi_v2"
        / (f"runs_{git_commit(ROOT)[:12]}_{config_sha256[:12]}_{split_sha256[:12]}")
    )


def _cache_paths(
    directory: Path,
    mode: SafeMIMode,
    fold: int,
    seed: int,
) -> dict[str, Path]:
    stem = f"{mode.name}_fold{fold}_seed{seed}"
    return {
        "metadata": directory / f"{stem}_metadata.json",
        **{
            name: directory / f"{stem}_{name}.csv"
            for name in (
                "validation_a",
                "validation_c",
                "calibration_a",
                "calibration_c",
                "test_a",
                "test_c",
            )
        },
    }


def _write_fit_cache(
    directory: Path,
    mode: SafeMIMode,
    fold: int,
    seed: int,
    result: SafeFitResult,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    paths = _cache_paths(directory, mode, fold, seed)
    hashes: dict[str, str] = {}
    for name in (
        "validation_a",
        "validation_c",
        "calibration_a",
        "calibration_c",
        "test_a",
        "test_c",
    ):
        frame = getattr(result, name)
        if frame.empty:
            continue
        frame.to_csv(paths[name], index=False, lineterminator="\n", float_format="%.10g")
        hashes[name] = sha256_file(paths[name])
    paths["metadata"].write_bytes(
        canonical_json_bytes(
            {
                "mode": asdict(mode),
                "fold": fold,
                "seed": seed,
                "best_epoch": result.best_epoch,
                "best_score": result.best_score,
                "epochs_completed": result.epochs_completed,
                "peak_memory_bytes": result.peak_memory_bytes,
                "validation_metrics": result.validation_metrics,
                "ledger_sha256": hashes,
            }
        )
    )


def _load_fit_cache(
    directory: Path,
    mode: SafeMIMode,
    fold: int,
    seed: int,
) -> SafeFitResult | None:
    paths = _cache_paths(directory, mode, fold, seed)
    if not paths["metadata"].exists():
        return None
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    if metadata.get("mode") != asdict(mode):
        raise ValueError("SAFE-MI cached mode does not match the current configuration")
    if metadata.get("fold") != fold or metadata.get("seed") != seed:
        raise ValueError("SAFE-MI cache fold/seed mismatch")
    frames: dict[str, pd.DataFrame] = {}
    for name in (
        "validation_a",
        "validation_c",
        "calibration_a",
        "calibration_c",
        "test_a",
        "test_c",
    ):
        expected = metadata["ledger_sha256"].get(name)
        if expected is None:
            frames[name] = pd.DataFrame()
            continue
        if not paths[name].exists() or sha256_file(paths[name]) != expected:
            raise ValueError(f"SAFE-MI cached ledger mismatch: {paths[name]}")
        frames[name] = pd.read_csv(paths[name], dtype={"source_id": str})
    return SafeFitResult(
        best_epoch=int(metadata["best_epoch"]),
        best_score=float(metadata["best_score"]),
        epochs_completed=int(metadata["epochs_completed"]),
        peak_memory_bytes=int(metadata["peak_memory_bytes"]),
        validation_metrics=metadata["validation_metrics"],
        **frames,
    )


def _decision_features(
    sessions: list[SessionTurns],
    embeddings: dict[tuple[int, int], np.ndarray],
    window: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    records: list[dict[str, Any]] = []
    features: list[np.ndarray] = []
    for session in sessions:
        for position in np.flatnonzero(session.next_action_targets >= 0):
            start = max(0, int(position) - window + 1)
            history = np.stack(
                [
                    embeddings[(session.transcript_id, int(session.utterance_ids[index]))]
                    for index in range(start, int(position) + 1)
                ]
            ).astype(np.float32)
            current = history[-1]
            summary = history.mean(axis=0)
            feature = np.concatenate([current, summary]).astype(np.float32)
            feature /= max(float(np.linalg.norm(feature)), 1e-8)
            records.append(
                {
                    "transcript_id": session.transcript_id,
                    "decision_utterance_id": int(session.utterance_ids[position]),
                    "source_id": session.source_id,
                    "normalized_text": normalize_text(session.texts[position]),
                    "label": LABELS[int(session.next_action_targets[position])],
                }
            )
            features.append(feature)
    if not records:
        raise ValueError("Prototype feature extraction found no Task C decisions")
    return pd.DataFrame(records), np.stack(features)


def _prototype_probabilities(
    reference_sessions: list[SessionTurns],
    query_sessions: list[SessionTurns],
    embeddings: dict[tuple[int, int], np.ndarray],
    ledger: pd.DataFrame,
    neighbours: int,
    config: dict[str, Any],
) -> np.ndarray:
    window = int(config["architecture"]["local_attention_window"])
    reference, reference_features = _decision_features(reference_sessions, embeddings, window)
    queries, query_features = _decision_features(query_sessions, embeddings, window)
    query_lookup = {
        (int(row.transcript_id), int(row.decision_utterance_id)): index
        for index, row in enumerate(queries.itertuples(index=False))
    }
    ordered_indices = np.asarray(
        [
            query_lookup[(int(row.transcript_id), int(row.decision_utterance_id))]
            for row in ledger.itertuples(index=False)
        ],
        dtype=int,
    )
    ordered_features = query_features[ordered_indices]
    ordered_queries = queries.iloc[ordered_indices].reset_index(drop=True)
    search_count = min(len(reference), max(neighbours + 64, 256))
    nearest = NearestNeighbors(
        n_neighbors=search_count,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    ).fit(reference_features)
    candidate_indices = nearest.kneighbors(
        ordered_features,
        return_distance=False,
    )
    exclude_source = bool(config["prototype_retrieval"]["exclude_same_source"])
    exclude_text = bool(config["prototype_retrieval"]["exclude_normalized_text_match"])
    output = np.empty((len(ledger), len(LABELS)), dtype=float)
    reference_sources = reference["source_id"].astype(str).to_numpy()
    reference_text = reference["normalized_text"].astype(str).to_numpy()
    reference_labels = np.asarray(
        [LABELS.index(str(value)) for value in reference["label"]], dtype=int
    )
    for row_index, (ledger_row, query_row, candidates) in enumerate(
        zip(
            ledger.itertuples(index=False),
            ordered_queries.itertuples(index=False),
            candidate_indices,
            strict=True,
        )
    ):
        accepted: list[int] = []
        query_text = str(query_row.normalized_text)
        for candidate in candidates:
            if exclude_source and reference_sources[candidate] == str(ledger_row.source_id):
                continue
            if exclude_text and reference_text[candidate] == query_text:
                continue
            accepted.append(int(candidate))
            if len(accepted) == neighbours:
                break
        if not accepted:
            output[row_index] = 1.0 / len(LABELS)
            continue
        counts = np.bincount(reference_labels[accepted], minlength=len(LABELS)).astype(float)
        counts += 0.25
        output[row_index] = counts / counts.sum()
    return output


def _mix_probabilities(
    ledger: pd.DataFrame,
    prototype_probabilities: np.ndarray,
    mixture_weight: float,
    model_name: str,
) -> pd.DataFrame:
    probability_columns = [f"prob_{label}" for label in LABELS]
    base = ledger[probability_columns].to_numpy(dtype=float)
    mixed = (1.0 - mixture_weight) * base + mixture_weight * prototype_probabilities
    result = ledger.copy()
    result["model"] = model_name
    for index, column in enumerate(probability_columns):
        result[column] = mixed[:, index]
    result["prediction"] = np.asarray(LABELS, dtype=object)[mixed.argmax(axis=1)]
    return result


def _prototype_variant(
    result: SafeFitResult,
    fit: list[SessionTurns],
    validation: list[SessionTurns],
    calibration: list[SessionTurns],
    test: list[SessionTurns],
    selection_embeddings: dict[tuple[int, int], np.ndarray],
    refit_embeddings: dict[tuple[int, int], np.ndarray],
    config: dict[str, Any],
) -> tuple[SafeFitResult, dict[str, Any]]:
    best_score = -math.inf
    best_neighbours = 0
    best_weight = 0.0
    best_frame = pd.DataFrame()
    for neighbours in config["prototype_retrieval"]["neighbours"]:
        prototype = _prototype_probabilities(
            fit,
            validation,
            selection_embeddings,
            result.validation_c,
            int(neighbours),
            config,
        )
        for weight in config["prototype_retrieval"]["mixture_weights"]:
            frame = _mix_probabilities(
                result.validation_c,
                prototype,
                float(weight),
                "r1_prototype",
            )
            metrics = evaluate_action_predictions(frame)
            score = float(metrics["source_balanced_macro_f1"]) - float(
                config["training"]["selection_brier_weight"]
            ) * float(metrics["source_balanced_brier"])
            if score > best_score + 1e-12:
                best_score = score
                best_neighbours = int(neighbours)
                best_weight = float(weight)
                best_frame = frame

    refit = fit + validation
    calibration_prototype = _prototype_probabilities(
        refit,
        calibration,
        refit_embeddings,
        result.calibration_c,
        best_neighbours,
        config,
    )
    test_prototype = _prototype_probabilities(
        refit,
        test,
        refit_embeddings,
        result.test_c,
        best_neighbours,
        config,
    )
    calibration_frame = _mix_probabilities(
        result.calibration_c,
        calibration_prototype,
        best_weight,
        "r1_prototype",
    )
    test_frame = _mix_probabilities(
        result.test_c,
        test_prototype,
        best_weight,
        "r1_prototype",
    )
    metrics = evaluate_action_predictions(best_frame)
    variant = SafeFitResult(
        best_epoch=result.best_epoch,
        best_score=best_score,
        epochs_completed=result.epochs_completed,
        peak_memory_bytes=result.peak_memory_bytes,
        validation_metrics={"task_c": metrics},
        validation_a=pd.DataFrame(),
        validation_c=best_frame,
        calibration_a=pd.DataFrame(),
        calibration_c=calibration_frame,
        test_a=pd.DataFrame(),
        test_c=test_frame,
    )
    selection = {
        "model": "r1_prototype",
        "base_model": str(result.test_c["model"].iloc[0]),
        "outer_fold": int(result.test_c["outer_fold"].iloc[0]),
        "seed": int(result.test_c["seed"].iloc[0]),
        "neighbours": best_neighbours,
        "mixture_weight": best_weight,
        "best_validation_score": best_score,
        "validation_metrics": metrics,
    }
    return variant, selection


def _concat_result_frames(
    registry: dict[tuple[str, int, int], SafeFitResult],
    model_names: set[str],
    attribute: str,
) -> pd.DataFrame:
    frames = [
        getattr(result, attribute)
        for (model, _, _), result in registry.items()
        if model in model_names and not getattr(result, attribute).empty
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _aggregate_results(
    registry: dict[tuple[str, int, int], SafeFitResult],
    model_names: set[str],
    alpha: float,
) -> dict[str, Any]:
    task_a_by_seed = _concat_result_frames(registry, model_names, "test_a")
    task_c_by_seed = _concat_result_frames(registry, model_names, "test_c")
    calibration_a_by_seed = _concat_result_frames(registry, model_names, "calibration_a")
    calibration_c_by_seed = _concat_result_frames(registry, model_names, "calibration_c")
    task_a = pd.DataFrame()
    task_c = pd.DataFrame()
    calibration_records_a: list[dict[str, Any]] = []
    calibration_records_c: list[dict[str, Any]] = []
    if not task_a_by_seed.empty:
        task_a = _ensemble_quality(task_a_by_seed)
        calibration_a = _ensemble_quality(calibration_a_by_seed)
        task_a, calibration_records_a = _calibrate_quality(calibration_a, task_a)
    if not task_c_by_seed.empty:
        task_c = _ensemble_action(task_c_by_seed)
        calibration_c = _ensemble_action(calibration_c_by_seed)
        task_c, calibration_records_c = _calibrate_action(
            calibration_c,
            task_c,
            alpha,
        )
    return {
        "task_a_by_seed": task_a_by_seed,
        "task_c_by_seed": task_c_by_seed,
        "task_a": task_a,
        "task_c": task_c,
        "task_a_metrics": {
            model: evaluate_quality_predictions(frame.reset_index(drop=True))
            for model, frame in task_a.groupby("model", sort=True)
        }
        if not task_a.empty
        else {},
        "task_c_metrics": {
            model: evaluate_action_predictions(frame.reset_index(drop=True))
            for model, frame in task_c.groupby("model", sort=True)
        }
        if not task_c.empty
        else {},
        "prediction_set_metrics": {
            model: evaluate_prediction_sets(frame.reset_index(drop=True))
            for model, frame in task_c.groupby("model", sort=True)
        }
        if not task_c.empty
        else {},
        "calibration_records": {
            "task_a": calibration_records_a,
            "task_c": calibration_records_c,
        },
    }


def _partitions_by_fold(
    sessions: list[SessionTurns],
    lookup: dict[str, int],
    config: dict[str, Any],
    fold_count: int,
) -> tuple[
    dict[
        int, tuple[list[SessionTurns], list[SessionTurns], list[SessionTurns], list[SessionTurns]]
    ],
    list[dict[str, Any]],
]:
    result: dict[
        int,
        tuple[list[SessionTurns], list[SessionTurns], list[SessionTurns], list[SessionTurns]],
    ] = {}
    records: list[dict[str, Any]] = []
    for fold in range(fold_count):
        outer_train = [session for session in sessions if lookup[session.source_id] != fold]
        test = [session for session in sessions if lookup[session.source_id] == fold]
        fit, validation, calibration, assignment = _inner_partitions(
            outer_train,
            fold,
            config,
        )
        result[fold] = (fit, validation, calibration, test)
        records.append(
            {
                "outer_fold": fold,
                "fit_source_ids": sorted({session.source_id for session in fit}),
                "validation_source_ids": sorted({session.source_id for session in validation}),
                "calibration_source_ids": sorted({session.source_id for session in calibration}),
                "test_source_ids": sorted({session.source_id for session in test}),
                "fit_transcripts": len(fit),
                "validation_transcripts": len(validation),
                "calibration_transcripts": len(calibration),
                "test_transcripts": len(test),
                "inner_assignment_sha256": hashlib.sha256(
                    canonical_json_bytes(assignment)
                ).hexdigest(),
            }
        )
    return result, records


def _execute_modes(
    corpus: Corpus,
    partitions: dict[
        int,
        tuple[list[SessionTurns], list[SessionTurns], list[SessionTurns], list[SessionTurns]],
    ],
    modes: list[SafeMIMode],
    seeds: list[int],
    frozen_embeddings: dict[tuple[int, int], np.ndarray],
    config: dict[str, Any],
    cache_dir: Path,
    registry: dict[tuple[str, int, int], SafeFitResult],
    selections: list[dict[str, Any]],
) -> None:
    for fold, (fit, validation, calibration, test) in partitions.items():
        print(
            f"SAFE-MI outer fold {fold}: fit={len(fit)}, validation={len(validation)}, "
            f"calibration={len(calibration)}, test={len(test)} sessions",
            flush=True,
        )
        for seed in seeds:
            adapted_selection: dict[tuple[int, int], np.ndarray] | None = None
            adapted_refit: dict[tuple[int, int], np.ndarray] | None = None
            for mode in modes:
                key = (mode.name, fold, seed)
                if key in registry:
                    continue
                result = _load_fit_cache(cache_dir, mode, fold, seed)
                if result is None:
                    if mode.encoder_variant == "adapted":
                        if adapted_selection is None:
                            adapted_selection = train_and_extract_adapted_embeddings(
                                corpus,
                                fit,
                                config,
                                fold,
                                seed,
                                "selection_fit",
                            )
                        if adapted_refit is None:
                            adapted_refit = train_and_extract_adapted_embeddings(
                                corpus,
                                fit + validation,
                                config,
                                fold,
                                seed,
                                "refit_fit_plus_validation",
                            )
                        selection_embeddings = adapted_selection
                        refit_embeddings = adapted_refit
                    else:
                        selection_embeddings = frozen_embeddings
                        refit_embeddings = frozen_embeddings
                    result = _fit_one(
                        fit,
                        validation,
                        calibration,
                        test,
                        selection_embeddings,
                        refit_embeddings,
                        config,
                        mode,
                        fold,
                        seed,
                    )
                    _write_fit_cache(cache_dir, mode, fold, seed, result)
                else:
                    print(
                        f"SAFE-MI cache hit: {mode.name}/fold={fold}/seed={seed}",
                        flush=True,
                    )
                registry[key] = result
                selections.append(
                    {
                        "model": mode.name,
                        "stage": next(
                            value["stage"]
                            for value in config["models"]
                            if value["model"] == mode.name
                        ),
                        "outer_fold": fold,
                        "seed": seed,
                        "best_epoch": result.best_epoch,
                        "best_validation_score": result.best_score,
                        "epochs_completed": result.epochs_completed,
                        "peak_memory_bytes": result.peak_memory_bytes,
                        "validation_metrics": result.validation_metrics,
                    }
                )


def _fold_rejections(
    task_c: pd.DataFrame,
    candidate: str,
    baseline: str,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    gate = protocol["screen_stopping_gate"]
    rows: list[dict[str, Any]] = []
    for fold in sorted(task_c["outer_fold"].unique()):
        candidate_frame = task_c[task_c["model"].eq(candidate) & task_c["outer_fold"].eq(fold)]
        baseline_frame = task_c[task_c["model"].eq(baseline) & task_c["outer_fold"].eq(fold)]
        candidate_metrics = evaluate_action_predictions(candidate_frame)
        baseline_metrics = evaluate_action_predictions(baseline_frame)
        f1_delta = float(
            candidate_metrics["source_balanced_macro_f1"]
            - baseline_metrics["source_balanced_macro_f1"]
        )
        brier_delta = float(
            candidate_metrics["source_balanced_brier"] - baseline_metrics["source_balanced_brier"]
        )
        bad = bool(
            f1_delta < -float(gate["maximum_task_c_macro_f1_degradation"])
            or brier_delta > float(gate["maximum_task_c_brier_degradation"])
        )
        rows.append(
            {
                "outer_fold": int(fold),
                "macro_f1_delta": f1_delta,
                "brier_delta": brier_delta,
                "bad": bad,
            }
        )
    bad_count = sum(bool(row["bad"]) for row in rows)
    return {
        "candidate": candidate,
        "baseline": baseline,
        "folds": rows,
        "bad_fold_count": bad_count,
        "rejected": bad_count >= int(gate["bad_fold_count_for_rejection"]),
    }


def _choose_screen_finalists(
    aggregate: dict[str, Any],
    modes: dict[str, SafeMIMode],
    protocol: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    baseline = "c0_frozen_gru"
    c_metrics = aggregate["task_c_metrics"]
    a_metrics = aggregate["task_a_metrics"]
    rejections = {
        model: _fold_rejections(aggregate["task_c"], model, baseline, protocol)
        for model in c_metrics
        if model != baseline
    }
    c_only_candidates = [
        model
        for model in c_metrics
        if model != baseline and (model == "r1_prototype" or not modes[model].task_a_loss)
    ]
    eligible_c = [model for model in c_only_candidates if not rejections[model]["rejected"]]
    c_pool = eligible_c or c_only_candidates
    best_c = max(
        c_pool,
        key=lambda model: (
            float(c_metrics[model]["source_balanced_macro_f1"]),
            -float(c_metrics[model]["source_balanced_brier"]),
        ),
    )

    baseline_c = c_metrics[baseline]
    joint_candidates = [
        model for model in c_metrics if modes.get(model, None) and modes[model].task_a_loss
    ]

    def joint_key(model: str) -> tuple[bool, float, float, float]:
        c_value = c_metrics[model]
        c_preserved = float(c_value["source_balanced_macro_f1"]) - float(
            baseline_c["source_balanced_macro_f1"]
        ) >= -float(protocol["final_exploratory_gate"]["joint_maximum_task_c_macro_f1_degradation"])
        a_value = a_metrics[model]["t10"]
        return (
            c_preserved,
            float(c_value["source_balanced_macro_f1"]),
            float(a_value["source_balanced_balanced_accuracy"]),
            -float(a_value["source_balanced_brier"]),
        )

    best_joint = max(joint_candidates, key=joint_key)
    diagnostics = {
        "baseline": baseline,
        "fold_rejections": rejections,
        "eligible_c_only_candidates": eligible_c,
        "selected_c_finalist": best_c,
        "selected_joint_finalist": best_joint,
        "selection_rule": "C preservation/macro-F1 first, then Task A balanced accuracy and Brier",
    }
    return best_c, best_joint, diagnostics


def _interval(values: pd.Series) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "low": float(values.quantile(0.025)),
        "high": float(values.quantile(0.975)),
    }


def _paired_bootstrap(
    candidate: str,
    task_a: pd.DataFrame,
    task_c: pd.DataFrame,
    baseline_a: pd.DataFrame,
    baseline_c_name: str,
    resamples: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidate_c = task_c[task_c["model"].eq(candidate)].sort_values(
        ["source_id", "transcript_id", "target_utterance_id"], kind="stable"
    )
    baseline_c = task_c[task_c["model"].eq(baseline_c_name)].sort_values(
        ["source_id", "transcript_id", "target_utterance_id"], kind="stable"
    )
    c_keys = ["source_id", "transcript_id", "target_utterance_id", "label"]
    if (
        not candidate_c[c_keys]
        .reset_index(drop=True)
        .equals(baseline_c[c_keys].reset_index(drop=True))
    ):
        raise ValueError("SAFE-MI candidate and Task C baseline ledgers do not align")
    candidate_a = task_a[
        task_a["model"].eq(candidate) & task_a["checkpoint"].eq("t10")
    ].sort_values(["source_id", "transcript_id"], kind="stable")
    baseline_a_t10 = baseline_a[baseline_a["checkpoint"].eq("t10")].sort_values(
        ["source_id", "transcript_id"], kind="stable"
    )
    if not candidate_a.empty:
        a_keys = ["source_id", "transcript_id", "checkpoint", "label"]
        if (
            not candidate_a[a_keys]
            .reset_index(drop=True)
            .equals(baseline_a_t10[a_keys].reset_index(drop=True))
        ):
            raise ValueError("SAFE-MI candidate and Task A baseline ledgers do not align")

    c_probability_columns = [f"prob_{label}" for label in LABELS]
    candidate_c_probabilities = candidate_c[c_probability_columns].to_numpy(dtype=float)
    baseline_c_probabilities = baseline_c[c_probability_columns].to_numpy(dtype=float)
    c_targets = np.asarray([LABELS.index(str(value)) for value in candidate_c["label"]])
    c_one_hot = np.eye(len(LABELS))[c_targets]
    c_sources = candidate_c["source_id"].astype(str).unique()
    a_sources = candidate_a["source_id"].astype(str).unique() if not candidate_a.empty else []
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for draw in range(resamples):
        sampled_c = rng.choice(c_sources, size=len(c_sources), replace=True)
        c_weights = _source_bootstrap_weights(candidate_c, sampled_c)
        active_c = c_weights > 0
        candidate_c_prediction = np.asarray(LABELS, dtype=object)[
            candidate_c_probabilities.argmax(axis=1)
        ]
        baseline_c_prediction = np.asarray(LABELS, dtype=object)[
            baseline_c_probabilities.argmax(axis=1)
        ]
        candidate_c_f1 = f1_score(
            candidate_c.loc[active_c, "label"],
            candidate_c_prediction[active_c],
            labels=list(LABELS),
            average="macro",
            sample_weight=c_weights[active_c],
            zero_division=0,
        )
        baseline_c_f1 = f1_score(
            baseline_c.loc[active_c, "label"],
            baseline_c_prediction[active_c],
            labels=list(LABELS),
            average="macro",
            sample_weight=c_weights[active_c],
            zero_division=0,
        )
        candidate_c_brier = np.average(
            np.square(candidate_c_probabilities - c_one_hot).sum(axis=1)[active_c],
            weights=c_weights[active_c],
        )
        baseline_c_brier = np.average(
            np.square(baseline_c_probabilities - c_one_hot).sum(axis=1)[active_c],
            weights=c_weights[active_c],
        )
        row: dict[str, Any] = {
            "candidate": candidate,
            "draw": draw,
            "task_c_macro_f1_delta": candidate_c_f1 - baseline_c_f1,
            "task_c_brier_delta": candidate_c_brier - baseline_c_brier,
            "task_a_t10_balanced_accuracy_delta": math.nan,
            "task_a_t10_brier_delta": math.nan,
        }
        if not candidate_a.empty:
            sampled_a = rng.choice(a_sources, size=len(a_sources), replace=True)
            a_weights = _source_bootstrap_weights(candidate_a, sampled_a)
            active_a = a_weights > 0
            candidate_prediction = np.where(
                candidate_a["prob_low"].to_numpy(dtype=float) >= 0.5,
                "low",
                "high",
            )
            baseline_prediction = np.where(
                baseline_a_t10["prob_low"].to_numpy(dtype=float) >= 0.5,
                "low",
                "high",
            )
            candidate_score = balanced_accuracy_score(
                candidate_a.loc[active_a, "label"],
                candidate_prediction[active_a],
                sample_weight=a_weights[active_a],
            )
            baseline_score = balanced_accuracy_score(
                baseline_a_t10.loc[active_a, "label"],
                baseline_prediction[active_a],
                sample_weight=a_weights[active_a],
            )
            targets = candidate_a["label"].eq("low").to_numpy(dtype=int)
            candidate_brier = np.average(
                np.square(candidate_a["prob_low"].to_numpy(dtype=float) - targets)[active_a],
                weights=a_weights[active_a],
            )
            baseline_brier = np.average(
                np.square(baseline_a_t10["prob_low"].to_numpy(dtype=float) - targets)[active_a],
                weights=a_weights[active_a],
            )
            row["task_a_t10_balanced_accuracy_delta"] = candidate_score - baseline_score
            row["task_a_t10_brier_delta"] = candidate_brier - baseline_brier
        rows.append(row)
    draws = pd.DataFrame(rows)
    intervals = {
        column: _interval(draws[column].dropna())
        for column in (
            "task_c_macro_f1_delta",
            "task_c_brier_delta",
            "task_a_t10_balanced_accuracy_delta",
            "task_a_t10_brier_delta",
        )
        if draws[column].notna().any()
    }
    return draws, {
        "candidate": candidate,
        "task_c_baseline": baseline_c_name,
        "task_a_baseline": "qtrace_v1/a_only" if not candidate_a.empty else None,
        "resamples": resamples,
        "seed": seed,
        "cluster_unit": "source_id",
        "intervals": intervals,
    }


def _seed_deltas(
    candidate: str,
    candidate_a: pd.DataFrame,
    candidate_c: pd.DataFrame,
    baseline_a: pd.DataFrame,
    baseline_c_name: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {"task_a": {}, "task_c": {}}
    for seed in sorted(candidate_c.loc[candidate_c["model"].eq(candidate), "seed"].unique()):
        c_value = candidate_c[candidate_c["model"].eq(candidate) & candidate_c["seed"].eq(seed)]
        c_base = candidate_c[
            candidate_c["model"].eq(baseline_c_name) & candidate_c["seed"].eq(seed)
        ]
        result["task_c"][str(int(seed))] = float(
            evaluate_action_predictions(c_value)["source_balanced_macro_f1"]
            - evaluate_action_predictions(c_base)["source_balanced_macro_f1"]
        )
        a_value = candidate_a[
            candidate_a["model"].eq(candidate)
            & candidate_a["seed"].eq(seed)
            & candidate_a["checkpoint"].eq("t10")
        ]
        if a_value.empty:
            continue
        a_base = baseline_a[
            baseline_a["model"].eq("a_only")
            & baseline_a["seed"].eq(seed)
            & baseline_a["checkpoint"].eq("t10")
        ]
        result["task_a"][str(int(seed))] = float(
            evaluate_quality_predictions(a_value)["t10"]["source_balanced_balanced_accuracy"]
            - evaluate_quality_predictions(a_base)["t10"]["source_balanced_balanced_accuracy"]
        )
    result["task_a_positive_seed_count"] = sum(value > 0 for value in result["task_a"].values())
    result["task_c_positive_seed_count"] = sum(value > 0 for value in result["task_c"].values())
    return result


def _candidate_gate(
    candidate: str,
    aggregate: dict[str, Any],
    baseline_a_metrics: dict[str, Any],
    inference: dict[str, Any],
    seed_deltas: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    config = protocol["final_exploratory_gate"]
    baseline_c = aggregate["task_c_metrics"]["c0_frozen_gru"]
    candidate_c = aggregate["task_c_metrics"][candidate]
    c_f1_delta = float(
        candidate_c["source_balanced_macro_f1"] - baseline_c["source_balanced_macro_f1"]
    )
    c_brier_delta = float(
        candidate_c["source_balanced_brier"] - baseline_c["source_balanced_brier"]
    )
    c_interval = inference["intervals"]["task_c_macro_f1_delta"]
    predictions = aggregate["task_c"]
    predicted_classes = set(
        predictions.loc[predictions["model"].eq(candidate), "prediction"].astype(str)
    )
    set_metrics = aggregate["prediction_set_metrics"][candidate]
    has_task_a = candidate in aggregate["task_a_metrics"]
    if has_task_a:
        candidate_a = aggregate["task_a_metrics"][candidate]["t10"]
        baseline_a = baseline_a_metrics["t10"]
        a_delta = float(
            candidate_a["source_balanced_balanced_accuracy"]
            - baseline_a["source_balanced_balanced_accuracy"]
        )
        a_brier_delta = float(
            candidate_a["source_balanced_brier"] - baseline_a["source_balanced_brier"]
        )
        components = {
            "joint_c_noninferiority": c_f1_delta
            >= -float(config["joint_maximum_task_c_macro_f1_degradation"]),
            "joint_c_brier": c_brier_delta <= float(config["maximum_task_c_brier_degradation"]),
            "joint_a_minimum_gain": a_delta
            >= float(config["joint_minimum_task_a_balanced_accuracy_gain"]),
            "joint_a_brier": a_brier_delta
            <= float(config["joint_maximum_task_a_brier_degradation"]),
        }
    else:
        a_delta = None
        a_brier_delta = None
        components = {
            "task_c_minimum_gain": c_f1_delta >= float(config["task_c_minimum_macro_f1_gain"]),
            "task_c_positive_interval": c_interval["low"] > 0,
            "task_c_brier": c_brier_delta <= float(config["maximum_task_c_brier_degradation"]),
            "task_c_positive_seeds": seed_deltas["task_c_positive_seed_count"]
            >= int(config["minimum_positive_seed_count"]),
        }
    components.update(
        {
            "all_classes_predicted": predicted_classes == set(LABELS),
            "prediction_set_coverage": float(set_metrics["source_balanced_coverage"])
            >= float(config["prediction_set_minimum_coverage"]),
            "prediction_set_efficiency": float(set_metrics["source_balanced_mean_set_size"])
            <= float(config["prediction_set_maximum_mean_size"]),
        }
    )
    return {
        "candidate": candidate,
        "exploratory_not_confirmatory": True,
        "task_c_macro_f1_delta": c_f1_delta,
        "task_c_brier_delta": c_brier_delta,
        "task_a_t10_balanced_accuracy_delta": a_delta,
        "task_a_t10_brier_delta": a_brier_delta,
        "predicted_classes": sorted(predicted_classes),
        "seed_deltas": seed_deltas,
        "components": components,
        "pass": all(components.values()),
    }


def _generate_prototype_results(
    corpus: Corpus,
    base_model: str,
    seeds: list[int],
    partitions: dict[
        int,
        tuple[list[SessionTurns], list[SessionTurns], list[SessionTurns], list[SessionTurns]],
    ],
    frozen_embeddings: dict[tuple[int, int], np.ndarray],
    config: dict[str, Any],
    modes: dict[str, SafeMIMode],
    registry: dict[tuple[str, int, int], SafeFitResult],
    prototype_selections: list[dict[str, Any]],
) -> None:
    mode = modes[base_model]
    for fold, (fit, validation, calibration, test) in partitions.items():
        for seed in seeds:
            key = ("r1_prototype", fold, seed)
            if key in registry:
                continue
            base_result = registry[(base_model, fold, seed)]
            if mode.encoder_variant == "adapted":
                selection_embeddings = train_and_extract_adapted_embeddings(
                    corpus,
                    fit,
                    config,
                    fold,
                    seed,
                    "selection_fit",
                )
                refit_embeddings = train_and_extract_adapted_embeddings(
                    corpus,
                    fit + validation,
                    config,
                    fold,
                    seed,
                    "refit_fit_plus_validation",
                )
            else:
                selection_embeddings = frozen_embeddings
                refit_embeddings = frozen_embeddings
            variant, selection = _prototype_variant(
                base_result,
                fit,
                validation,
                calibration,
                test,
                selection_embeddings,
                refit_embeddings,
                config,
            )
            registry[key] = variant
            prototype_selections.append(selection)


def run_safe_mi(
    corpus: Corpus,
    split_manifest: dict[str, Any],
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute the bounded SAFE-MI screen and five-seed finalist campaign."""

    started = time.perf_counter()
    output_dir = output_dir or RESEARCH_RESULTS / "safe_mi_v2"
    if (output_dir / "summary.json").exists():
        validate_safe_mi_evidence(output_dir)
        return read_json(output_dir / "summary.json")
    protocol, config = _require_registered_state(corpus, split_manifest)
    sessions = build_session_turns(corpus, (3, 5, 10, 20))
    lookup = fold_lookup(split_manifest)
    partitions, partition_records = _partitions_by_fold(
        sessions,
        lookup,
        config,
        len(split_manifest["folds"]),
    )
    frozen_embeddings = extract_turn_embeddings(corpus, config)
    mode_lookup = {
        value["model"]: mode_from_config(config, value["model"]) for value in config["models"]
    }
    config_sha256 = sha256_file(SAFE_MI_CONFIG)
    cache_dir = _cache_directory(config_sha256, split_manifest["manifest_sha256"])
    registry: dict[tuple[str, int, int], SafeFitResult] = {}
    selections: list[dict[str, Any]] = []
    prototype_selections: list[dict[str, Any]] = []

    screen_modes = list(mode_lookup.values())
    screen_seeds = [int(value) for value in config["training"]["screen_seeds"]]
    _execute_modes(
        corpus,
        partitions,
        screen_modes,
        screen_seeds,
        frozen_embeddings,
        config,
        cache_dir,
        registry,
        selections,
    )

    stage1_names = {value["model"] for value in config["models"] if value["stage"] == "c_screen"}
    stage1 = _aggregate_results(
        registry,
        stage1_names,
        float(config["calibration"]["prediction_set_alpha"]),
    )
    prototype_base = max(
        stage1["task_c_metrics"],
        key=lambda model: (
            float(stage1["task_c_metrics"][model]["source_balanced_macro_f1"]),
            -float(stage1["task_c_metrics"][model]["source_balanced_brier"]),
        ),
    )
    _generate_prototype_results(
        corpus,
        prototype_base,
        screen_seeds,
        partitions,
        frozen_embeddings,
        config,
        mode_lookup,
        registry,
        prototype_selections,
    )

    screen_names = set(mode_lookup) | {"r1_prototype"}
    screen = _aggregate_results(
        registry,
        screen_names,
        float(config["calibration"]["prediction_set_alpha"]),
    )
    best_c, best_joint, screen_diagnostics = _choose_screen_finalists(
        screen,
        mode_lookup,
        protocol,
    )
    screen_diagnostics["prototype_base_model"] = prototype_base
    print(
        f"SAFE-MI screen finalists: C={best_c}; joint={best_joint}",
        flush=True,
    )

    final_seeds = [int(value) for value in config["training"]["final_seeds"]]
    remaining_seeds = [value for value in final_seeds if value not in screen_seeds]
    required_mode_names = {"c0_frozen_gru", best_joint}
    if best_c == "r1_prototype":
        required_mode_names.add(prototype_base)
    else:
        required_mode_names.add(best_c)
    _execute_modes(
        corpus,
        partitions,
        [mode_lookup[name] for name in sorted(required_mode_names)],
        remaining_seeds,
        frozen_embeddings,
        config,
        cache_dir,
        registry,
        selections,
    )
    if best_c == "r1_prototype":
        _generate_prototype_results(
            corpus,
            prototype_base,
            remaining_seeds,
            partitions,
            frozen_embeddings,
            config,
            mode_lookup,
            registry,
            prototype_selections,
        )

    final_names = {"c0_frozen_gru", best_c, best_joint}
    final = _aggregate_results(
        registry,
        final_names,
        float(config["calibration"]["prediction_set_alpha"]),
    )
    qtrace_root = RESEARCH_RESULTS / "ac_v1" / "qtrace_mi"
    baseline_a = pd.read_csv(
        qtrace_root / "task_a_predictions_seed_ensemble.csv",
        dtype={"source_id": str},
    )
    baseline_a = baseline_a[baseline_a["model"].eq("a_only")].reset_index(drop=True)
    baseline_a_by_seed = pd.read_csv(
        qtrace_root / "task_a_predictions_by_seed.csv",
        dtype={"source_id": str},
    )
    baseline_a_metrics = evaluate_quality_predictions(baseline_a)

    bootstrap_frames: list[pd.DataFrame] = []
    inference: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    seed_delta_records: dict[str, Any] = {}
    for candidate in sorted(final_names - {"c0_frozen_gru"}):
        draws, candidate_inference = _paired_bootstrap(
            candidate,
            final["task_a"],
            final["task_c"],
            baseline_a,
            "c0_frozen_gru",
            int(config["inference"]["bootstrap_resamples"]),
            int(config["inference"]["bootstrap_seed"]),
        )
        bootstrap_frames.append(draws)
        inference[candidate] = candidate_inference
        candidate_seed_deltas = _seed_deltas(
            candidate,
            final["task_a_by_seed"],
            final["task_c_by_seed"],
            baseline_a_by_seed,
            "c0_frozen_gru",
        )
        seed_delta_records[candidate] = candidate_seed_deltas
        gates[candidate] = _candidate_gate(
            candidate,
            final,
            baseline_a_metrics,
            candidate_inference,
            candidate_seed_deltas,
            protocol,
        )
    bootstrap_draws = pd.concat(bootstrap_frames, ignore_index=True)

    payloads = {
        "screen_task_a_predictions_by_seed.csv": _csv_payload(screen["task_a_by_seed"]),
        "screen_task_c_predictions_by_seed.csv": _csv_payload(screen["task_c_by_seed"]),
        "screen_task_a_predictions_seed_ensemble.csv": _csv_payload(screen["task_a"]),
        "screen_task_c_predictions_seed_ensemble.csv": _csv_payload(screen["task_c"]),
        "final_task_a_predictions_by_seed.csv": _csv_payload(final["task_a_by_seed"]),
        "final_task_c_predictions_by_seed.csv": _csv_payload(final["task_c_by_seed"]),
        "final_task_a_predictions_seed_ensemble.csv": _csv_payload(final["task_a"]),
        "final_task_c_predictions_seed_ensemble.csv": _csv_payload(final["task_c"]),
        "bootstrap_draws.csv": _csv_payload(bootstrap_draws),
        "selection.json": canonical_json_bytes(
            {
                "neural_fits": selections,
                "prototype": prototype_selections,
                "screen": screen_diagnostics,
                "final_models": sorted(final_names),
            }
        ),
        "partitions.json": canonical_json_bytes({"partitions": partition_records}),
        "calibration.json": canonical_json_bytes(
            {"screen": screen["calibration_records"], "final": final["calibration_records"]}
        ),
    }
    hashes = {
        name: write_create_only(output_dir / name, payload) for name, payload in payloads.items()
    }
    summary = {
        "result_id": "annomi-safe-mi-exploratory-source-cv-v2",
        "status": "complete_exploratory_not_confirmatory",
        "protocol_id": protocol["protocol_id"],
        "config_id": config["config_id"],
        "code_commit": git_commit(ROOT),
        "config_sha256": config_sha256,
        "protocol_sha256": sha256_file(SAFE_MI_PROTOCOL),
        "split_manifest_sha256": split_manifest["manifest_sha256"],
        "dataset_sha256": {
            "simple": sha256_file(SIMPLE_DATA),
            "full": sha256_file(FULL_DATA),
        },
        "screen": {
            "task_a_metrics": screen["task_a_metrics"],
            "task_c_metrics": screen["task_c_metrics"],
            "prediction_set_metrics": screen["prediction_set_metrics"],
            "diagnostics": screen_diagnostics,
        },
        "finalists": sorted(final_names),
        "final": {
            "task_a_metrics": final["task_a_metrics"],
            "task_c_metrics": final["task_c_metrics"],
            "prediction_set_metrics": final["prediction_set_metrics"],
            "seed_deltas": seed_delta_records,
            "paired_source_bootstrap": inference,
            "exploratory_gates": gates,
        },
        "external_confirmation": {
            "completed": False,
            "reason": "MI-TAGS must first pass an overlap audit and locked-split registration.",
        },
        "evidence_sha256": hashes,
        "runtime_environment": _runtime_environment(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_create_only(output_dir / "summary.json", canonical_json_bytes(summary))
    validate_safe_mi_evidence(output_dir)
    return summary


def _assert_close(actual: float, expected: float, name: str) -> None:
    if not np.isclose(actual, expected, atol=1e-8):
        raise ValueError(f"SAFE-MI metric reconstruction mismatch: {name}")


def validate_safe_mi_evidence(output_dir: Path) -> None:
    summary = read_json(output_dir / "summary.json")
    if summary["status"] != "complete_exploratory_not_confirmatory":
        raise ValueError("SAFE-MI evidence lost its exploratory status")
    for name, expected_hash in summary["evidence_sha256"].items():
        path = output_dir / name
        if not path.exists() or sha256_file(path) != expected_hash:
            raise ValueError(f"SAFE-MI evidence hash mismatch: {name}")
    task_a = pd.read_csv(
        output_dir / "final_task_a_predictions_seed_ensemble.csv",
        dtype={"source_id": str},
    )
    task_c = pd.read_csv(
        output_dir / "final_task_c_predictions_seed_ensemble.csv",
        dtype={"source_id": str},
    )
    for model, expected in summary["final"]["task_a_metrics"].items():
        actual = evaluate_quality_predictions(task_a[task_a["model"].eq(model)])
        for checkpoint, values in expected.items():
            for metric in (
                "source_balanced_balanced_accuracy",
                "source_balanced_brier",
                "source_balanced_log_loss",
            ):
                _assert_close(float(actual[checkpoint][metric]), float(values[metric]), metric)
    for model, expected in summary["final"]["task_c_metrics"].items():
        frame = task_c[task_c["model"].eq(model)]
        actual = evaluate_action_predictions(frame)
        for metric in (
            "source_balanced_macro_f1",
            "source_balanced_brier",
            "source_balanced_log_loss",
        ):
            _assert_close(float(actual[metric]), float(expected[metric]), metric)
        set_actual = evaluate_prediction_sets(frame)
        set_expected = summary["final"]["prediction_set_metrics"][model]
        for metric in (
            "source_balanced_coverage",
            "source_balanced_mean_set_size",
            "source_balanced_singleton_rate",
        ):
            _assert_close(float(set_actual[metric]), float(set_expected[metric]), metric)
    partitions = read_json(output_dir / "partitions.json")["partitions"]
    for record in partitions:
        groups = [
            set(record["fit_source_ids"]),
            set(record["validation_source_ids"]),
            set(record["calibration_source_ids"]),
            set(record["test_source_ids"]),
        ]
        for first in range(len(groups)):
            for second in range(first + 1, len(groups)):
                if groups[first] & groups[second]:
                    raise ValueError("SAFE-MI evidence contains overlapping source partitions")


def run_safe_mi_smoke(
    corpus: Corpus,
    split_manifest: dict[str, Any],
) -> dict[str, Any]:
    protocol, config = _require_registered_state(corpus, split_manifest)
    output_path = RESEARCH_RESULTS / "gate1" / "safe_mi_smoke_v2.json"
    if output_path.exists():
        return read_json(output_path)
    sessions = build_session_turns(corpus, (3, 5, 10, 20))
    lookup = fold_lookup(split_manifest)
    outer_train = [session for session in sessions if lookup[session.source_id] != 0]
    fit, _, _, _ = _inner_partitions(outer_train, 0, config)
    embeddings = extract_turn_embeddings(corpus, config)
    mode = mode_from_config(config, "c2_frozen_attention")
    device = _require_device()
    _seed_everything(1907)
    model = _make_model(fit, embeddings, config, mode, device)
    optimizer = _optimizer(model, config)
    loader = _loader(
        fit,
        embeddings,
        int(config["training"]["batch_size_sessions"]),
        shuffle=True,
        seed=1907,
    )
    batch = _move_batch(next(iter(loader)), device)
    weights = _safe_class_weights(fit, device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(batch["embeddings"], batch["roles"], batch["lengths"])
        loss, components = safe_mi_loss(
            output,
            batch,
            mode,
            config["training"],
            weights,
        )
    loss.backward()
    optimizer.step()
    payload = {
        "gate_id": "annomi-safe-mi-v2-cuda-smoke",
        "status": "pass",
        "engineering_gate_not_performance_result": True,
        "outer_test_labels_used": False,
        "exploratory_protocol": protocol["evidence_boundary"],
        "code_commit": git_commit(ROOT),
        "config_sha256": sha256_file(SAFE_MI_CONFIG),
        "loss": float(loss.detach()),
        "loss_components": components,
        "transition_residual_initially_zero": float(output["transition_strength"]) == 0.0,
        "probabilities_finite": bool(torch.isfinite(output["action_probabilities"]).all()),
        "maximum_probability_sum_error": float(
            (output["action_probabilities"].float().sum(dim=-1) - 1).abs().max()
        ),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "elapsed_seconds": time.perf_counter() - started,
        "runtime_environment": _runtime_environment(),
    }
    del model
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    write_create_only(output_path, canonical_json_bytes(payload))
    return payload
