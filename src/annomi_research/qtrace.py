from __future__ import annotations

import gc
import hashlib
import io
import json
import math
import os
import platform
import random
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from .ac_data import SessionTurns, build_session_turns
from .ac_metrics import (
    add_prediction_sets,
    apply_binary_temperature,
    apply_multiclass_temperature,
    aps_scores,
    evaluate_action_predictions,
    evaluate_prediction_sets,
    evaluate_quality_predictions,
    fit_binary_temperature,
    fit_multiclass_temperature,
    source_crc_threshold,
)
from .constants import (
    AC_PROTOCOL,
    ARTIFACTS,
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
from .data import Corpus, normalize_text
from .io import canonical_json_bytes, git_commit, read_json, sha256_file, write_create_only
from .metrics import source_balanced_weights
from .qtrace_model import QTraceMode, QTraceModel, mode_from_config, qtrace_loss
from .splits import fold_lookup, validate_source_folds


@dataclass(frozen=True)
class FitResult:
    best_epoch: int
    best_score: float
    epochs_completed: int
    peak_memory_bytes: int
    validation_metrics: dict[str, Any]
    calibration_a: pd.DataFrame
    calibration_c: pd.DataFrame
    test_a: pd.DataFrame
    test_c: pd.DataFrame


class _SessionDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        sessions: list[SessionTurns],
        embeddings: dict[tuple[int, int], np.ndarray],
    ) -> None:
        self.sessions = sessions
        self.embeddings = embeddings
        source_counts = Counter(session.source_id for session in sessions)
        raw_weights = np.asarray(
            [1.0 / source_counts[session.source_id] for session in sessions], dtype=np.float32
        )
        self.session_weights = raw_weights / raw_weights.mean()

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, index: int) -> dict[str, Any]:
        session = self.sessions[index]
        values = np.stack(
            [
                self.embeddings[(session.transcript_id, int(utterance_id))]
                for utterance_id in session.utterance_ids
            ]
        ).astype(np.float32, copy=False)
        quality_mask = np.zeros(len(values), dtype=bool)
        quality_mask[list(set(session.quality_positions.values()))] = True
        return {
            "session": session,
            "embeddings": values,
            "roles": session.roles,
            "therapist_labels": session.therapist_labels,
            "client_labels": session.client_labels,
            "next_action_targets": session.next_action_targets,
            "quality_mask": quality_mask,
            "quality": session.quality,
            "session_weight": self.session_weights[index],
        }


def _collate_sessions(items: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = np.asarray([len(item["roles"]) for item in items], dtype=np.int64)
    maximum = int(lengths.max())
    embedding_size = int(items[0]["embeddings"].shape[1])
    batch_size = len(items)
    embeddings = np.zeros((batch_size, maximum, embedding_size), dtype=np.float32)
    roles = np.zeros((batch_size, maximum), dtype=np.int64)
    therapist = np.full((batch_size, maximum), -100, dtype=np.int64)
    client = np.full((batch_size, maximum), -100, dtype=np.int64)
    actions = np.full((batch_size, maximum), -100, dtype=np.int64)
    quality_mask = np.zeros((batch_size, maximum), dtype=bool)
    for row, item in enumerate(items):
        length = lengths[row]
        embeddings[row, :length] = item["embeddings"]
        roles[row, :length] = item["roles"]
        therapist[row, :length] = item["therapist_labels"]
        client[row, :length] = item["client_labels"]
        actions[row, :length] = item["next_action_targets"]
        quality_mask[row, :length] = item["quality_mask"]
    return {
        "sessions": [item["session"] for item in items],
        "embeddings": torch.from_numpy(embeddings),
        "roles": torch.from_numpy(roles),
        "lengths": torch.from_numpy(lengths),
        "therapist_labels": torch.from_numpy(therapist),
        "client_labels": torch.from_numpy(client),
        "next_action_targets": torch.from_numpy(actions),
        "quality_mask": torch.from_numpy(quality_mask),
        "quality": torch.tensor([item["quality"] for item in items], dtype=torch.long),
        "session_weights": torch.tensor(
            [item["session_weight"] for item in items], dtype=torch.float32
        ),
    }


def _git_is_clean() -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not completed.stdout.strip()


def _runtime_environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "bf16_supported": bool(torch.cuda.is_bf16_supported())
        if torch.cuda.is_available()
        else False,
    }


def _require_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("Q-TRACE-MI requires a CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Q-TRACE-MI requires BF16 support")
    return torch.device("cuda", 0)


def _seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _require_registered_state(
    split_manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = read_json(AC_PROTOCOL)
    config = read_json(QTRACE_CONFIG)
    if protocol["status"] != "locked_before_task_ac_evaluation":
        raise ValueError("Task A/C protocol is not prospectively locked")
    if config["status"] != "registered_before_qtrace_neural_evaluation":
        raise ValueError("Q-TRACE configuration is not prospectively registered")
    if config["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("Q-TRACE configuration and Task A/C protocol disagree")
    validate_source_folds(load_corpus_for_validation(), split_manifest)
    for data_path, manifest_path in (
        (SIMPLE_DATA, SIMPLE_MANIFEST),
        (FULL_DATA, FULL_MANIFEST),
    ):
        if sha256_file(data_path) != read_json(manifest_path)["sha256"]:
            raise ValueError(f"Dataset hash mismatch: {data_path}")
    if not _git_is_clean():
        raise RuntimeError(
            "Commit tracked code and configuration before generating Q-TRACE evidence"
        )
    return protocol, config


def load_corpus_for_validation() -> Corpus:
    # Local import prevents a circular import in lightweight unit tests.
    from .data import load_corpus

    return load_corpus(SIMPLE_DATA, FULL_DATA)


def _embedding_cache_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    encoder = config["pretrained_encoder"]
    identity = hashlib.sha256(
        canonical_json_bytes(
            {
                "dataset": sha256_file(SIMPLE_DATA),
                "model": encoder["model_id"],
                "revision": encoder["revision"],
                "max_length": encoder["max_length"],
                "pooling": encoder["pooling"],
            }
        )
    ).hexdigest()[:20]
    root = ARTIFACTS / "qtrace_mi_v1"
    return root / f"turn_embeddings_{identity}.npz", root / f"turn_embeddings_{identity}.json"


def extract_turn_embeddings(
    corpus: Corpus, config: dict[str, Any]
) -> dict[tuple[int, int], np.ndarray]:
    cache_path, metadata_path = _embedding_cache_paths(config)
    frame = corpus.utterances.sort_values(
        ["transcript_id", "utterance_id"], kind="stable"
    ).reset_index(drop=True)
    if cache_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("dataset_sha256") != sha256_file(SIMPLE_DATA):
            raise ValueError("Cached Q-TRACE embeddings use another dataset")
        with np.load(cache_path) as values:
            matrix = values["embeddings"].astype(np.float32)
            transcript_ids = values["transcript_ids"].astype(int)
            utterance_ids = values["utterance_ids"].astype(int)
        if len(matrix) != len(frame):
            raise ValueError("Cached Q-TRACE embedding row count mismatch")
        return {
            (int(transcript_id), int(utterance_id)): matrix[index]
            for index, (transcript_id, utterance_id) in enumerate(
                zip(transcript_ids, utterance_ids, strict=True)
            )
        }

    device = _require_device()
    encoder_config = config["pretrained_encoder"]
    tokenizer = AutoTokenizer.from_pretrained(
        encoder_config["model_id"],
        revision=encoder_config["revision"],
        cache_dir=ARTIFACTS / "huggingface",
        trust_remote_code=bool(encoder_config["trust_remote_code"]),
        use_fast=True,
    )
    model = AutoModel.from_pretrained(
        encoder_config["model_id"],
        revision=encoder_config["revision"],
        cache_dir=ARTIFACTS / "huggingface",
        trust_remote_code=bool(encoder_config["trust_remote_code"]),
    ).to(device)
    model.eval()
    outputs: list[np.ndarray] = []
    batch_size = int(encoder_config["batch_size"])
    try:
        for start in range(0, len(frame), batch_size):
            texts = frame["utterance_text"].iloc[start : start + batch_size].astype(str).tolist()
            encoded = tokenizer(
                texts,
                max_length=int(encoder_config["max_length"]),
                padding=True,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hidden = model(
                    input_ids=encoded["input_ids"].to(device, non_blocking=True),
                    attention_mask=encoded["attention_mask"].to(device, non_blocking=True),
                ).last_hidden_state
                mask = encoded["attention_mask"].to(device)[..., None]
                pooled = (hidden.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            outputs.append(pooled.cpu().numpy().astype(np.float16))
            if start == 0 or (start // batch_size + 1) % 25 == 0:
                print(
                    f"Q-TRACE embeddings: {min(start + batch_size, len(frame))}/{len(frame)}",
                    flush=True,
                )
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    matrix = np.concatenate(outputs, axis=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            embeddings=matrix,
            transcript_ids=frame["transcript_id"].to_numpy(dtype=np.int64),
            utterance_ids=frame["utterance_id"].to_numpy(dtype=np.int64),
        )
    metadata_path.write_bytes(
        canonical_json_bytes(
            {
                "dataset_sha256": sha256_file(SIMPLE_DATA),
                "rows": len(frame),
                "dimensions": int(matrix.shape[1]),
                "dtype": str(matrix.dtype),
                "encoder": encoder_config,
                "npz_sha256": sha256_file(cache_path),
            }
        )
    )
    return {
        (int(row.transcript_id), int(row.utterance_id)): matrix[index].astype(np.float32)
        for index, row in enumerate(frame.itertuples(index=False))
    }


def estimate_quality_transitions(
    sessions: list[SessionTurns],
    dirichlet_strength: float = 4.0,
) -> np.ndarray:
    counts = np.zeros((2, len(LABELS), len(CLIENT_LABELS), len(LABELS)), dtype=float)
    for session in sessions:
        previous_therapist = -100
        for position, role in enumerate(session.roles):
            if role == 1:
                previous_therapist = int(session.therapist_labels[position])
                continue
            target = int(session.next_action_targets[position])
            client = int(session.client_labels[position])
            if target >= 0 and client >= 0 and previous_therapist >= 0:
                counts[session.quality, previous_therapist, client, target] += 1.0
    global_counts = counts.sum(axis=0) + 1.0
    global_probabilities = global_counts / global_counts.sum(axis=-1, keepdims=True)
    smoothed = counts + float(dirichlet_strength) * global_probabilities[None, ...]
    probabilities = smoothed / smoothed.sum(axis=-1, keepdims=True)
    if not np.allclose(probabilities.sum(axis=-1), 1.0):
        raise AssertionError("Estimated transition probabilities do not sum to one")
    return probabilities


def _inner_partitions(
    outer_train: list[SessionTurns],
    outer_fold: int,
    config: dict[str, Any],
) -> tuple[list[SessionTurns], list[SessionTurns], list[SessionTurns], dict[str, int]]:
    training = config["training"]
    splitter = StratifiedGroupKFold(
        n_splits=int(training["inner_source_folds"]),
        shuffle=True,
        random_state=int(training["inner_split_seed"]) + outer_fold,
    )
    targets = np.asarray([session.quality for session in outer_train])
    groups = np.asarray([session.source_id for session in outer_train])
    assignment: dict[str, int] = {}
    for inner_fold, (_, held_indices) in enumerate(
        splitter.split(np.arange(len(outer_train)), targets, groups=groups)
    ):
        for index in held_indices:
            source = outer_train[int(index)].source_id
            previous = assignment.setdefault(source, inner_fold)
            if previous != inner_fold:
                raise ValueError("An inner source occurs in multiple folds")
    validation_fold = int(training["validation_fold"])
    calibration_fold = int(training["calibration_fold"])
    fit = [
        session
        for session in outer_train
        if assignment[session.source_id] not in {validation_fold, calibration_fold}
    ]
    validation = [
        session for session in outer_train if assignment[session.source_id] == validation_fold
    ]
    calibration = [
        session for session in outer_train if assignment[session.source_id] == calibration_fold
    ]
    partitions = [
        {session.source_id for session in values} for values in (fit, validation, calibration)
    ]
    if (
        partitions[0] & partitions[1]
        or partitions[0] & partitions[2]
        or partitions[1] & partitions[2]
    ):
        raise AssertionError("Internal fit/validation/calibration sources overlap")
    return fit, validation, calibration, assignment


def _class_weights(sessions: list[SessionTurns], device: torch.device) -> dict[str, torch.Tensor]:
    def inverse(values: list[int], classes: int) -> torch.Tensor:
        counts = np.bincount(np.asarray(values, dtype=int), minlength=classes).astype(float)
        if (counts == 0).any():
            raise ValueError("A fitting partition is missing a registered class")
        weights = len(values) / (classes * counts)
        return torch.tensor(weights, dtype=torch.float32, device=device)

    quality = [session.quality for session in sessions]
    therapist = [
        int(value) for session in sessions for value in session.therapist_labels if int(value) >= 0
    ]
    client = [
        int(value) for session in sessions for value in session.client_labels if int(value) >= 0
    ]
    actions = [
        int(value)
        for session in sessions
        for value in session.next_action_targets
        if int(value) >= 0
    ]
    return {
        "quality": inverse(quality, 2),
        "therapist": inverse(therapist, len(LABELS)),
        "client": inverse(client, len(CLIENT_LABELS)),
        "action": inverse(actions, len(LABELS)),
    }


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _loader(
    sessions: list[SessionTurns],
    embeddings: dict[tuple[int, int], np.ndarray],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        _SessionDataset(sessions, embeddings),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=_collate_sessions,
    )


@torch.inference_mode()
def _predict_sessions(
    model: QTraceModel,
    sessions: list[SessionTurns],
    embeddings: dict[tuple[int, int], np.ndarray],
    device: torch.device,
    batch_size: int,
    model_name: str,
    fold: int,
    seed: int,
    fit_texts: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model.eval()
    quality_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    loader = _loader(sessions, embeddings, batch_size, shuffle=False, seed=seed)
    for batch in loader:
        batch_on_device = _move_batch(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(
                batch_on_device["embeddings"],
                batch_on_device["roles"],
                batch_on_device["lengths"],
            )
        quality_probabilities = output["online_quality_probabilities"].float().cpu().numpy()
        action_probabilities = output["action_probabilities"].float().cpu().numpy()
        gates = output["transition_gate"].float().cpu().numpy()
        text_evidence = output["text_quality_evidence"].float().cpu().numpy()
        action_evidence = output["action_quality_evidence"].float().cpu().numpy()
        for row, session in enumerate(batch["sessions"]):
            if model.mode.task_a_loss:
                for checkpoint, position in session.quality_positions.items():
                    probability_low = float(quality_probabilities[row, position, 1])
                    quality_rows.append(
                        {
                            "model": model_name,
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
            if model.mode.task_c_loss:
                positions = np.flatnonzero(session.next_action_targets >= 0)
                for position in positions:
                    probabilities = action_probabilities[row, position]
                    target_position = position + 1
                    record: dict[str, Any] = {
                        "model": model_name,
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
                        "predicted_low_quality_probability": float(
                            quality_probabilities[row, position, 1]
                        ),
                        "transition_gate_high": float(gates[row, position, 0]),
                        "transition_gate_low": float(gates[row, position, 1]),
                    }
                    for index, label in enumerate(LABELS):
                        record[f"prob_{label}"] = float(probabilities[index])
                    action_rows.append(record)
    return pd.DataFrame(quality_rows), pd.DataFrame(action_rows)


def _selection_metrics(
    quality: pd.DataFrame,
    action: pd.DataFrame,
    mode: QTraceMode,
) -> tuple[float, dict[str, Any]]:
    metrics: dict[str, Any] = {}
    scores: list[float] = []
    if mode.task_a_loss:
        metrics["task_a"] = evaluate_quality_predictions(quality)
        primary = metrics["task_a"]["t10"]["source_balanced_balanced_accuracy"]
        scores.append(float(primary))
    if mode.task_c_loss:
        metrics["task_c"] = evaluate_action_predictions(action)
        scores.append(float(metrics["task_c"]["source_balanced_macro_f1"]))
    if not scores:
        raise AssertionError("A Q-TRACE mode has no active task")
    return float(np.mean(scores)), metrics


def _source_weighted_low_prior(sessions: list[SessionTurns]) -> float:
    sources = [session.source_id for session in sessions]
    weights = source_balanced_weights(sources)
    return float(np.average([session.quality == 1 for session in sessions], weights=weights))


def _fit_one(
    fit: list[SessionTurns],
    validation: list[SessionTurns],
    calibration: list[SessionTurns],
    test: list[SessionTurns],
    embeddings: dict[tuple[int, int], np.ndarray],
    config: dict[str, Any],
    mode: QTraceMode,
    fold: int,
    seed: int,
) -> FitResult:
    device = _require_device()
    _seed_everything(seed)
    training = config["training"]
    architecture = config["architecture"]
    batch_size = int(training["batch_size_sessions"])
    train_loader = _loader(fit, embeddings, batch_size, shuffle=True, seed=seed)
    fit_texts = {normalize_text(text) for session in fit for text in session.texts}
    transition = estimate_quality_transitions(
        fit, float(architecture["transition_dirichlet_strength"])
    )
    embedding_size = len(next(iter(embeddings.values())))
    model = QTraceModel(
        embedding_size,
        transition,
        _source_weighted_low_prior(fit),
        architecture,
        mode,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    class_weights = _class_weights(fit, device)
    maximum_epochs = int(training["maximum_epochs"])
    minimum_epochs = int(training["minimum_epochs"])
    patience = int(training["early_stopping_patience"])
    best_score = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] = {}
    stale = 0
    epochs_completed = 0
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for epoch in range(1, maximum_epochs + 1):
            model.train()
            epoch_losses: list[float] = []
            for batch in train_loader:
                batch_on_device = _move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(
                        batch_on_device["embeddings"],
                        batch_on_device["roles"],
                        batch_on_device["lengths"],
                    )
                    loss, _ = qtrace_loss(output, batch_on_device, mode, training, class_weights)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Q-TRACE training produced a non-finite loss")
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["maximum_gradient_norm"])
                )
                optimizer.step()
                epoch_losses.append(float(loss.detach()))
            epochs_completed = epoch
            validation_a, validation_c = _predict_sessions(
                model,
                validation,
                embeddings,
                device,
                batch_size,
                mode.name,
                fold,
                seed,
                fit_texts,
            )
            score, metrics = _selection_metrics(validation_a, validation_c, mode)
            if score > best_score + 1e-8:
                best_score = score
                best_epoch = epoch
                best_metrics = metrics
                best_state = {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                }
                stale = 0
                print(
                    f"{mode.name}/fold={fold}/seed={seed}: epoch={epoch}, "
                    f"selection={score:.4f}, loss={np.mean(epoch_losses):.4f}",
                    flush=True,
                )
            else:
                stale += 1
            if epoch >= minimum_epochs and stale >= patience:
                break
        if best_state is None:
            raise AssertionError("Q-TRACE training never selected an epoch")
        model.load_state_dict(best_state)
        calibration_a, calibration_c = _predict_sessions(
            model,
            calibration,
            embeddings,
            device,
            batch_size,
            mode.name,
            fold,
            seed,
            fit_texts,
        )
        test_a, test_c = _predict_sessions(
            model,
            test,
            embeddings,
            device,
            batch_size,
            mode.name,
            fold,
            seed,
            fit_texts,
        )
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    finally:
        del model
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()
    return FitResult(
        best_epoch=best_epoch,
        best_score=best_score,
        epochs_completed=epochs_completed,
        peak_memory_bytes=peak_memory,
        validation_metrics=best_metrics,
        calibration_a=calibration_a,
        calibration_c=calibration_c,
        test_a=test_a,
        test_c=test_c,
    )


def _cache_directory(config_sha256: str, split_sha256: str) -> Path:
    return (
        ARTIFACTS
        / "qtrace_mi_v1"
        / (f"runs_{git_commit(ROOT)[:12]}_{config_sha256[:12]}_{split_sha256[:12]}")
    )


def _cache_paths(directory: Path, mode: QTraceMode, fold: int, seed: int) -> dict[str, Path]:
    stem = f"{mode.name}_fold{fold}_seed{seed}"
    return {
        "metadata": directory / f"{stem}_metadata.json",
        "calibration_a": directory / f"{stem}_calibration_a.csv",
        "calibration_c": directory / f"{stem}_calibration_c.csv",
        "test_a": directory / f"{stem}_test_a.csv",
        "test_c": directory / f"{stem}_test_c.csv",
    }


def _write_fit_cache(
    directory: Path,
    mode: QTraceMode,
    fold: int,
    seed: int,
    result: FitResult,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    paths = _cache_paths(directory, mode, fold, seed)
    hashes: dict[str, str] = {}
    for name, frame in (
        ("calibration_a", result.calibration_a),
        ("calibration_c", result.calibration_c),
        ("test_a", result.test_a),
        ("test_c", result.test_c),
    ):
        if frame.empty:
            continue
        frame.to_csv(paths[name], index=False, lineterminator="\n", float_format="%.10g")
        hashes[name] = sha256_file(paths[name])
    paths["metadata"].write_bytes(
        canonical_json_bytes(
            {
                "model": mode.name,
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
    mode: QTraceMode,
    fold: int,
    seed: int,
) -> FitResult | None:
    paths = _cache_paths(directory, mode, fold, seed)
    if not paths["metadata"].exists():
        return None
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    if (
        metadata.get("model") != mode.name
        or metadata.get("fold") != fold
        or metadata.get("seed") != seed
    ):
        raise ValueError("Q-TRACE cache identity mismatch")
    frames: dict[str, pd.DataFrame] = {}
    for name in ("calibration_a", "calibration_c", "test_a", "test_c"):
        expected = metadata["ledger_sha256"].get(name)
        if expected is None:
            frames[name] = pd.DataFrame()
            continue
        if not paths[name].exists() or sha256_file(paths[name]) != expected:
            raise ValueError(f"Q-TRACE cached ledger mismatch: {paths[name]}")
        frames[name] = pd.read_csv(paths[name], dtype={"source_id": str})
    return FitResult(
        best_epoch=int(metadata["best_epoch"]),
        best_score=float(metadata["best_score"]),
        epochs_completed=int(metadata["epochs_completed"]),
        peak_memory_bytes=int(metadata["peak_memory_bytes"]),
        validation_metrics=metadata["validation_metrics"],
        calibration_a=frames["calibration_a"],
        calibration_c=frames["calibration_c"],
        test_a=frames["test_a"],
        test_c=frames["test_c"],
    )


def _ensemble_quality(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "model",
        "outer_fold",
        "transcript_id",
        "source_id",
        "checkpoint",
        "last_utterance_id",
        "label",
    ]
    values = ["prob_low", "cumulative_text_evidence", "cumulative_action_evidence"]
    ensemble = frame.groupby(keys, sort=False)[values].mean().reset_index()
    ensemble["raw_prob_low"] = ensemble["prob_low"]
    ensemble["prob_high"] = 1.0 - ensemble["prob_low"]
    ensemble["prediction"] = np.where(ensemble["prob_low"].ge(0.5), "low", "high")
    return ensemble


def _ensemble_action(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "model",
        "outer_fold",
        "transcript_id",
        "decision_utterance_id",
        "target_utterance_id",
        "source_id",
        "label",
        "seen_text_in_outer_train",
    ]
    probability_columns = [f"prob_{label}" for label in LABELS]
    diagnostic_columns = [
        "predicted_low_quality_probability",
        "transition_gate_high",
        "transition_gate_low",
    ]
    ensemble = (
        frame.groupby(keys, sort=False)[probability_columns + diagnostic_columns]
        .mean()
        .reset_index()
    )
    for column in probability_columns:
        ensemble[f"raw_{column}"] = ensemble[column]
    ensemble["prediction"] = np.asarray(LABELS, dtype=object)[
        ensemble[probability_columns].to_numpy().argmax(axis=1)
    ]
    return ensemble


def _calibrate_quality(
    calibration: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    calibrated: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    for (model, fold), test_fold in test.groupby(["model", "outer_fold"], sort=True):
        calibration_fold = calibration[
            calibration["model"].eq(model) & calibration["outer_fold"].eq(fold)
        ]
        temperature = fit_binary_temperature(
            calibration_fold["prob_low"].to_numpy(dtype=float),
            calibration_fold["label"].to_numpy(dtype=object),
            calibration_fold["source_id"].to_numpy(dtype=object),
        )
        values = test_fold.copy()
        values["prob_low"] = apply_binary_temperature(
            values["raw_prob_low"].to_numpy(dtype=float), temperature
        )
        values["prob_high"] = 1.0 - values["prob_low"]
        values["prediction"] = np.where(values["prob_low"].ge(0.5), "low", "high")
        values["temperature"] = temperature
        calibrated.append(values)
        records.append(
            {
                "model": model,
                "outer_fold": int(fold),
                "task": "A",
                "temperature": temperature,
                "calibration_sources": int(calibration_fold["source_id"].nunique()),
                "calibration_rows": len(calibration_fold),
            }
        )
    return pd.concat(calibrated, ignore_index=True), records


def _calibrate_action(
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    alpha: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    probability_columns = [f"prob_{label}" for label in LABELS]
    calibrated: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    for (model, fold), test_fold in test.groupby(["model", "outer_fold"], sort=True):
        calibration_fold = calibration[
            calibration["model"].eq(model) & calibration["outer_fold"].eq(fold)
        ].copy()
        raw_calibration = calibration_fold[probability_columns].to_numpy(dtype=float)
        temperature = fit_multiclass_temperature(
            raw_calibration,
            calibration_fold["label"].to_numpy(dtype=object),
            calibration_fold["source_id"].to_numpy(dtype=object),
        )
        calibration_probabilities = apply_multiclass_temperature(raw_calibration, temperature)
        scores = aps_scores(calibration_probabilities)
        target_indices = np.asarray(
            [LABELS.index(str(label)) for label in calibration_fold["label"]], dtype=int
        )
        risk = source_crc_threshold(
            scores[np.arange(len(scores)), target_indices],
            calibration_fold["source_id"].to_numpy(dtype=object),
            alpha,
        )
        values = test_fold.copy()
        calibrated_probabilities = apply_multiclass_temperature(
            values[[f"raw_prob_{label}" for label in LABELS]].to_numpy(dtype=float),
            temperature,
        )
        for index, label in enumerate(LABELS):
            values[f"prob_{label}"] = calibrated_probabilities[:, index]
        values["prediction"] = np.asarray(LABELS, dtype=object)[
            calibrated_probabilities.argmax(axis=1)
        ]
        values["temperature"] = temperature
        values["prediction_set_threshold"] = float(risk["threshold"])
        values = add_prediction_sets(values, float(risk["threshold"]))
        calibrated.append(values)
        records.append(
            {
                "model": model,
                "outer_fold": int(fold),
                "task": "C",
                "temperature": temperature,
                **risk,
                "calibration_rows": len(calibration_fold),
            }
        )
    return pd.concat(calibrated, ignore_index=True), records


def _source_bootstrap_weights(frame: pd.DataFrame, sampled_sources: np.ndarray) -> np.ndarray:
    multiplicity = Counter(str(value) for value in sampled_sources)
    row_counts = frame["source_id"].astype(str).value_counts()
    return (
        frame["source_id"]
        .astype(str)
        .map(lambda value: multiplicity.get(value, 0) / float(row_counts[value]))
        .to_numpy(dtype=float)
    )


def _bootstrap_inference(
    quality: pd.DataFrame,
    action: pd.DataFrame,
    resamples: int,
    seed: int = 20260904,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    qtrace_a = quality[
        quality["model"].eq("qtrace_mi") & quality["checkpoint"].eq("t10")
    ].sort_values(["source_id", "transcript_id"], kind="stable")
    baseline_a = quality[
        quality["model"].eq("a_only") & quality["checkpoint"].eq("t10")
    ].sort_values(["source_id", "transcript_id"], kind="stable")
    qtrace_c = action[action["model"].eq("qtrace_mi")].sort_values(
        ["source_id", "transcript_id", "target_utterance_id"], kind="stable"
    )
    baseline_c = action[action["model"].eq("c_only")].sort_values(
        ["source_id", "transcript_id", "target_utterance_id"], kind="stable"
    )
    a_keys = ["source_id", "transcript_id", "checkpoint", "label"]
    c_keys = ["source_id", "transcript_id", "target_utterance_id", "label"]
    if (
        not qtrace_a[a_keys]
        .reset_index(drop=True)
        .equals(baseline_a[a_keys].reset_index(drop=True))
    ):
        raise ValueError("Task A candidate and baseline ledgers do not align")
    if (
        not qtrace_c[c_keys]
        .reset_index(drop=True)
        .equals(baseline_c[c_keys].reset_index(drop=True))
    ):
        raise ValueError("Task C candidate and baseline ledgers do not align")
    a_sources = qtrace_a["source_id"].astype(str).unique()
    c_sources = qtrace_c["source_id"].astype(str).unique()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    c_probability_columns = [f"prob_{label}" for label in LABELS]
    c_targets = np.asarray([LABELS.index(str(value)) for value in qtrace_c["label"]])
    c_one_hot = np.eye(len(LABELS))[c_targets]
    for draw in range(resamples):
        sampled_a = rng.choice(a_sources, size=len(a_sources), replace=True)
        sampled_c = rng.choice(c_sources, size=len(c_sources), replace=True)
        a_weights = _source_bootstrap_weights(qtrace_a, sampled_a)
        c_weights = _source_bootstrap_weights(qtrace_c, sampled_c)
        active_a = a_weights > 0
        active_c = c_weights > 0
        candidate_a_prediction = np.where(qtrace_a["prob_low"].to_numpy() >= 0.5, "low", "high")
        baseline_a_prediction = np.where(baseline_a["prob_low"].to_numpy() >= 0.5, "low", "high")
        candidate_a = balanced_accuracy_score(
            qtrace_a.loc[active_a, "label"],
            candidate_a_prediction[active_a],
            sample_weight=a_weights[active_a],
        )
        baseline_a_score = balanced_accuracy_score(
            baseline_a.loc[active_a, "label"],
            baseline_a_prediction[active_a],
            sample_weight=a_weights[active_a],
        )
        candidate_c_probabilities = qtrace_c[c_probability_columns].to_numpy(dtype=float)
        baseline_c_probabilities = baseline_c[c_probability_columns].to_numpy(dtype=float)
        candidate_c_prediction = np.asarray(LABELS)[candidate_c_probabilities.argmax(axis=1)]
        baseline_c_prediction = np.asarray(LABELS)[baseline_c_probabilities.argmax(axis=1)]
        candidate_c = f1_score(
            qtrace_c.loc[active_c, "label"],
            candidate_c_prediction[active_c],
            labels=list(LABELS),
            average="macro",
            sample_weight=c_weights[active_c],
            zero_division=0,
        )
        baseline_c_score = f1_score(
            baseline_c.loc[active_c, "label"],
            baseline_c_prediction[active_c],
            labels=list(LABELS),
            average="macro",
            sample_weight=c_weights[active_c],
            zero_division=0,
        )
        candidate_brier = np.average(
            np.square(candidate_c_probabilities - c_one_hot).sum(axis=1)[active_c],
            weights=c_weights[active_c],
        )
        baseline_brier = np.average(
            np.square(baseline_c_probabilities - c_one_hot).sum(axis=1)[active_c],
            weights=c_weights[active_c],
        )
        rows.append(
            {
                "draw": draw,
                "task_a_t10_balanced_accuracy_delta": candidate_a - baseline_a_score,
                "task_c_macro_f1_delta": candidate_c - baseline_c_score,
                "task_c_brier_delta": candidate_brier - baseline_brier,
            }
        )
    draws = pd.DataFrame(rows)

    def interval(column: str) -> dict[str, float]:
        return {
            "mean": float(draws[column].mean()),
            "low": float(draws[column].quantile(0.025)),
            "high": float(draws[column].quantile(0.975)),
        }

    return draws, {
        "resamples": resamples,
        "seed": seed,
        "cluster_unit": "source_id",
        "intervals": {
            column: interval(column)
            for column in (
                "task_a_t10_balanced_accuracy_delta",
                "task_c_macro_f1_delta",
                "task_c_brier_delta",
            )
        },
    }


def _seed_deltas(task_a: pd.DataFrame, task_c: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {"task_a": {}, "task_c": {}}
    for seed in sorted(set(task_a["seed"]) | set(task_c["seed"])):
        if seed in set(task_a["seed"]):
            candidate = task_a[
                task_a["model"].eq("qtrace_mi")
                & task_a["seed"].eq(seed)
                & task_a["checkpoint"].eq("t10")
            ]
            baseline = task_a[
                task_a["model"].eq("a_only")
                & task_a["seed"].eq(seed)
                & task_a["checkpoint"].eq("t10")
            ]
            candidate_metric = evaluate_quality_predictions(candidate)["t10"][
                "source_balanced_balanced_accuracy"
            ]
            baseline_metric = evaluate_quality_predictions(baseline)["t10"][
                "source_balanced_balanced_accuracy"
            ]
            result["task_a"][str(seed)] = candidate_metric - baseline_metric
        if seed in set(task_c["seed"]):
            candidate_c = task_c[task_c["model"].eq("qtrace_mi") & task_c["seed"].eq(seed)]
            baseline_c = task_c[task_c["model"].eq("c_only") & task_c["seed"].eq(seed)]
            candidate_metric_c = evaluate_action_predictions(candidate_c)[
                "source_balanced_macro_f1"
            ]
            baseline_metric_c = evaluate_action_predictions(baseline_c)["source_balanced_macro_f1"]
            result["task_c"][str(seed)] = candidate_metric_c - baseline_metric_c
    result["task_a_positive_seed_count"] = sum(value > 0 for value in result["task_a"].values())
    result["task_c_positive_seed_count"] = sum(value > 0 for value in result["task_c"].values())
    return result


def _csv_payload(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, lineterminator="\n", float_format="%.10g")
    return buffer.getvalue().encode("utf-8")


def run_qtrace(
    corpus: Corpus,
    split_manifest: dict[str, Any],
    output_dir: Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = output_dir or RESEARCH_RESULTS / "ac_v1" / "qtrace_mi"
    if (output_dir / "summary.json").exists():
        validate_qtrace_evidence(output_dir)
        return read_json(output_dir / "summary.json")
    protocol, config = _require_registered_state(split_manifest)
    sessions = build_session_turns(corpus, tuple(protocol["task_a"]["therapist_turn_budgets"]))
    if sum(int((session.next_action_targets >= 0).sum()) for session in sessions) != int(
        protocol["task_c"]["expected_decisions"]
    ):
        raise ValueError("Task C decision count differs from its registration")
    embeddings = extract_turn_embeddings(corpus, config)
    lookup = fold_lookup(split_manifest)
    config_sha256 = sha256_file(QTRACE_CONFIG)
    cache_dir = _cache_directory(config_sha256, split_manifest["manifest_sha256"])
    all_a: list[pd.DataFrame] = []
    all_c: list[pd.DataFrame] = []
    all_calibration_a: list[pd.DataFrame] = []
    all_calibration_c: list[pd.DataFrame] = []
    selections: list[dict[str, Any]] = []
    partitions: list[dict[str, Any]] = []
    modes = [mode_from_config(config, value["model"]) for value in config["models"]]
    for fold in range(len(split_manifest["folds"])):
        outer_train = [session for session in sessions if lookup[session.source_id] != fold]
        test = [session for session in sessions if lookup[session.source_id] == fold]
        fit, validation, calibration, assignment = _inner_partitions(outer_train, fold, config)
        partitions.append(
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
        print(
            f"Q-TRACE outer fold {fold}: fit={len(fit)}, validation={len(validation)}, "
            f"calibration={len(calibration)}, test={len(test)} sessions",
            flush=True,
        )
        for mode in modes:
            for seed in config["training"]["seeds"]:
                seed = int(seed)
                result = _load_fit_cache(cache_dir, mode, fold, seed)
                if result is None:
                    result = _fit_one(
                        fit,
                        validation,
                        calibration,
                        test,
                        embeddings,
                        config,
                        mode,
                        fold,
                        seed,
                    )
                    _write_fit_cache(cache_dir, mode, fold, seed, result)
                else:
                    print(f"Q-TRACE cache hit: {mode.name}/fold={fold}/seed={seed}", flush=True)
                selections.append(
                    {
                        "model": mode.name,
                        "outer_fold": fold,
                        "seed": seed,
                        "best_epoch": result.best_epoch,
                        "best_validation_score": result.best_score,
                        "epochs_completed": result.epochs_completed,
                        "peak_memory_bytes": result.peak_memory_bytes,
                        "validation_metrics": result.validation_metrics,
                    }
                )
                if not result.test_a.empty:
                    all_a.append(result.test_a)
                    all_calibration_a.append(result.calibration_a)
                if not result.test_c.empty:
                    all_c.append(result.test_c)
                    all_calibration_c.append(result.calibration_c)

    task_a_by_seed = pd.concat(all_a, ignore_index=True)
    task_c_by_seed = pd.concat(all_c, ignore_index=True)
    calibration_a = _ensemble_quality(pd.concat(all_calibration_a, ignore_index=True))
    calibration_c = _ensemble_action(pd.concat(all_calibration_c, ignore_index=True))
    task_a_ensemble = _ensemble_quality(task_a_by_seed)
    task_c_ensemble = _ensemble_action(task_c_by_seed)
    task_a_ensemble, calibration_a_records = _calibrate_quality(calibration_a, task_a_ensemble)
    task_c_ensemble, calibration_c_records = _calibrate_action(
        calibration_c,
        task_c_ensemble,
        float(config["calibration"]["prediction_set_alpha"]),
    )
    task_a_metrics = {
        model: evaluate_quality_predictions(frame.reset_index(drop=True))
        for model, frame in task_a_ensemble.groupby("model", sort=True)
    }
    task_c_metrics = {
        model: evaluate_action_predictions(frame.reset_index(drop=True))
        for model, frame in task_c_ensemble.groupby("model", sort=True)
    }
    prediction_set_metrics = {
        model: evaluate_prediction_sets(frame.reset_index(drop=True))
        for model, frame in task_c_ensemble.groupby("model", sort=True)
    }
    seed_deltas = _seed_deltas(task_a_by_seed, task_c_by_seed)
    bootstrap_draws, inference = _bootstrap_inference(
        task_a_ensemble,
        task_c_ensemble,
        int(protocol["inference"]["bootstrap_resamples"]),
    )
    gate_config = protocol["candidate_success_gate"]
    a_interval = inference["intervals"]["task_a_t10_balanced_accuracy_delta"]
    c_interval = inference["intervals"]["task_c_macro_f1_delta"]
    qtrace_c_metric = task_c_metrics["qtrace_mi"]
    baseline_c_metric = task_c_metrics["c_only"]
    c_point_delta = (
        qtrace_c_metric["source_balanced_macro_f1"] - baseline_c_metric["source_balanced_macro_f1"]
    )
    c_brier_delta = (
        qtrace_c_metric["source_balanced_brier"] - baseline_c_metric["source_balanced_brier"]
    )
    class_collapse = set(
        task_c_ensemble.loc[task_c_ensemble["model"].eq("qtrace_mi"), "prediction"]
    ) != set(LABELS)
    gate = {
        "task_a_positive_interval": bool(a_interval["low"] > 0),
        "task_c_minimum_delta": bool(
            c_point_delta >= float(gate_config["task_c_minimum_macro_f1_delta_vs_c_only"])
        ),
        "task_c_positive_interval": bool(c_interval["low"] > 0),
        "task_c_brier_within_limit": bool(
            c_brier_delta <= float(gate_config["maximum_task_c_brier_degradation"])
        ),
        "task_a_positive_seed_count": seed_deltas["task_a_positive_seed_count"],
        "task_c_positive_seed_count": seed_deltas["task_c_positive_seed_count"],
        "minimum_positive_seed_count": int(gate_config["minimum_positive_seed_count"]),
        "class_collapse": class_collapse,
    }
    gate["pass"] = bool(
        gate["task_a_positive_interval"]
        and gate["task_c_minimum_delta"]
        and gate["task_c_positive_interval"]
        and gate["task_c_brier_within_limit"]
        and gate["task_a_positive_seed_count"] >= gate["minimum_positive_seed_count"]
        and gate["task_c_positive_seed_count"] >= gate["minimum_positive_seed_count"]
        and not class_collapse
    )

    payloads = {
        "task_a_predictions_by_seed.csv": _csv_payload(task_a_by_seed),
        "task_c_predictions_by_seed.csv": _csv_payload(task_c_by_seed),
        "task_a_predictions_seed_ensemble.csv": _csv_payload(task_a_ensemble),
        "task_c_predictions_seed_ensemble.csv": _csv_payload(task_c_ensemble),
        "bootstrap_draws.csv": _csv_payload(bootstrap_draws),
        "selection.json": canonical_json_bytes({"selection": selections}),
        "partitions.json": canonical_json_bytes({"partitions": partitions}),
        "calibration.json": canonical_json_bytes(
            {"task_a": calibration_a_records, "task_c": calibration_c_records}
        ),
    }
    hashes = {
        name: write_create_only(output_dir / name, payload) for name, payload in payloads.items()
    }
    summary = {
        "result_id": "annomi-qtrace-mi-source-cv-v1",
        "protocol_id": protocol["protocol_id"],
        "config_id": config["config_id"],
        "code_commit": git_commit(ROOT),
        "config_sha256": config_sha256,
        "split_manifest_sha256": split_manifest["manifest_sha256"],
        "dataset_sha256": {
            "simple": sha256_file(SIMPLE_DATA),
            "full": sha256_file(FULL_DATA),
        },
        "task_a_metrics": task_a_metrics,
        "task_c_metrics": task_c_metrics,
        "prediction_set_metrics": prediction_set_metrics,
        "seed_deltas_qtrace_minus_single_task": seed_deltas,
        "paired_source_bootstrap": inference,
        "candidate_success_gate": gate,
        "evidence_sha256": hashes,
        "runtime_environment": _runtime_environment(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_create_only(output_dir / "summary.json", canonical_json_bytes(summary))
    validate_qtrace_evidence(output_dir)
    return summary


def run_qtrace_smoke(corpus: Corpus, split_manifest: dict[str, Any]) -> dict[str, Any]:
    protocol, config = _require_registered_state(split_manifest)
    output_path = RESEARCH_RESULTS / "gate1" / "qtrace_mi_smoke_v1.json"
    if output_path.exists():
        return read_json(output_path)
    device = _require_device()
    sessions = build_session_turns(corpus, tuple(protocol["task_a"]["therapist_turn_budgets"]))
    lookup = fold_lookup(split_manifest)
    outer_train = [session for session in sessions if lookup[session.source_id] != 0]
    fit, _, _, _ = _inner_partitions(outer_train, 0, config)
    embeddings = extract_turn_embeddings(corpus, config)
    mode = mode_from_config(config, "qtrace_mi")
    architecture = config["architecture"]
    transition = estimate_quality_transitions(
        fit, float(architecture["transition_dirichlet_strength"])
    )
    model = QTraceModel(
        len(next(iter(embeddings.values()))),
        transition,
        _source_weighted_low_prior(fit),
        architecture,
        mode,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    loader = _loader(
        fit,
        embeddings,
        int(config["training"]["batch_size_sessions"]),
        shuffle=True,
        seed=1907,
    )
    batch = _move_batch(next(iter(loader)), device)
    class_weights = _class_weights(fit, device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(batch["embeddings"], batch["roles"], batch["lengths"])
        loss, components = qtrace_loss(output, batch, mode, config["training"], class_weights)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["maximum_gradient_norm"]))
    optimizer.step()
    action_sums = output["action_probabilities"].float().sum(dim=-1)
    quality_sums = output["online_quality_probabilities"].float().sum(dim=-1)
    payload = {
        "gate_id": "annomi-qtrace-mi-cuda-smoke-v1",
        "status": "pass",
        "engineering_gate_not_performance_result": True,
        "outer_test_labels_used": False,
        "frozen_embedding_extraction_all_rows": True,
        "code_commit": git_commit(ROOT),
        "config_sha256": sha256_file(QTRACE_CONFIG),
        "split_manifest_sha256": split_manifest["manifest_sha256"],
        "fit_sessions_available": len(fit),
        "batch_sessions": len(batch["sessions"]),
        "batch_turns": int(batch["lengths"].sum()),
        "loss": float(loss.detach()),
        "loss_components": components,
        "probabilities_finite": bool(
            torch.isfinite(output["action_probabilities"]).all()
            and torch.isfinite(output["online_quality_probabilities"]).all()
        ),
        "maximum_probability_sum_error": float(
            max((action_sums - 1).abs().max(), (quality_sums - 1).abs().max())
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


def _assert_close(actual: float, expected: float, name: str) -> None:
    if not np.isclose(actual, expected, atol=1e-8):
        raise ValueError(f"Q-TRACE metric reconstruction mismatch: {name}")


def validate_qtrace_evidence(output_dir: Path) -> None:
    summary = read_json(output_dir / "summary.json")
    for name, expected_hash in summary["evidence_sha256"].items():
        path = output_dir / name
        if not path.exists() or sha256_file(path) != expected_hash:
            raise ValueError(f"Q-TRACE evidence hash mismatch: {name}")
    task_a_seed = pd.read_csv(
        output_dir / "task_a_predictions_by_seed.csv", dtype={"source_id": str}
    )
    task_c_seed = pd.read_csv(
        output_dir / "task_c_predictions_by_seed.csv", dtype={"source_id": str}
    )
    task_a = pd.read_csv(
        output_dir / "task_a_predictions_seed_ensemble.csv", dtype={"source_id": str}
    )
    task_c = pd.read_csv(
        output_dir / "task_c_predictions_seed_ensemble.csv", dtype={"source_id": str}
    )
    if task_a_seed.duplicated(["model", "seed", "transcript_id", "checkpoint"]).any():
        raise ValueError("Duplicate Q-TRACE Task A per-seed prediction")
    if task_c_seed.duplicated(["model", "seed", "transcript_id", "target_utterance_id"]).any():
        raise ValueError("Duplicate Q-TRACE Task C per-seed prediction")
    if task_a.duplicated(["model", "transcript_id", "checkpoint"]).any():
        raise ValueError("Duplicate Q-TRACE Task A ensemble prediction")
    if task_c.duplicated(["model", "transcript_id", "target_utterance_id"]).any():
        raise ValueError("Duplicate Q-TRACE Task C ensemble prediction")
    action_values = task_c[[f"prob_{label}" for label in LABELS]].to_numpy(dtype=float)
    if not np.isfinite(action_values).all() or not np.allclose(
        action_values.sum(axis=1), 1.0, atol=1e-6
    ):
        raise ValueError("Invalid Q-TRACE Task C probabilities")
    for model, frame in task_a.groupby("model", sort=True):
        rebuilt = evaluate_quality_predictions(frame.reset_index(drop=True))
        recorded = summary["task_a_metrics"][model]
        for checkpoint, metrics in rebuilt.items():
            for metric in (
                "source_balanced_balanced_accuracy",
                "source_balanced_macro_f1",
                "source_balanced_brier",
                "source_balanced_log_loss",
            ):
                _assert_close(
                    metrics[metric],
                    recorded[checkpoint][metric],
                    f"A/{model}/{checkpoint}/{metric}",
                )
    for model, frame in task_c.groupby("model", sort=True):
        rebuilt = evaluate_action_predictions(frame.reset_index(drop=True))
        recorded = summary["task_c_metrics"][model]
        for metric in (
            "source_balanced_macro_f1",
            "source_balanced_brier",
            "source_balanced_log_loss",
        ):
            _assert_close(rebuilt[metric], recorded[metric], f"C/{model}/{metric}")
        set_metrics = evaluate_prediction_sets(frame.reset_index(drop=True))
        recorded_sets = summary["prediction_set_metrics"][model]
        for metric in (
            "source_balanced_coverage",
            "source_balanced_mean_set_size",
            "source_balanced_singleton_rate",
        ):
            _assert_close(set_metrics[metric], recorded_sets[metric], f"sets/{model}/{metric}")
