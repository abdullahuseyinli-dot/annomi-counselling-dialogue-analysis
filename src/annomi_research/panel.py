from __future__ import annotations

import copy
import hashlib
import io
import itertools
import math
import os
import platform
import random
import subprocess
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from .constants import (
    ARTIFACTS,
    FULL_DATA,
    FULL_MANIFEST,
    PANEL_CONFIG,
    PROTOCOL,
    RESEARCH_RESULTS,
    ROOT,
    SIMPLE_DATA,
    SIMPLE_MANIFEST,
)
from .data import Corpus, MultiAnnotatorTask, build_multiannotator_task
from .io import canonical_json_bytes, git_commit, read_json, sha256_file, write_create_only
from .metrics import source_balanced_weights

DEFAULT_PANEL_OUTPUT = RESEARCH_RESULTS / "multiannotator_v1" / "panel_mi"


@dataclass(frozen=True)
class LinearRecipe:
    recipe_id: str
    inverse_l2: float


@dataclass(frozen=True)
class PanelRecipe:
    recipe_id: str
    rank: int
    annotator_shrinkage: float


@dataclass(frozen=True)
class FeatureTransform:
    pca: PCA
    scaler: StandardScaler

    def transform(self, values: np.ndarray) -> np.ndarray:
        return self.scaler.transform(self.pca.transform(values)).astype(np.float32)


@dataclass(frozen=True)
class PanelFitOutcome:
    probabilities: np.ndarray
    head_disagreement: np.ndarray
    best_score: float | None
    best_epoch: int
    epochs_completed: int
    base_linear_solver: str


class PanelMIHead(nn.Module):
    """Shared classifier with centered low-rank annotator-specific deviations."""

    def __init__(
        self,
        input_size: int,
        n_classes: int,
        n_annotators: int,
        rank: int,
    ) -> None:
        super().__init__()
        self.base = nn.Linear(input_size, n_classes)
        self.annotator_bias_raw = nn.Parameter(torch.zeros(n_annotators, n_classes))
        self.item_projection = nn.Linear(input_size, rank, bias=False)
        self.annotator_factor_raw = nn.Parameter(torch.empty(n_annotators, rank))
        self.class_factor = nn.Parameter(torch.empty(rank, n_classes))
        nn.init.normal_(self.item_projection.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.annotator_factor_raw, mean=0.0, std=0.02)
        nn.init.normal_(self.class_factor, mean=0.0, std=0.02)

    def centered_deviations(self) -> tuple[torch.Tensor, torch.Tensor]:
        bias = self.annotator_bias_raw - self.annotator_bias_raw.mean(dim=0, keepdim=True)
        factors = self.annotator_factor_raw - self.annotator_factor_raw.mean(dim=0, keepdim=True)
        return bias, factors

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        bias, annotator_factors = self.centered_deviations()
        item_factors = self.item_projection(features)
        interaction = torch.einsum(
            "nr,ar,rc->nac", item_factors, annotator_factors, self.class_factor
        )
        return self.base(features)[:, None, :] + bias[None, :, :] + interaction


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
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": __import__("sklearn").__version__,
        "scipy": __import__("scipy").__version__,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _require_registered_state() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = read_json(PROTOCOL)
    config = read_json(PANEL_CONFIG)
    if config["status"] != "registered_before_any_multiannotator_model_evaluation":
        raise ValueError("PANEL-MI configuration is not prospectively registered")
    if config["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("PANEL-MI configuration and research protocol disagree")
    if config["final_seeds"] != protocol["development"]["seeds"]:
        raise ValueError("PANEL-MI and protocol seed lists disagree")
    for data_path, manifest_path in (
        (SIMPLE_DATA, SIMPLE_MANIFEST),
        (FULL_DATA, FULL_MANIFEST),
    ):
        if sha256_file(data_path) != read_json(manifest_path)["sha256"]:
            raise ValueError(f"Dataset hash mismatch: {data_path}")
    if not _git_is_clean():
        raise RuntimeError("Commit tracked code/config changes before running PANEL-MI evidence")
    return protocol, config


def _tasks(corpus: Corpus, config: dict[str, Any]) -> dict[str, MultiAnnotatorTask]:
    tasks: dict[str, MultiAnnotatorTask] = {}
    for task, specification in config["tasks"].items():
        tasks[task] = build_multiannotator_task(
            corpus,
            task=task,
            label_column=str(specification["label_column"]),
            labels=tuple(specification["labels"]),
            expected_annotations_per_item=int(config["study"]["expected_annotations_per_item"]),
            expected_items=int(specification["expected_items"]),
        )
    observed_transcripts = sorted(
        {
            int(value)
            for task_data in tasks.values()
            for value in task_data.items["transcript_id"].unique()
        }
    )
    if observed_transcripts != config["study"]["outer_transcripts"]:
        raise ValueError("Observed multi-annotator transcripts differ from registration")
    if sum(len(task_data.items) for task_data in tasks.values()) != int(
        config["study"]["expected_items"]
    ):
        raise ValueError("Observed multi-annotator item count differs from registration")
    if any(
        len(task_data.annotator_ids) != int(config["study"]["expected_anonymous_annotators"])
        for task_data in tasks.values()
    ):
        raise ValueError("Observed annotation-panel size differs from registration")
    return tasks


def _vote_probabilities(task_data: MultiAnnotatorTask) -> np.ndarray:
    return task_data.items[[f"vote_prob_{label}" for label in task_data.labels]].to_numpy(
        dtype=float
    )


def _cohort_fingerprint(tasks: dict[str, MultiAnnotatorTask], config: dict[str, Any]) -> str:
    records: list[dict[str, Any]] = []
    for task, task_data in tasks.items():
        for row in task_data.items.itertuples(index=False):
            records.append(
                {
                    "task": task,
                    "transcript_id": int(row.transcript_id),
                    "utterance_id": int(row.utterance_id),
                    "role_prefixed_text": str(row.role_prefixed_text),
                }
            )
    payload = {
        "encoder": config["pretrained_encoder"],
        "records": records,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


@torch.inference_mode()
def _encode_texts(texts: list[str], config: dict[str, Any]) -> tuple[np.ndarray, int]:
    encoder_spec = config["pretrained_encoder"]
    cache_dir = ARTIFACTS / "huggingface"
    tokenizer = AutoTokenizer.from_pretrained(
        encoder_spec["model_id"],
        revision=encoder_spec["revision"],
        cache_dir=cache_dir,
        trust_remote_code=bool(encoder_spec["trust_remote_code"]),
        use_fast=True,
    )
    model = AutoModel.from_pretrained(
        encoder_spec["model_id"],
        revision=encoder_spec["revision"],
        cache_dir=cache_dir,
        trust_remote_code=bool(encoder_spec["trust_remote_code"]),
    )
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    outputs: list[np.ndarray] = []
    batch_size = 32 if device.type == "cuda" else 8
    maximum_length = int(encoder_spec["maximum_length"])
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=maximum_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with context:
            hidden = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        outputs.append(pooled.float().cpu().numpy())

    embeddings = np.concatenate(outputs, axis=0).astype(np.float32)
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not np.isfinite(embeddings).all():
        raise FloatingPointError("Frozen encoder produced non-finite embeddings")
    return embeddings, peak_memory


def _task_embeddings(
    tasks: dict[str, MultiAnnotatorTask],
    config: dict[str, Any],
    commit: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    fingerprint = _cohort_fingerprint(tasks, config)
    cache_dir = ARTIFACTS / "research" / "panel_mi_v1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"embeddings_{commit[:12]}_{fingerprint[:16]}.npz"
    task_names = list(tasks)
    row_counts = [len(tasks[task].items) for task in task_names]
    peak_memory = 0
    cache_hit = cache_path.exists()
    if cache_hit:
        with np.load(cache_path, allow_pickle=False) as cached:
            if str(cached["fingerprint"].item()) != fingerprint:
                raise ValueError("PANEL-MI embedding-cache fingerprint mismatch")
            embeddings = cached["embeddings"].astype(np.float32)
            cached_counts = cached["row_counts"].astype(int).tolist()
        if cached_counts != row_counts:
            raise ValueError("PANEL-MI embedding-cache row counts mismatch")
    else:
        texts = [
            text
            for task in task_names
            for text in tasks[task].items["role_prefixed_text"].astype(str).tolist()
        ]
        embeddings, peak_memory = _encode_texts(texts, config)
        np.savez_compressed(
            cache_path,
            embeddings=embeddings,
            fingerprint=np.asarray(fingerprint),
            row_counts=np.asarray(row_counts, dtype=np.int64),
        )
    if len(embeddings) != sum(row_counts):
        raise ValueError("PANEL-MI embedding cache has an unexpected length")

    by_task: dict[str, np.ndarray] = {}
    start = 0
    for task, count in zip(task_names, row_counts, strict=True):
        by_task[task] = embeddings[start : start + count]
        start += count
    return by_task, {
        "cohort_fingerprint": fingerprint,
        "cache_filename": cache_path.name,
        "cache_hit": cache_hit,
        "rows": len(embeddings),
        "dimensions": int(embeddings.shape[1]),
        "peak_encoder_memory_bytes": peak_memory,
    }


def _fit_feature_transform(
    train_embeddings: np.ndarray,
    config: dict[str, Any],
) -> tuple[FeatureTransform, np.ndarray]:
    requested = int(config["preprocessing"]["pca_components"])
    components = min(requested, len(train_embeddings) - 1, train_embeddings.shape[1])
    if components < 1:
        raise ValueError("Not enough training items to fit registered PCA")
    pca = PCA(
        n_components=components,
        whiten=bool(config["preprocessing"]["pca_whiten"]),
        svd_solver="full",
    )
    projected = pca.fit_transform(train_embeddings)
    scaler = StandardScaler().fit(projected)
    transform = FeatureTransform(pca=pca, scaler=scaler)
    return transform, scaler.transform(projected).astype(np.float32)


def _item_weights(transcript_ids: pd.Series) -> np.ndarray:
    return source_balanced_weights(transcript_ids.astype(str))


def _fit_linear_model(
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    inverse_l2: float,
    maximum_iterations: int,
    tolerance: float = 1e-6,
    fallback_solver: str = "newton-cholesky",
) -> LogisticRegression:
    n_items, n_classes = targets.shape
    expanded_features = np.repeat(features, n_classes, axis=0)
    expanded_labels = np.tile(np.arange(n_classes, dtype=int), n_items)
    expanded_weights = (weights[:, None] * targets).reshape(-1)
    keep = expanded_weights > 0
    convergence_failures: list[str] = []
    for solver in ("lbfgs", fallback_solver):
        model = LogisticRegression(
            C=float(inverse_l2),
            solver=solver,
            max_iter=int(maximum_iterations),
            tol=float(tolerance),
            fit_intercept=True,
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(
                    expanded_features[keep],
                    expanded_labels[keep],
                    sample_weight=expanded_weights[keep],
                )
        except ConvergenceWarning as error:
            convergence_failures.append(f"{solver}: {error}")
            continue
        if not np.array_equal(model.classes_, np.arange(n_classes)):
            raise ValueError("Linear label-distribution model did not fit every class")
        model.annomi_solver_used_ = solver
        return model
    raise RuntimeError(
        "All registered linear solvers failed to converge: " + " | ".join(convergence_failures)
    )


def _hard_targets(votes: np.ndarray) -> np.ndarray:
    targets = np.zeros_like(votes)
    targets[np.arange(len(votes)), votes.argmax(axis=1)] = 1.0
    return targets


def _normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return -(probabilities * np.log(clipped)).sum(axis=1) / math.log(probabilities.shape[1])


def _row_distribution_losses(
    votes: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    log_score = -(votes * np.log(probabilities)).sum(axis=1)
    brier = np.square(votes - probabilities).sum(axis=1)
    midpoint = 0.5 * (votes + probabilities)
    vote_log = np.zeros_like(votes)
    np.log(votes / midpoint, out=vote_log, where=votes > 0)
    pred_log = np.log(probabilities / midpoint)
    jsd = 0.5 * (votes * vote_log).sum(axis=1) + 0.5 * (probabilities * pred_log).sum(axis=1)
    return log_score, brier, jsd


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.allclose(left, left[0]) or np.allclose(right, right[0]):
        return None
    statistic = float(spearmanr(left, right).statistic)
    return statistic if np.isfinite(statistic) else None


def evaluate_vote_predictions(
    items: pd.DataFrame,
    probabilities: np.ndarray,
    labels: tuple[str, ...],
    head_disagreement: np.ndarray | None = None,
) -> dict[str, Any]:
    votes = items[[f"vote_prob_{label}" for label in labels]].to_numpy(dtype=float)
    if probabilities.shape != votes.shape:
        raise ValueError("Vote and prediction matrices have different shapes")
    if not np.isfinite(probabilities).all():
        raise FloatingPointError("Vote prediction contains a non-finite value")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Vote prediction probabilities do not sum to one")
    weights = _item_weights(items["transcript_id"])
    log_score, brier, jsd = _row_distribution_losses(votes, probabilities)
    predicted_entropy = _normalized_entropy(probabilities)
    vote_entropy = items["vote_entropy"].to_numpy(dtype=float)
    prediction = np.asarray(labels, dtype=object)[probabilities.argmax(axis=1)]
    plurality = items["plurality_label"].to_numpy(dtype=object)

    per_transcript: dict[str, Any] = {}
    for transcript_id, indices in items.groupby("transcript_id", sort=True).indices.items():
        index = np.asarray(indices, dtype=int)
        per_transcript[str(int(transcript_id))] = {
            "n_items": len(index),
            "vote_log_score": float(log_score[index].mean()),
            "vote_brier": float(brier[index].mean()),
            "jensen_shannon_divergence": float(jsd[index].mean()),
            "vote_entropy_mean_absolute_error": float(
                np.abs(predicted_entropy[index] - vote_entropy[index]).mean()
            ),
        }

    result: dict[str, Any] = {
        "n_items": len(items),
        "n_transcripts": int(items["transcript_id"].nunique()),
        "transcript_balanced_vote_log_score": float(np.average(log_score, weights=weights)),
        "transcript_balanced_vote_brier": float(np.average(brier, weights=weights)),
        "transcript_balanced_jensen_shannon_divergence": float(np.average(jsd, weights=weights)),
        "transcript_balanced_plurality_macro_f1": float(
            f1_score(
                plurality,
                prediction,
                labels=list(labels),
                average="macro",
                sample_weight=weights,
                zero_division=0,
            )
        ),
        "vote_entropy_prediction_spearman": _safe_spearman(vote_entropy, predicted_entropy),
        "vote_entropy_mean_absolute_error": float(
            np.average(np.abs(predicted_entropy - vote_entropy), weights=weights)
        ),
        "per_transcript": per_transcript,
    }
    if head_disagreement is not None:
        if len(head_disagreement) != len(items):
            raise ValueError("Panel-head diagnostic length mismatch")
        result["mean_annotator_head_disagreement"] = float(
            np.average(head_disagreement, weights=weights)
        )
        result["annotator_head_disagreement_to_vote_entropy_spearman"] = _safe_spearman(
            head_disagreement, vote_entropy
        )
    return result


def _weighted_log_score(
    votes: np.ndarray,
    probabilities: np.ndarray,
    transcript_ids: pd.Series,
) -> float:
    log_score, _, _ = _row_distribution_losses(votes, probabilities)
    return float(np.average(log_score, weights=_item_weights(transcript_ids)))


def _seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _panel_probabilities(
    model: PanelMIHead,
    features: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.inference_mode():
        head_probabilities = torch.softmax(model(features), dim=-1)
        aggregate = head_probabilities.mean(dim=1)
        midpoint = 0.5 * (head_probabilities + aggregate[:, None, :])
        head_jsd = 0.5 * (
            head_probabilities
            * (
                torch.log(head_probabilities.clamp_min(1e-12))
                - torch.log(midpoint.clamp_min(1e-12))
            )
        ).sum(dim=-1)
        head_jsd += 0.5 * (
            aggregate[:, None, :]
            * (
                torch.log(aggregate[:, None, :].clamp_min(1e-12))
                - torch.log(midpoint.clamp_min(1e-12))
            )
        ).sum(dim=-1)
    probabilities = aggregate.cpu().numpy()
    disagreement = head_jsd.mean(dim=1).cpu().numpy()
    return probabilities, disagreement


def _fit_panel(
    train_features: np.ndarray,
    train_votes: np.ndarray,
    train_annotation_labels: np.ndarray,
    train_transcript_ids: pd.Series,
    evaluation_features: np.ndarray,
    evaluation_votes: np.ndarray | None,
    evaluation_transcript_ids: pd.Series | None,
    recipe: PanelRecipe,
    config: dict[str, Any],
    seed: int,
    fixed_epochs: int | None = None,
) -> PanelFitOutcome:
    _seed_everything(seed)
    settings = config["training"]
    n_classes = train_votes.shape[1]
    n_annotators = train_annotation_labels.shape[1]
    base_model = _fit_linear_model(
        train_features,
        train_votes,
        _item_weights(train_transcript_ids),
        inverse_l2=1.0,
        maximum_iterations=int(settings["linear_max_iterations"]),
        tolerance=float(settings["linear_tolerance"]),
        fallback_solver="newton-cholesky",
    )
    model = PanelMIHead(
        input_size=train_features.shape[1],
        n_classes=n_classes,
        n_annotators=n_annotators,
        rank=recipe.rank,
    )
    with torch.no_grad():
        model.base.weight.copy_(torch.tensor(base_model.coef_, dtype=torch.float32))
        model.base.bias.copy_(torch.tensor(base_model.intercept_, dtype=torch.float32))

    deviation_parameters = [
        model.annotator_bias_raw,
        model.item_projection.weight,
        model.annotator_factor_raw,
        model.class_factor,
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.base.parameters()),
                "weight_decay": float(settings["panel_base_weight_decay"]),
            },
            {"params": deviation_parameters, "weight_decay": 0.0},
        ],
        lr=float(settings["panel_learning_rate"]),
    )
    x_train = torch.tensor(train_features, dtype=torch.float32)
    q_train = torch.tensor(train_votes, dtype=torch.float32)
    y_train = torch.tensor(train_annotation_labels, dtype=torch.long)
    train_weights = torch.tensor(_item_weights(train_transcript_ids), dtype=torch.float32)
    x_evaluation = torch.tensor(evaluation_features, dtype=torch.float32)
    distribution_weight = float(config["models"]["panel_mi"]["aggregate_distribution_loss_weight"])
    individual_weight = float(config["models"]["panel_mi"]["individual_vote_loss_weight"])
    maximum_epochs = (
        int(fixed_epochs) if fixed_epochs is not None else int(settings["panel_maximum_epochs"])
    )
    minimum_epochs = int(settings["panel_minimum_epochs"])
    patience = int(settings["panel_early_stopping_patience"])
    best_score = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    epochs_completed = 0

    for epoch in range(1, maximum_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train)
        head_probabilities = torch.softmax(logits, dim=-1)
        aggregate = head_probabilities.mean(dim=1)
        individual_losses = (
            F.cross_entropy(logits.reshape(-1, n_classes), y_train.reshape(-1), reduction="none")
            .reshape(len(x_train), n_annotators)
            .mean(dim=1)
        )
        distribution_losses = -(q_train * torch.log(aggregate.clamp_min(1e-12))).sum(dim=1)
        data_loss = (
            train_weights
            * (individual_weight * individual_losses + distribution_weight * distribution_losses)
        ).sum() / train_weights.sum()
        bias, annotator_factors = model.centered_deviations()
        shrinkage = (
            bias.square().mean()
            + annotator_factors.square().mean()
            + model.item_projection.weight.square().mean()
            + model.class_factor.square().mean()
        )
        loss = data_loss + float(recipe.annotator_shrinkage) * shrinkage
        if not torch.isfinite(loss):
            raise FloatingPointError("PANEL-MI training loss became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["panel_gradient_norm"]))
        optimizer.step()
        epochs_completed = epoch

        if fixed_epochs is not None or epoch < minimum_epochs:
            continue
        if evaluation_votes is None or evaluation_transcript_ids is None:
            raise ValueError("Validation targets are required for early stopping")
        validation_probabilities, _ = _panel_probabilities(model, x_evaluation)
        score = _weighted_log_score(
            evaluation_votes, validation_probabilities, evaluation_transcript_ids
        )
        if score < best_score - 1e-9:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    if fixed_epochs is None:
        if best_state is None:
            raise RuntimeError("PANEL-MI early stopping did not record a checkpoint")
        model.load_state_dict(best_state)
    else:
        best_epoch = int(fixed_epochs)
    probabilities, head_disagreement = _panel_probabilities(model, x_evaluation)
    if not np.isfinite(probabilities).all() or not np.isfinite(head_disagreement).all():
        raise FloatingPointError("PANEL-MI produced a non-finite prediction")
    return PanelFitOutcome(
        probabilities=probabilities,
        head_disagreement=head_disagreement,
        best_score=None if fixed_epochs is not None else float(best_score),
        best_epoch=best_epoch,
        epochs_completed=epochs_completed,
        base_linear_solver=str(base_model.annomi_solver_used_),
    )


def _linear_recipes(config: dict[str, Any], model: str) -> list[LinearRecipe]:
    recipes = [LinearRecipe(**value) for value in config["models"][model]["recipes"]]
    if len(recipes) != 4 or len({value.recipe_id for value in recipes}) != 4:
        raise ValueError(f"{model} must have four uniquely named recipes")
    return recipes


def _panel_recipes(config: dict[str, Any]) -> list[PanelRecipe]:
    recipes = [PanelRecipe(**value) for value in config["models"]["panel_mi"]["recipes"]]
    if len(recipes) != 4 or len({value.recipe_id for value in recipes}) != 4:
        raise ValueError("panel_mi must have four uniquely named recipes")
    return recipes


def _inner_partitions(
    task_data: MultiAnnotatorTask,
    embeddings: np.ndarray,
    outer_transcript: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    items = task_data.items
    votes = _vote_probabilities(task_data)
    annotation_labels = task_data.annotation_label_indices
    outer_train = items["transcript_id"].ne(outer_transcript).to_numpy()
    partitions: list[dict[str, Any]] = []
    for validation_transcript in sorted(items.loc[outer_train, "transcript_id"].unique()):
        validation = items["transcript_id"].eq(validation_transcript).to_numpy()
        train = outer_train & ~validation
        transform, train_features = _fit_feature_transform(embeddings[train], config)
        partitions.append(
            {
                "validation_transcript": int(validation_transcript),
                "train_features": train_features,
                "train_votes": votes[train],
                "train_annotation_labels": annotation_labels[train],
                "train_transcript_ids": items.loc[train, "transcript_id"].reset_index(drop=True),
                "validation_features": transform.transform(embeddings[validation]),
                "validation_votes": votes[validation],
                "validation_transcript_ids": items.loc[validation, "transcript_id"].reset_index(
                    drop=True
                ),
            }
        )
    if len(partitions) != len(config["study"]["outer_transcripts"]) - 1:
        raise ValueError("Inner leave-one-transcript-out partition count mismatch")
    return partitions


def _select_linear(
    partitions: list[dict[str, Any]],
    config: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    maximum_iterations = int(config["training"]["linear_max_iterations"])
    evaluations: list[dict[str, Any]] = []
    for recipe in _linear_recipes(config, model_name):
        fold_scores: list[dict[str, Any]] = []
        for partition in partitions:
            targets = (
                _hard_targets(partition["train_votes"])
                if model_name == "hard_linear"
                else partition["train_votes"]
            )
            fitted = _fit_linear_model(
                partition["train_features"],
                targets,
                _item_weights(partition["train_transcript_ids"]),
                recipe.inverse_l2,
                maximum_iterations,
                tolerance=float(config["training"]["linear_tolerance"]),
                fallback_solver="newton-cholesky",
            )
            probabilities = fitted.predict_proba(partition["validation_features"])
            score = _weighted_log_score(
                partition["validation_votes"],
                probabilities,
                partition["validation_transcript_ids"],
            )
            fold_scores.append(
                {
                    "validation_transcript": partition["validation_transcript"],
                    "vote_log_score": score,
                    "solver": str(fitted.annomi_solver_used_),
                }
            )
        evaluations.append(
            {
                "recipe_id": recipe.recipe_id,
                "inverse_l2": recipe.inverse_l2,
                "mean_inner_vote_log_score": float(
                    np.mean([value["vote_log_score"] for value in fold_scores])
                ),
                "inner_folds": fold_scores,
            }
        )
    winner = min(
        evaluations,
        key=lambda value: (value["mean_inner_vote_log_score"], value["recipe_id"]),
    )
    return {"selected_recipe": winner["recipe_id"], "candidates": evaluations}


def _select_panel(
    partitions: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    evaluations: list[dict[str, Any]] = []
    for recipe in _panel_recipes(config):
        fold_scores: list[dict[str, Any]] = []
        for partition in partitions:
            outcome = _fit_panel(
                partition["train_features"],
                partition["train_votes"],
                partition["train_annotation_labels"],
                partition["train_transcript_ids"],
                partition["validation_features"],
                partition["validation_votes"],
                partition["validation_transcript_ids"],
                recipe,
                config,
                seed=int(config["selection"]["seed"]),
            )
            fold_scores.append(
                {
                    "validation_transcript": partition["validation_transcript"],
                    "vote_log_score": outcome.best_score,
                    "best_epoch": outcome.best_epoch,
                    "epochs_completed": outcome.epochs_completed,
                    "base_linear_solver": outcome.base_linear_solver,
                }
            )
        evaluations.append(
            {
                "recipe_id": recipe.recipe_id,
                "rank": recipe.rank,
                "annotator_shrinkage": recipe.annotator_shrinkage,
                "mean_inner_vote_log_score": float(
                    np.mean([value["vote_log_score"] for value in fold_scores])
                ),
                "inner_folds": fold_scores,
            }
        )
    winner = min(
        evaluations,
        key=lambda value: (value["mean_inner_vote_log_score"], value["recipe_id"]),
    )
    epochs = [int(value["best_epoch"]) for value in winner["inner_folds"]]
    final_epochs = math.floor(float(np.median(epochs)) + 0.5)
    return {
        "selected_recipe": winner["recipe_id"],
        "final_epochs": final_epochs,
        "candidates": evaluations,
    }


def _find_linear_recipe(config: dict[str, Any], model_name: str, recipe_id: str) -> LinearRecipe:
    return next(
        value for value in _linear_recipes(config, model_name) if value.recipe_id == recipe_id
    )


def _find_panel_recipe(config: dict[str, Any], recipe_id: str) -> PanelRecipe:
    return next(value for value in _panel_recipes(config) if value.recipe_id == recipe_id)


def _transcript_prior(items: pd.DataFrame, votes: np.ndarray) -> np.ndarray:
    per_transcript = []
    for indices in items.groupby("transcript_id", sort=True).indices.values():
        per_transcript.append(votes[np.asarray(indices, dtype=int)].mean(axis=0))
    prior = np.mean(per_transcript, axis=0)
    return prior / prior.sum()


def _ledger(
    task_data: MultiAnnotatorTask,
    indices: np.ndarray,
    probabilities: np.ndarray,
    model_name: str,
    seed: int,
    head_disagreement: np.ndarray | None = None,
) -> pd.DataFrame:
    items = task_data.items.iloc[indices].reset_index(drop=True)
    votes = items[[f"vote_prob_{label}" for label in task_data.labels]].to_numpy(dtype=float)
    log_score, brier, jsd = _row_distribution_losses(votes, probabilities)
    ledger = items[
        [
            "task",
            "transcript_id",
            "utterance_id",
            "source_id",
            "annotation_count",
            "plurality_label",
            "plurality_tie",
            "vote_entropy",
            *[f"vote_prob_{label}" for label in task_data.labels],
        ]
    ].copy()
    ledger.insert(1, "model", model_name)
    ledger.insert(2, "seed", seed)
    ledger.insert(3, "outer_transcript_id", ledger["transcript_id"].to_numpy())
    ledger["prediction"] = np.asarray(task_data.labels, dtype=object)[probabilities.argmax(axis=1)]
    ledger["predicted_entropy"] = _normalized_entropy(probabilities)
    ledger["row_vote_log_score"] = log_score
    ledger["row_vote_brier"] = brier
    ledger["row_jensen_shannon_divergence"] = jsd
    for column, label in enumerate(task_data.labels):
        ledger[f"prob_{label}"] = probabilities[:, column]
    ledger["annotator_head_disagreement"] = (
        np.nan if head_disagreement is None else head_disagreement
    )
    return ledger


def _metrics_from_ledger(
    ledger: pd.DataFrame,
    labels: tuple[str, ...],
    include_head_diagnostic: bool,
) -> dict[str, Any]:
    probabilities = ledger[[f"prob_{label}" for label in labels]].to_numpy(dtype=float)
    head_disagreement = (
        ledger["annotator_head_disagreement"].to_numpy(dtype=float)
        if include_head_diagnostic
        else None
    )
    return evaluate_vote_predictions(ledger, probabilities, labels, head_disagreement)


def _paired_cluster_inference(
    ensemble: pd.DataFrame,
    task: str,
    candidate: str,
    baseline: str,
    config: dict[str, Any],
    seed_offset: int,
) -> dict[str, Any]:
    columns = {
        "vote_log_score": "row_vote_log_score",
        "vote_brier": "row_vote_brier",
        "jensen_shannon_divergence": "row_jensen_shannon_divergence",
    }
    keys = ["transcript_id", "utterance_id"]
    candidate_rows = ensemble[ensemble["task"].eq(task) & ensemble["model"].eq(candidate)]
    baseline_rows = ensemble[ensemble["task"].eq(task) & ensemble["model"].eq(baseline)]
    merged = candidate_rows.merge(
        baseline_rows,
        on=keys,
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    transcripts = sorted(int(value) for value in merged["transcript_id"].unique())
    cluster_deltas: dict[str, dict[str, float]] = {}
    delta_matrix = np.empty((len(transcripts), len(columns)), dtype=float)
    for row_index, transcript_id in enumerate(transcripts):
        group = merged[merged["transcript_id"].eq(transcript_id)]
        values: dict[str, float] = {}
        for column_index, (metric, column) in enumerate(columns.items()):
            delta = float((group[f"{column}_candidate"] - group[f"{column}_baseline"]).mean())
            values[metric] = delta
            delta_matrix[row_index, column_index] = delta
        cluster_deltas[str(transcript_id)] = values

    rng = np.random.default_rng(int(config["inference"]["bootstrap_seed"]) + seed_offset)
    n_resamples = int(config["inference"]["bootstrap_resamples"])
    sampled = rng.integers(0, len(transcripts), size=(n_resamples, len(transcripts)))
    bootstrap = delta_matrix[sampled].mean(axis=1)
    alpha = 1.0 - float(config["inference"]["confidence_level"])
    intervals = {
        metric: {
            "low": float(np.quantile(bootstrap[:, index], alpha / 2)),
            "high": float(np.quantile(bootstrap[:, index], 1 - alpha / 2)),
        }
        for index, metric in enumerate(columns)
    }
    point_deltas = {
        metric: float(delta_matrix[:, index].mean()) for index, metric in enumerate(columns)
    }

    log_deltas = delta_matrix[:, 0]
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(transcripts))))
    null_statistics = (signs * log_deltas[None, :]).mean(axis=1)
    observed = float(log_deltas.mean())
    exact_p = float(np.mean(null_statistics <= observed + 1e-15))
    return {
        "task": task,
        "candidate": candidate,
        "baseline": baseline,
        "direction": "candidate_minus_baseline; negative favors candidate",
        "n_transcript_clusters": len(transcripts),
        "point_deltas": point_deltas,
        "cluster_bootstrap_intervals": intervals,
        "bootstrap_resamples": n_resamples,
        "exact_one_sided_sign_flip_p_vote_log_score": exact_p,
        "improved_transcripts_vote_log_score": int((log_deltas < 0).sum()),
        "tied_transcripts_vote_log_score": int(np.isclose(log_deltas, 0.0).sum()),
        "per_transcript_deltas": cluster_deltas,
    }


def _all_inference(ensemble: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    specifications = [
        ("panel_mi", "soft_linear"),
        ("soft_linear", "transcript_balanced_prior"),
        ("soft_linear", "hard_linear"),
    ]
    offset = 0
    for task in config["tasks"]:
        for candidate, baseline in specifications:
            key = f"{task}:{candidate}_vs_{baseline}"
            comparisons[key] = _paired_cluster_inference(
                ensemble, task, candidate, baseline, config, seed_offset=offset
            )
            offset += 1
    primary = comparisons["therapist:panel_mi_vs_soft_linear"]
    gate = config["candidate_success_gate"]
    checks = {
        "vote_log_score_delta_negative": primary["point_deltas"]["vote_log_score"] < 0,
        "bootstrap_upper_bound_below_zero": primary["cluster_bootstrap_intervals"][
            "vote_log_score"
        ]["high"]
        < 0,
        "exact_one_sided_sign_flip_p_at_most_threshold": primary[
            "exact_one_sided_sign_flip_p_vote_log_score"
        ]
        <= float(gate["exact_one_sided_sign_flip_p_at_most"]),
        "minimum_improved_transcripts": primary["improved_transcripts_vote_log_score"]
        >= int(gate["minimum_improved_transcripts"]),
        "jensen_shannon_divergence_not_increased": primary["point_deltas"][
            "jensen_shannon_divergence"
        ]
        <= 0,
    }
    return {
        "comparisons": comparisons,
        "candidate_success_gate": {"checks": checks, "pass": all(checks.values())},
    }


def _serialize_csv(frame: pd.DataFrame) -> tuple[bytes, pd.DataFrame]:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, lineterminator="\n", float_format="%.10g")
    payload = buffer.getvalue().encode("utf-8")
    reparsed = pd.read_csv(io.BytesIO(payload), dtype={"source_id": str})
    return payload, reparsed


def run_panel_mi(
    corpus: Corpus,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir = output_dir or DEFAULT_PANEL_OUTPUT
    if (output_dir / "summary.json").exists():
        validate_panel_evidence(output_dir)
        return read_json(output_dir / "summary.json")
    protocol, config = _require_registered_state()
    commit = git_commit(ROOT)
    tasks = _tasks(corpus, config)
    embeddings, embedding_metadata = _task_embeddings(tasks, config, commit)
    run_ledgers: list[pd.DataFrame] = []
    ensemble_ledgers: list[pd.DataFrame] = []
    selections: list[dict[str, Any]] = []
    started = time.perf_counter()

    for task_name, task_data in tasks.items():
        items = task_data.items
        votes = _vote_probabilities(task_data)
        for outer_transcript in config["study"]["outer_transcripts"]:
            print(
                f"Starting PANEL-MI {task_name} outer transcript {outer_transcript}",
                flush=True,
            )
            test = items["transcript_id"].eq(outer_transcript).to_numpy()
            train = ~test
            test_indices = np.flatnonzero(test)
            partitions = _inner_partitions(
                task_data, embeddings[task_name], int(outer_transcript), config
            )
            hard_selection = _select_linear(partitions, config, "hard_linear")
            soft_selection = _select_linear(partitions, config, "soft_linear")
            panel_selection = _select_panel(partitions, config)
            selections.append(
                {
                    "task": task_name,
                    "outer_transcript": int(outer_transcript),
                    "hard_linear": hard_selection,
                    "soft_linear": soft_selection,
                    "panel_mi": panel_selection,
                }
            )

            transform, train_features = _fit_feature_transform(embeddings[task_name][train], config)
            test_features = transform.transform(embeddings[task_name][test])
            train_items = items.loc[train].reset_index(drop=True)
            train_votes = votes[train]
            train_weights = _item_weights(train_items["transcript_id"])

            prior = _transcript_prior(train_items, train_votes)
            prior_probabilities = np.repeat(prior[None, :], test.sum(), axis=0)
            prior_ledger = _ledger(
                task_data,
                test_indices,
                prior_probabilities,
                "transcript_balanced_prior",
                seed=-1,
            )
            run_ledgers.append(prior_ledger)
            ensemble_ledgers.append(prior_ledger)

            for model_name, selection in (
                ("hard_linear", hard_selection),
                ("soft_linear", soft_selection),
            ):
                recipe = _find_linear_recipe(config, model_name, selection["selected_recipe"])
                targets = _hard_targets(train_votes) if model_name == "hard_linear" else train_votes
                fitted = _fit_linear_model(
                    train_features,
                    targets,
                    train_weights,
                    recipe.inverse_l2,
                    int(config["training"]["linear_max_iterations"]),
                    tolerance=float(config["training"]["linear_tolerance"]),
                    fallback_solver="newton-cholesky",
                )
                selection["final_solver"] = str(fitted.annomi_solver_used_)
                linear_ledger = _ledger(
                    task_data,
                    test_indices,
                    fitted.predict_proba(test_features),
                    model_name,
                    seed=-1,
                )
                run_ledgers.append(linear_ledger)
                ensemble_ledgers.append(linear_ledger)

            panel_recipe = _find_panel_recipe(config, panel_selection["selected_recipe"])
            seed_probabilities: list[np.ndarray] = []
            seed_disagreements: list[np.ndarray] = []
            final_base_solvers: list[str] = []
            for seed in config["final_seeds"]:
                outcome = _fit_panel(
                    train_features,
                    train_votes,
                    task_data.annotation_label_indices[train],
                    train_items["transcript_id"],
                    test_features,
                    evaluation_votes=None,
                    evaluation_transcript_ids=None,
                    recipe=panel_recipe,
                    config=config,
                    seed=int(seed),
                    fixed_epochs=int(panel_selection["final_epochs"]),
                )
                seed_probabilities.append(outcome.probabilities)
                seed_disagreements.append(outcome.head_disagreement)
                final_base_solvers.append(outcome.base_linear_solver)
                run_ledgers.append(
                    _ledger(
                        task_data,
                        test_indices,
                        outcome.probabilities,
                        "panel_mi",
                        seed=int(seed),
                        head_disagreement=outcome.head_disagreement,
                    )
                )
            panel_selection["final_seed_base_solvers"] = final_base_solvers
            ensemble_ledgers.append(
                _ledger(
                    task_data,
                    test_indices,
                    np.mean(seed_probabilities, axis=0),
                    "panel_mi",
                    seed=-1,
                    head_disagreement=np.mean(seed_disagreements, axis=0),
                )
            )

    runs = pd.concat(run_ledgers, ignore_index=True).sort_values(
        ["task", "model", "seed", "transcript_id", "utterance_id"], kind="stable"
    )
    ensemble = pd.concat(ensemble_ledgers, ignore_index=True).sort_values(
        ["task", "model", "transcript_id", "utterance_id"], kind="stable"
    )
    runs = runs.reset_index(drop=True)
    ensemble = ensemble.reset_index(drop=True)
    runs_payload, runs_roundtrip = _serialize_csv(runs)
    ensemble_payload, ensemble_roundtrip = _serialize_csv(ensemble)

    metrics: dict[str, Any] = {}
    per_seed: dict[str, Any] = {}
    for task_name, specification in config["tasks"].items():
        labels = tuple(specification["labels"])
        metrics[task_name] = {}
        task_ensemble = ensemble_roundtrip[ensemble_roundtrip["task"].eq(task_name)]
        for model_name, group in task_ensemble.groupby("model", sort=True):
            metrics[task_name][model_name] = _metrics_from_ledger(
                group.reset_index(drop=True),
                labels,
                include_head_diagnostic=model_name == "panel_mi",
            )
        task_runs = runs_roundtrip[
            runs_roundtrip["task"].eq(task_name) & runs_roundtrip["model"].eq("panel_mi")
        ]
        per_seed[task_name] = {
            str(int(seed)): _metrics_from_ledger(
                group.reset_index(drop=True), labels, include_head_diagnostic=True
            )
            for seed, group in task_runs.groupby("seed", sort=True)
        }

    inference = _all_inference(ensemble_roundtrip, config)
    selection_payload = canonical_json_bytes({"selection": selections})
    selection_hash = write_create_only(output_dir / "selection.json", selection_payload)
    runs_hash = write_create_only(output_dir / "predictions_by_run.csv", runs_payload)
    ensemble_hash = write_create_only(
        output_dir / "predictions_seed_ensemble.csv", ensemble_payload
    )
    summary = {
        "result_id": "annomi-panel-mi-seven-transcript-loto-v1",
        "protocol_id": protocol["protocol_id"],
        "config_id": config["config_id"],
        "code_commit": commit,
        "config_sha256": sha256_file(PANEL_CONFIG),
        "dataset_sha256": {
            "simple": sha256_file(SIMPLE_DATA),
            "full": sha256_file(FULL_DATA),
        },
        "cohort": {
            "items": sum(len(value.items) for value in tasks.values()),
            "transcripts": config["study"]["outer_transcripts"],
            "annotations_per_item": config["study"]["expected_annotations_per_item"],
            "anonymous_annotators": config["study"]["expected_anonymous_annotators"],
            "items_by_task": {name: len(value.items) for name, value in tasks.items()},
            "plurality_ties_by_task": {
                name: int(value.items["plurality_tie"].sum()) for name, value in tasks.items()
            },
        },
        "embedding": embedding_metadata,
        "final_seeds": config["final_seeds"],
        "metrics": {"seed_ensemble": metrics, "panel_mi_per_seed": per_seed},
        "inference": inference,
        "selection_sha256": selection_hash,
        "prediction_ledger_sha256": runs_hash,
        "ensemble_prediction_ledger_sha256": ensemble_hash,
        "runtime_environment": _runtime_environment(),
        "elapsed_seconds_current_invocation": time.perf_counter() - started,
    }
    write_create_only(output_dir / "summary.json", canonical_json_bytes(summary))
    validate_panel_evidence(output_dir)
    return summary


def run_panel_smoke(corpus: Corpus) -> dict[str, Any]:
    _, config = _require_registered_state()
    output_path = RESEARCH_RESULTS / "gate1" / "panel_mi_smoke_v2.json"
    if output_path.exists():
        return read_json(output_path)
    tasks = _tasks(corpus, config)
    task_data = tasks["therapist"]
    outer_test_transcript = int(config["study"]["outer_transcripts"][0])
    development = task_data.items["transcript_id"].ne(outer_test_transcript).to_numpy()
    validation_transcript = int(min(task_data.items.loc[development, "transcript_id"].unique()))
    validation = task_data.items["transcript_id"].eq(validation_transcript).to_numpy()
    train = development & ~validation
    texts = [
        *task_data.items.loc[train, "role_prefixed_text"].astype(str).tolist(),
        *task_data.items.loc[validation, "role_prefixed_text"].astype(str).tolist(),
    ]
    started = time.perf_counter()
    encoded, peak_memory = _encode_texts(texts, config)
    n_train = int(train.sum())
    transform, train_features = _fit_feature_transform(encoded[:n_train], config)
    validation_features = transform.transform(encoded[n_train:])
    votes = _vote_probabilities(task_data)
    recipe = _panel_recipes(config)[0]
    outcome = _fit_panel(
        train_features,
        votes[train],
        task_data.annotation_label_indices[train],
        task_data.items.loc[train, "transcript_id"].reset_index(drop=True),
        validation_features,
        votes[validation],
        task_data.items.loc[validation, "transcript_id"].reset_index(drop=True),
        recipe,
        config,
        seed=1907,
        fixed_epochs=2,
    )
    payload = {
        "gate_id": "panel-mi-frozen-encoder-smoke-v1",
        "status": "pass",
        "engineering_gate_not_performance_result": True,
        "outer_test_votes_touched": False,
        "outer_test_transcript": outer_test_transcript,
        "development_validation_transcript": validation_transcript,
        "train_items": n_train,
        "validation_items": int(validation.sum()),
        "optimizer_epochs": outcome.epochs_completed,
        "probabilities_finite": bool(np.isfinite(outcome.probabilities).all()),
        "maximum_probability_sum_error": float(
            np.max(np.abs(outcome.probabilities.sum(axis=1) - 1.0))
        ),
        "embedding_dimensions": int(encoded.shape[1]),
        "projected_dimensions": int(train_features.shape[1]),
        "peak_encoder_memory_bytes": peak_memory,
        "code_commit": git_commit(ROOT),
        "config_sha256": sha256_file(PANEL_CONFIG),
        "elapsed_seconds": time.perf_counter() - started,
        "runtime_environment": _runtime_environment(),
    }
    write_create_only(output_path, canonical_json_bytes(payload))
    return payload


def _nested_close(observed: Any, expected: Any, context: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(observed) != set(expected):
            raise ValueError(f"PANEL-MI evidence keys mismatch at {context}")
        for key in expected:
            _nested_close(observed[key], expected[key], f"{context}/{key}")
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise ValueError(f"PANEL-MI evidence list mismatch at {context}")
        for index, (left, right) in enumerate(zip(observed, expected, strict=True)):
            _nested_close(left, right, f"{context}/{index}")
        return
    if isinstance(expected, float):
        if not np.isclose(float(observed), expected, atol=1e-9, rtol=1e-9):
            raise ValueError(f"PANEL-MI evidence value mismatch at {context}")
        return
    if observed != expected:
        raise ValueError(f"PANEL-MI evidence value mismatch at {context}")


def validate_panel_evidence(output_dir: Path) -> None:
    summary = read_json(output_dir / "summary.json")
    config = read_json(PANEL_CONFIG)
    run_path = output_dir / "predictions_by_run.csv"
    ensemble_path = output_dir / "predictions_seed_ensemble.csv"
    selection_path = output_dir / "selection.json"
    if sha256_file(run_path) != summary["prediction_ledger_sha256"]:
        raise ValueError("PANEL-MI run-ledger hash mismatch")
    if sha256_file(ensemble_path) != summary["ensemble_prediction_ledger_sha256"]:
        raise ValueError("PANEL-MI ensemble-ledger hash mismatch")
    if sha256_file(selection_path) != summary["selection_sha256"]:
        raise ValueError("PANEL-MI selection hash mismatch")
    runs = pd.read_csv(run_path, dtype={"source_id": str})
    ensemble = pd.read_csv(ensemble_path, dtype={"source_id": str})
    forbidden = {"utterance_text", "normalized_text", "role_prefixed_text", "annotator_id"}
    if forbidden & set(runs.columns) or forbidden & set(ensemble.columns):
        raise ValueError("PANEL-MI ledger exposes text or annotator identity")
    keys = ["task", "model", "seed", "transcript_id", "utterance_id"]
    if runs.duplicated(keys).any():
        raise ValueError("PANEL-MI run ledger contains duplicate predictions")
    if ensemble.duplicated(["task", "model", "transcript_id", "utterance_id"]).any():
        raise ValueError("PANEL-MI ensemble ledger contains duplicate predictions")

    rebuilt_metrics: dict[str, Any] = {}
    rebuilt_per_seed: dict[str, Any] = {}
    for task_name, specification in config["tasks"].items():
        labels = tuple(specification["labels"])
        expected_items = int(specification["expected_items"])
        task_ensemble = ensemble[ensemble["task"].eq(task_name)]
        rebuilt_metrics[task_name] = {}
        for model_name in (
            "transcript_balanced_prior",
            "hard_linear",
            "soft_linear",
            "panel_mi",
        ):
            group = task_ensemble[task_ensemble["model"].eq(model_name)].reset_index(drop=True)
            if len(group) != expected_items:
                raise ValueError(
                    f"Incomplete PANEL-MI ensemble coverage for {task_name}/{model_name}"
                )
            rebuilt_metrics[task_name][model_name] = _metrics_from_ledger(
                group, labels, include_head_diagnostic=model_name == "panel_mi"
            )

        task_panel_runs = runs[runs["task"].eq(task_name) & runs["model"].eq("panel_mi")]
        rebuilt_per_seed[task_name] = {}
        for seed in config["final_seeds"]:
            group = task_panel_runs[task_panel_runs["seed"].eq(seed)].reset_index(drop=True)
            if len(group) != expected_items:
                raise ValueError(f"Incomplete PANEL-MI seed coverage for {task_name}/{seed}")
            rebuilt_per_seed[task_name][str(seed)] = _metrics_from_ledger(
                group, labels, include_head_diagnostic=True
            )

        panel_ensemble = task_ensemble[task_ensemble["model"].eq("panel_mi")].sort_values(
            ["transcript_id", "utterance_id"], kind="stable"
        )
        for label in labels:
            mean_probability = (
                task_panel_runs.groupby(["transcript_id", "utterance_id"], sort=True)[
                    f"prob_{label}"
                ]
                .mean()
                .to_numpy()
            )
            if not np.allclose(
                panel_ensemble[f"prob_{label}"].to_numpy(dtype=float),
                mean_probability,
                atol=2e-9,
            ):
                raise ValueError("PANEL-MI ensemble is not the mean of seed probabilities")

    rebuilt = {
        "seed_ensemble": rebuilt_metrics,
        "panel_mi_per_seed": rebuilt_per_seed,
    }
    _nested_close(summary["metrics"], rebuilt, "metrics")
    _nested_close(summary["inference"], _all_inference(ensemble, config), "inference")
