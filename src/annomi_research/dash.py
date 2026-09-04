from __future__ import annotations

import gc
import hashlib
import io
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

from .constants import (
    ARTIFACTS,
    DASH_CONFIG,
    FULL_DATA,
    FULL_MANIFEST,
    LABELS,
    PROTOCOL,
    RESEARCH_RESULTS,
    ROOT,
    SIMPLE_DATA,
    SIMPLE_MANIFEST,
)
from .data import Corpus, add_therapist_vote_distributions, build_therapist_examples
from .io import canonical_json_bytes, git_commit, read_json, sha256_file, write_create_only
from .metrics import evaluate_predictions
from .neural import (
    _assert_metric_close,
    _git_is_clean,
    _load_cached_fold,
    _loader,
    _require_cuda_bf16,
    _round_median_epoch,
    _runtime_environment,
    _schedule_multiplier,
    _seed_everything,
    _stratified_cap,
    _training_weights,
    _validation_score,
    _write_cached_fold,
    validate_neural_evidence,
)
from .splits import fold_lookup


@dataclass(frozen=True)
class DashRecipe:
    recipe_id: str
    encoder_learning_rate: float
    head_learning_rate: float
    history_max_length: int
    disagreement_mix: float


@dataclass(frozen=True)
class DashPredictions:
    probabilities: np.ndarray
    target_only_probabilities: np.ndarray
    context_gate_mean: np.ndarray
    context_attention_entropy: np.ndarray
    context_attention_max: np.ndarray
    context_residual_l2: np.ndarray


@dataclass(frozen=True)
class DashFitOutcome:
    best_score: float | None
    best_epoch: int
    epochs_completed: int
    optimizer_steps: int
    peak_memory_bytes: int
    predictions: DashPredictions | None = None


class DashMIModel(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        encoder = config["pretrained_encoder"]
        self.encoder = AutoModel.from_pretrained(
            encoder["model_id"],
            revision=encoder["revision"],
            cache_dir=ARTIFACTS / "huggingface",
            trust_remote_code=bool(encoder["trust_remote_code"]),
            add_pooling_layer=False,
        )
        architecture = config["architecture"]
        hidden = int(self.encoder.config.hidden_size)
        dropout = float(architecture["dropout"])
        self.dropout = nn.Dropout(dropout)
        self.target_dense = nn.Linear(hidden, hidden)
        self.target_output = nn.Linear(hidden, len(LABELS))
        self.context_attention = nn.MultiheadAttention(
            hidden,
            int(architecture["attention_heads"]),
            dropout=dropout,
            batch_first=True,
        )
        self.context_projection = nn.Linear(hidden, hidden)
        self.context_gate = nn.Linear(hidden * 4, hidden)
        self.context_normalization = nn.LayerNorm(hidden)
        self.residual_dense = nn.Linear(hidden, hidden)
        self.residual_output = nn.Linear(hidden, len(LABELS))
        nn.init.constant_(
            self.context_gate.bias, float(architecture["initial_gate_bias"])
        )
        if architecture["zero_initialize_context_residual_output"]:
            nn.init.zeros_(self.residual_output.weight)
            nn.init.zeros_(self.residual_output.bias)

    def forward(
        self,
        target_input_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
        history_input_ids: torch.Tensor,
        history_attention_mask: torch.Tensor,
        context_available: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        target_states = self.encoder(
            input_ids=target_input_ids,
            attention_mask=target_attention_mask,
        ).last_hidden_state
        history_states = self.encoder(
            input_ids=history_input_ids,
            attention_mask=history_attention_mask,
        ).last_hidden_state
        target_state = target_states[:, 0]
        target_hidden = torch.tanh(self.target_dense(self.dropout(target_state)))
        target_logits = self.target_output(self.dropout(target_hidden))

        attended, attention = self.context_attention(
            query=target_state.unsqueeze(1),
            key=history_states,
            value=history_states,
            key_padding_mask=~history_attention_mask.bool(),
            need_weights=True,
            average_attn_weights=True,
        )
        context_state = attended.squeeze(1)
        gate_features = torch.cat(
            [
                target_state,
                context_state,
                target_state * context_state,
                torch.abs(target_state - context_state),
            ],
            dim=-1,
        )
        gate = torch.sigmoid(self.context_gate(self.dropout(gate_features)))
        context_delta = gate * torch.tanh(self.context_projection(context_state))
        context_delta = self.context_normalization(context_delta)
        residual_hidden = torch.tanh(self.residual_dense(self.dropout(context_delta)))
        residual_logits = self.residual_output(self.dropout(residual_hidden))
        available = context_available.to(residual_logits.dtype).unsqueeze(1)
        residual_logits = residual_logits * available
        logits = target_logits + residual_logits

        attention = attention.squeeze(1).float()
        mask = history_attention_mask.bool()
        attention = attention.masked_fill(~mask, 0.0)
        valid_tokens = mask.sum(dim=1).clamp_min(2).float()
        entropy = -(attention.clamp_min(1e-12) * attention.clamp_min(1e-12).log()).sum(
            dim=1
        ) / valid_tokens.log()
        diagnostics = {
            "context_gate_mean": gate.float().mean(dim=1) * context_available.float(),
            "context_attention_entropy": entropy * context_available.float(),
            "context_attention_max": attention.max(dim=1).values * context_available.float(),
            "context_residual_l2": residual_logits.float().norm(dim=1),
        }
        return logits, target_logits, diagnostics


def _require_registered_state() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = read_json(PROTOCOL)
    config = read_json(DASH_CONFIG)
    if config["status"] != "registered_before_dash_mi_evaluation":
        raise ValueError("DASH-MI configuration is not prospectively registered")
    if config["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("DASH-MI configuration and research protocol disagree")
    if config["selection"]["inner_folds"] != protocol["development"]["inner_folds"]:
        raise ValueError("DASH-MI and protocol inner-fold counts disagree")
    if config["final_seeds"] != protocol["development"]["seeds"]:
        raise ValueError("DASH-MI and protocol seed lists disagree")
    if config["input_contract"]["context_turns"] != protocol["data"]["context_turns"]:
        raise ValueError("DASH-MI and protocol context windows disagree")
    for data_path, manifest_path in (
        (SIMPLE_DATA, SIMPLE_MANIFEST),
        (FULL_DATA, FULL_MANIFEST),
    ):
        if sha256_file(data_path) != read_json(manifest_path)["sha256"]:
            raise ValueError(f"Dataset hash mismatch: {data_path}")
    if not _git_is_clean():
        raise RuntimeError("Commit tracked code/config changes before running DASH-MI evidence")
    return protocol, config


def _recipes(config: dict[str, Any]) -> list[DashRecipe]:
    recipes = [DashRecipe(**value) for value in config["recipes"]]
    if len(recipes) != 4 or len({recipe.recipe_id for recipe in recipes}) != len(recipes):
        raise ValueError("DASH-MI must define four uniquely named recipes")
    if any(not 0.0 <= recipe.disagreement_mix <= 1.0 for recipe in recipes):
        raise ValueError("DASH-MI disagreement mixing must be between zero and one")
    return recipes


def _soft_training_targets(frame: pd.DataFrame, disagreement_mix: float) -> np.ndarray:
    labels = np.asarray([LABELS.index(str(value)) for value in frame["label"]], dtype=int)
    one_hot = np.eye(len(LABELS), dtype=np.float32)[labels]
    if disagreement_mix == 0.0:
        return one_hot
    vote_columns = [f"vote_prob_{label}" for label in LABELS]
    votes = frame[vote_columns].to_numpy(dtype=np.float32)
    targets = (1.0 - disagreement_mix) * one_hot + disagreement_mix * votes
    if not np.allclose(targets.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("DASH-MI soft training targets do not sum to one")
    return targets


def _tensor_dataset(
    frame: pd.DataFrame,
    tokenizer: Any,
    config: dict[str, Any],
    recipe: DashRecipe,
    include_training_weights: bool,
) -> TensorDataset:
    target = tokenizer(
        frame["utterance_text"].astype(str).tolist(),
        max_length=int(config["architecture"]["target_max_length"]),
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    history = tokenizer(
        frame["recent_history_text"].astype(str).tolist(),
        max_length=recipe.history_max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    label_ids = torch.tensor([LABELS.index(str(value)) for value in frame["label"]])
    weights = (
        _training_weights(frame)
        if include_training_weights
        else np.ones(len(frame), dtype=float)
    )
    disagreement_mix = recipe.disagreement_mix if include_training_weights else 0.0
    targets = _soft_training_targets(frame, disagreement_mix)
    context_available = frame["recent_history_text"].astype(str).str.len().gt(0).to_numpy()
    return TensorDataset(
        target["input_ids"],
        target["attention_mask"],
        history["input_ids"],
        history["attention_mask"],
        label_ids,
        torch.tensor(weights, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
        torch.tensor(context_available, dtype=torch.float32),
    )


def _new_model(config: dict[str, Any], device: torch.device) -> DashMIModel:
    return DashMIModel(config).to(device)


@torch.inference_mode()
def _predict(
    model: DashMIModel,
    loader: DataLoader,
    device: torch.device,
) -> DashPredictions:
    model.eval()
    probabilities: list[np.ndarray] = []
    target_probabilities: list[np.ndarray] = []
    diagnostic_values: dict[str, list[np.ndarray]] = {
        "context_gate_mean": [],
        "context_attention_entropy": [],
        "context_attention_max": [],
        "context_residual_l2": [],
    }
    for batch in loader:
        target_ids, target_mask, history_ids, history_mask, _, _, _, available = batch
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, target_logits, diagnostics = model(
                target_ids.to(device, non_blocking=True),
                target_mask.to(device, non_blocking=True),
                history_ids.to(device, non_blocking=True),
                history_mask.to(device, non_blocking=True),
                available.to(device, non_blocking=True),
            )
        probabilities.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        target_probabilities.append(
            torch.softmax(target_logits.float(), dim=-1).cpu().numpy()
        )
        for name, value in diagnostics.items():
            diagnostic_values[name].append(value.cpu().numpy())

    main = np.concatenate(probabilities, axis=0)
    target_only = np.concatenate(target_probabilities, axis=0)
    for name, values in (("full", main), ("target-only", target_only)):
        if not np.isfinite(values).all():
            raise FloatingPointError(f"DASH-MI {name} prediction is non-finite")
        if not np.allclose(values.sum(axis=1), 1.0, atol=1e-5):
            raise FloatingPointError(f"DASH-MI {name} probabilities do not sum to one")
    return DashPredictions(
        probabilities=main,
        target_only_probabilities=target_only,
        **{
            name: np.concatenate(values, axis=0)
            for name, values in diagnostic_values.items()
        },
    )


def _fit_once(
    train: pd.DataFrame,
    validation_or_test: pd.DataFrame,
    tokenizer: Any,
    config: dict[str, Any],
    recipe: DashRecipe,
    seed: int,
    fixed_epochs: int | None = None,
) -> DashFitOutcome:
    device = _require_cuda_bf16()
    _seed_everything(seed)
    settings = config["training"]
    architecture = config["architecture"]
    train_data = _tensor_dataset(train, tokenizer, config, recipe, True)
    eval_data = _tensor_dataset(validation_or_test, tokenizer, config, recipe, False)
    train_loader = _loader(
        train_data, int(settings["train_batch_size"]), shuffle=True, seed=seed
    )
    eval_loader = _loader(
        eval_data, int(settings["eval_batch_size"]), shuffle=False, seed=seed
    )
    maximum_epochs = fixed_epochs or int(settings["maximum_epochs"])
    accumulation = int(settings["gradient_accumulation_steps"])
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_steps = updates_per_epoch * maximum_epochs
    warmup_steps = round(float(settings["warmup_fraction"]) * total_steps)

    model = _new_model(config, device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": recipe.encoder_learning_rate,
            },
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if not name.startswith("encoder.")
                ],
                "lr": recipe.head_learning_rate,
            },
        ],
        weight_decay=float(settings["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _schedule_multiplier(step, warmup_steps, total_steps),
    )
    torch.cuda.reset_peak_memory_stats(device)
    best_score: float | None = None
    best_epoch = maximum_epochs if fixed_epochs else 1
    stale_epochs = 0
    optimizer_steps = 0
    epochs_completed = 0

    try:
        for epoch in range(1, maximum_epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            for batch_index, batch in enumerate(train_loader, start=1):
                (
                    target_ids,
                    target_mask,
                    history_ids,
                    history_mask,
                    _,
                    weights,
                    soft_targets,
                    context_available,
                ) = batch
                available = context_available.to(device, non_blocking=True)
                dropout = float(architecture["context_example_dropout"])
                if dropout:
                    available = available * (
                        torch.rand(available.shape, device=device) >= dropout
                    )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, target_logits, _ = model(
                        target_ids.to(device, non_blocking=True),
                        target_mask.to(device, non_blocking=True),
                        history_ids.to(device, non_blocking=True),
                        history_mask.to(device, non_blocking=True),
                        available,
                    )
                    targets = soft_targets.to(device, non_blocking=True)
                    full_losses = -(targets * F.log_softmax(logits.float(), dim=-1)).sum(
                        dim=-1
                    )
                    target_losses = -(
                        targets * F.log_softmax(target_logits.float(), dim=-1)
                    ).sum(dim=-1)
                    auxiliary_weight = float(
                        architecture["target_auxiliary_loss_weight"]
                    )
                    row_losses = (
                        full_losses + auxiliary_weight * target_losses
                    ) / (1.0 + auxiliary_weight)
                    loss = (
                        row_losses * weights.to(device, non_blocking=True)
                    ).mean() / accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError("DASH-MI training produced a non-finite loss")
                loss.backward()
                running_loss += float(loss.detach()) * accumulation
                should_step = batch_index % accumulation == 0 or batch_index == len(
                    train_loader
                )
                if should_step:
                    nn.utils.clip_grad_norm_(
                        model.parameters(), float(settings["maximum_gradient_norm"])
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
            epochs_completed = epoch

            if fixed_epochs is not None:
                print(
                    f"dash_mi/{recipe.recipe_id}/seed={seed}: "
                    f"epoch {epoch}/{maximum_epochs}, "
                    f"train_loss={running_loss / len(train_loader):.4f}",
                    flush=True,
                )
                continue

            predictions = _predict(model, eval_loader, device)
            score = _validation_score(validation_or_test, predictions.probabilities)
            print(
                f"dash_mi/{recipe.recipe_id}/seed={seed}: epoch {epoch}, "
                f"inner_source_macro_f1={score:.4f}",
                flush=True,
            )
            if best_score is None or score > best_score + 1e-8:
                best_score = score
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
            if (
                epoch >= int(settings["minimum_epochs"])
                and stale_epochs >= int(settings["early_stopping_patience"])
            ):
                break

        final_predictions = (
            _predict(model, eval_loader, device) if fixed_epochs is not None else None
        )
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    finally:
        del model
        del optimizer
        del scheduler
        gc.collect()
        torch.cuda.empty_cache()

    return DashFitOutcome(
        best_score=best_score,
        best_epoch=best_epoch,
        epochs_completed=epochs_completed,
        optimizer_steps=optimizer_steps,
        peak_memory_bytes=peak_memory,
        predictions=final_predictions,
    )


def _tokenizer(config: dict[str, Any]) -> Any:
    encoder = config["pretrained_encoder"]
    return AutoTokenizer.from_pretrained(
        encoder["model_id"],
        revision=encoder["revision"],
        cache_dir=ARTIFACTS / "huggingface",
        trust_remote_code=bool(encoder["trust_remote_code"]),
        use_fast=True,
    )


def _ledger_rows(
    frame: pd.DataFrame,
    predictions: DashPredictions,
    fold: int,
    seed: int,
    recipe: DashRecipe,
    epochs: int,
    train_texts: set[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    predicted = np.asarray(LABELS, dtype=object)[predictions.probabilities.argmax(axis=1)]
    target_predicted = np.asarray(LABELS, dtype=object)[
        predictions.target_only_probabilities.argmax(axis=1)
    ]
    ledger = frame[
        [
            "transcript_id",
            "utterance_id",
            "source_id",
            "label",
            "annotation_count",
            "vote_entropy",
            "hard_label_vote_probability",
            "annotator_disagreement",
        ]
    ].copy()
    ledger.insert(0, "model", "dash_mi")
    ledger.insert(1, "seed", seed)
    ledger.insert(2, "outer_fold", fold)
    ledger["prediction"] = predicted
    ledger["target_only_prediction"] = target_predicted
    for index, label in enumerate(LABELS):
        ledger[f"prob_{label}"] = predictions.probabilities[:, index]
        ledger[f"prob_target_only_{label}"] = predictions.target_only_probabilities[
            :, index
        ]
    ledger["context_gate_mean"] = predictions.context_gate_mean
    ledger["context_attention_entropy"] = predictions.context_attention_entropy
    ledger["context_attention_max"] = predictions.context_attention_max
    ledger["context_residual_l2"] = predictions.context_residual_l2
    ledger["context_available"] = frame["recent_history_text"].astype(str).str.len().gt(0)
    ledger["seen_text_in_outer_train"] = frame["normalized_text"].isin(train_texts).to_numpy()
    ledger["normalized_text_sha256"] = frame["normalized_text"].map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    ledger["selected_recipe"] = recipe.recipe_id
    ledger["max_length"] = (
        f"target{config['architecture']['target_max_length']}+history"
        f"{recipe.history_max_length}"
    )
    ledger["target_max_length"] = int(config["architecture"]["target_max_length"])
    ledger["history_max_length"] = recipe.history_max_length
    ledger["disagreement_mix"] = recipe.disagreement_mix
    ledger["epochs"] = epochs
    ledger["pretrained_revision"] = config["pretrained_encoder"]["revision"]
    return ledger


def _cache_dir(commit: str, config_sha256: str) -> Path:
    return ARTIFACTS / "dash_mi_v1" / f"{commit[:12]}_{config_sha256[:12]}"


def _run_outer_fold(
    examples: pd.DataFrame,
    fold: int,
    protocol: dict[str, Any],
    config: dict[str, Any],
    tokenizer: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train = examples[examples["outer_fold"].ne(fold)].reset_index(drop=True)
    test = examples[examples["outer_fold"].eq(fold)].reset_index(drop=True)
    selection_seed = int(config["selection"]["seed"])
    splitter = StratifiedGroupKFold(
        n_splits=int(config["selection"]["inner_folds"]),
        shuffle=True,
        random_state=selection_seed + fold,
    )
    inner_splits = list(
        splitter.split(np.arange(len(train)), train["label"], groups=train["source_id"])
    )
    candidate_records: list[dict[str, Any]] = []
    for recipe in _recipes(config):
        inner_records: list[dict[str, Any]] = []
        for inner_fold, (train_indices, validation_indices) in enumerate(inner_splits):
            inner_train = train.iloc[train_indices].reset_index(drop=True)
            outcome = _fit_once(
                inner_train,
                train.iloc[validation_indices].reset_index(drop=True),
                tokenizer,
                config,
                recipe,
                selection_seed + fold * 10 + inner_fold,
            )
            inner_records.append(
                {
                    "inner_fold": inner_fold,
                    "source_balanced_macro_f1": outcome.best_score,
                    "best_epoch": outcome.best_epoch,
                    "epochs_completed": outcome.epochs_completed,
                    "peak_memory_bytes": outcome.peak_memory_bytes,
                    "multiannotated_training_rows": int(
                        inner_train["annotation_count"].gt(1).sum()
                    ),
                }
            )
        candidate_records.append(
            {
                "recipe_id": recipe.recipe_id,
                "mean_inner_source_balanced_macro_f1": float(
                    np.mean([record["source_balanced_macro_f1"] for record in inner_records])
                ),
                "inner_folds": inner_records,
            }
        )
    selected_record = min(
        candidate_records,
        key=lambda item: (-item["mean_inner_source_balanced_macro_f1"], item["recipe_id"]),
    )
    selected_recipe = next(
        recipe
        for recipe in _recipes(config)
        if recipe.recipe_id == selected_record["recipe_id"]
    )
    selected_epochs = _round_median_epoch(
        [record["best_epoch"] for record in selected_record["inner_folds"]]
    )
    ledgers: list[pd.DataFrame] = []
    seed_records: list[dict[str, Any]] = []
    train_texts = set(train["normalized_text"])
    for seed in config["final_seeds"]:
        outcome = _fit_once(
            train,
            test,
            tokenizer,
            config,
            selected_recipe,
            int(seed),
            fixed_epochs=selected_epochs,
        )
        if outcome.predictions is None:
            raise AssertionError("Final DASH-MI training did not produce predictions")
        ledgers.append(
            _ledger_rows(
                test,
                outcome.predictions,
                fold,
                int(seed),
                selected_recipe,
                selected_epochs,
                train_texts,
                config,
            )
        )
        seed_records.append(
            {
                "seed": int(seed),
                "optimizer_steps": outcome.optimizer_steps,
                "peak_memory_bytes": outcome.peak_memory_bytes,
            }
        )
    return pd.concat(ledgers, ignore_index=True), {
        "outer_fold": fold,
        "n_outer_train": len(train),
        "n_outer_test": len(test),
        "multiannotated_outer_train_rows": int(train["annotation_count"].gt(1).sum()),
        "multiannotated_outer_test_rows": int(test["annotation_count"].gt(1).sum()),
        "selected_recipe": selected_recipe.recipe_id,
        "selected_epochs": selected_epochs,
        "candidate_records": candidate_records,
        "final_seed_records": seed_records,
        "outer_test_used_for_selection": False,
        "protocol_id": protocol["protocol_id"],
    }


def _ensemble_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    keys = ["outer_fold", "transcript_id", "utterance_id", "source_id", "label"]
    probability_columns = [f"prob_{label}" for label in LABELS]
    target_columns = [f"prob_target_only_{label}" for label in LABELS]
    invariant_columns = [
        "annotation_count",
        "vote_entropy",
        "hard_label_vote_probability",
        "annotator_disagreement",
        "context_available",
        "seen_text_in_outer_train",
        "normalized_text_sha256",
        "selected_recipe",
        "max_length",
        "target_max_length",
        "history_max_length",
        "disagreement_mix",
        "epochs",
        "pretrained_revision",
    ]
    diagnostic_columns = [
        "context_gate_mean",
        "context_attention_entropy",
        "context_attention_max",
        "context_residual_l2",
    ]
    grouped = ledger.groupby(keys, sort=False, dropna=False)
    for column in invariant_columns:
        if grouped[column].nunique(dropna=False).max() != 1:
            raise ValueError(f"DASH-MI seed ledgers disagree on {column}")
    ensemble = grouped[[*probability_columns, *target_columns, *diagnostic_columns]].mean()
    ensemble = ensemble.reset_index()
    invariants = grouped[invariant_columns].first().reset_index()
    ensemble = ensemble.merge(invariants, on=keys, validate="one_to_one")
    ensemble.insert(0, "model", "dash_mi_seed_ensemble")
    ensemble.insert(1, "seed", "mean")
    ensemble["prediction"] = np.asarray(LABELS, dtype=object)[
        ensemble[probability_columns].to_numpy(dtype=float).argmax(axis=1)
    ]
    ensemble["target_only_prediction"] = np.asarray(LABELS, dtype=object)[
        ensemble[target_columns].to_numpy(dtype=float).argmax(axis=1)
    ]
    return ensemble


def _target_only_view(ledger: pd.DataFrame) -> pd.DataFrame:
    result = ledger.copy()
    result["prediction"] = result["target_only_prediction"]
    for label in LABELS:
        result[f"prob_{label}"] = result[f"prob_target_only_{label}"]
    return result


def _diagnostic_summary(ensemble: pd.DataFrame) -> dict[str, Any]:
    return {
        "context_available_rows": int(ensemble["context_available"].astype(bool).sum()),
        "context_changed_prediction_rows": int(
            ensemble["prediction"].ne(ensemble["target_only_prediction"]).sum()
        ),
        "context_changed_prediction_fraction": float(
            ensemble["prediction"].ne(ensemble["target_only_prediction"]).mean()
        ),
        "mean_context_gate": float(ensemble["context_gate_mean"].mean()),
        "mean_context_attention_entropy": float(
            ensemble["context_attention_entropy"].mean()
        ),
        "mean_context_attention_max": float(ensemble["context_attention_max"].mean()),
        "mean_context_residual_l2": float(ensemble["context_residual_l2"].mean()),
    }


def run_dash_mi(
    corpus: Corpus,
    split_manifest: dict[str, Any],
    output_dir: Path | None = None,
) -> dict[str, Any]:
    protocol, config = _require_registered_state()
    _recipes(config)
    output_dir = output_dir or RESEARCH_RESULTS / "neural_v1" / "dash_mi"
    if (output_dir / "summary.json").exists():
        validate_dash_evidence(output_dir)
        return read_json(output_dir / "summary.json")

    _require_cuda_bf16()
    commit = git_commit(ROOT)
    config_sha256 = sha256_file(DASH_CONFIG)
    expected_cache = {
        "model": "dash_mi",
        "code_commit": commit,
        "config_sha256": config_sha256,
        "split_manifest_sha256": split_manifest["manifest_sha256"],
    }
    examples = add_therapist_vote_distributions(
        build_therapist_examples(corpus, int(protocol["data"]["context_turns"])),
        corpus.annotations,
    )
    examples["outer_fold"] = examples["source_id"].map(fold_lookup(split_manifest))
    if examples["outer_fold"].isna().any():
        raise ValueError("At least one source lacks an outer-fold assignment")

    tokenizer = _tokenizer(config)
    cache_dir = _cache_dir(commit, config_sha256)
    fold_ledgers: list[pd.DataFrame] = []
    selections: list[dict[str, Any]] = []
    started = time.perf_counter()
    for fold in range(int(split_manifest["n_splits"])):
        cached = _load_cached_fold(cache_dir, fold, expected_cache)
        if cached is None:
            print(f"Starting dash_mi outer fold {fold}", flush=True)
            fold_started = time.perf_counter()
            fold_ledger, selection = _run_outer_fold(
                examples, fold, protocol, config, tokenizer
            )
            metadata = {
                **expected_cache,
                "selection": selection,
                "elapsed_seconds": time.perf_counter() - fold_started,
            }
            _write_cached_fold(cache_dir, fold, fold_ledger, metadata)
        else:
            fold_ledger, metadata = cached
            selection = metadata["selection"]
            print(f"Loaded completed dash_mi outer fold {fold} from cache", flush=True)
        fold_ledgers.append(fold_ledger)
        selections.append(selection)

    ledger = pd.concat(fold_ledgers, ignore_index=True).sort_values(
        ["seed", "outer_fold", "transcript_id", "utterance_id"], kind="stable"
    )
    ledger = ledger.reset_index(drop=True)
    expected_rows = len(examples)
    if ledger.duplicated(["seed", "transcript_id", "utterance_id"]).any():
        raise ValueError("Duplicate out-of-fold DASH-MI prediction")
    counts = ledger.groupby("seed").size()
    if len(counts) != len(config["final_seeds"]) or not counts.eq(expected_rows).all():
        raise ValueError(f"Incomplete DASH-MI out-of-fold coverage: {counts.to_dict()}")

    ensemble = _ensemble_ledger(ledger).sort_values(
        ["outer_fold", "transcript_id", "utterance_id"], kind="stable"
    )
    ensemble = ensemble.reset_index(drop=True)
    per_seed_metrics = {
        str(seed): evaluate_predictions(group.reset_index(drop=True))
        for seed, group in ledger.groupby("seed", sort=True)
    }
    target_per_seed_metrics = {
        str(seed): evaluate_predictions(_target_only_view(group.reset_index(drop=True)))
        for seed, group in ledger.groupby("seed", sort=True)
    }
    ensemble_metrics = evaluate_predictions(ensemble)
    target_ensemble_metrics = evaluate_predictions(_target_only_view(ensemble))

    ledger_buffer = io.StringIO()
    ledger.to_csv(ledger_buffer, index=False, lineterminator="\n", float_format="%.10g")
    ensemble_buffer = io.StringIO()
    ensemble.to_csv(
        ensemble_buffer, index=False, lineterminator="\n", float_format="%.10g"
    )
    ledger_hash = write_create_only(
        output_dir / "predictions_by_seed.csv", ledger_buffer.getvalue().encode("utf-8")
    )
    ensemble_hash = write_create_only(
        output_dir / "predictions_seed_ensemble.csv",
        ensemble_buffer.getvalue().encode("utf-8"),
    )
    metadata = {
        "result_id": "annomi-dash-mi-source-cv-neural-v1",
        "protocol_id": protocol["protocol_id"],
        "model": "dash_mi",
        "development_status": config["development_status"],
        "pretrained_encoder": config["pretrained_encoder"],
        "architecture": config["architecture"],
        "code_commit": commit,
        "config_sha256": config_sha256,
        "split_manifest_sha256": split_manifest["manifest_sha256"],
        "dataset_sha256": {
            "simple": read_json(ROOT / protocol["data"]["simple_manifest"])["sha256"],
            "full": read_json(ROOT / protocol["data"]["full_manifest"])["sha256"],
        },
        "seeds": config["final_seeds"],
        "selection": selections,
        "multiannotator_rows": int(examples["annotation_count"].gt(1).sum()),
        "metrics": {
            "seed_ensemble": ensemble_metrics,
            "per_seed": per_seed_metrics,
            "target_only_ablation_seed_ensemble": target_ensemble_metrics,
            "target_only_ablation_per_seed": target_per_seed_metrics,
        },
        "context_diagnostics": _diagnostic_summary(ensemble),
        "prediction_ledger_sha256": ledger_hash,
        "ensemble_prediction_ledger_sha256": ensemble_hash,
        "runtime_environment": _runtime_environment(),
        "elapsed_seconds_current_invocation": time.perf_counter() - started,
    }
    write_create_only(
        output_dir / "selection.json", canonical_json_bytes({"selection": selections})
    )
    write_create_only(output_dir / "summary.json", canonical_json_bytes(metadata))
    validate_dash_evidence(output_dir)
    return metadata


def run_dash_smoke(
    corpus: Corpus,
    split_manifest: dict[str, Any],
) -> dict[str, Any]:
    protocol, config = _require_registered_state()
    recipes = _recipes(config)
    output_path = RESEARCH_RESULTS / "gate1" / "dash_mi_smoke_v1.json"
    if output_path.exists():
        return read_json(output_path)
    _require_cuda_bf16()
    examples = add_therapist_vote_distributions(
        build_therapist_examples(corpus, int(protocol["data"]["context_turns"])),
        corpus.annotations,
    )
    examples["outer_fold"] = examples["source_id"].map(fold_lookup(split_manifest))
    outer_train = examples[examples["outer_fold"].ne(0)].reset_index(drop=True)
    splitter = StratifiedGroupKFold(
        n_splits=int(config["selection"]["inner_folds"]),
        shuffle=True,
        random_state=int(config["selection"]["seed"]),
    )
    train_indices, validation_indices = next(
        splitter.split(
            np.arange(len(outer_train)),
            outer_train["label"],
            groups=outer_train["source_id"],
        )
    )
    train = _stratified_cap(outer_train.iloc[train_indices], 64, seed=1907)
    validation = _stratified_cap(outer_train.iloc[validation_indices], 32, seed=1907)
    recipe = max(
        recipes, key=lambda value: (value.history_max_length, value.disagreement_mix)
    )
    started = time.perf_counter()
    outcome = _fit_once(
        train,
        validation,
        _tokenizer(config),
        config,
        recipe,
        seed=1907,
        fixed_epochs=1,
    )
    if outcome.predictions is None:
        raise AssertionError("DASH-MI smoke training did not produce predictions")
    payload = {
        "gate_id": "dash-mi-cuda-smoke-v1",
        "status": "pass",
        "engineering_gate_not_performance_result": True,
        "outer_test_partition_touched": False,
        "model": "dash_mi",
        "recipe": recipe.recipe_id,
        "code_commit": git_commit(ROOT),
        "config_sha256": sha256_file(DASH_CONFIG),
        "split_manifest_sha256": split_manifest["manifest_sha256"],
        "train_rows": len(train),
        "validation_rows": len(validation),
        "optimizer_steps": outcome.optimizer_steps,
        "probabilities_finite": bool(
            np.isfinite(outcome.predictions.probabilities).all()
            and np.isfinite(outcome.predictions.target_only_probabilities).all()
        ),
        "maximum_probability_sum_error": float(
            max(
                np.max(np.abs(outcome.predictions.probabilities.sum(axis=1) - 1.0)),
                np.max(
                    np.abs(
                        outcome.predictions.target_only_probabilities.sum(axis=1) - 1.0
                    )
                ),
            )
        ),
        "peak_memory_bytes": outcome.peak_memory_bytes,
        "elapsed_seconds": time.perf_counter() - started,
        "runtime_environment": _runtime_environment(),
    }
    write_create_only(output_path, canonical_json_bytes(payload))
    return payload


def validate_dash_evidence(output_dir: Path) -> None:
    validate_neural_evidence(output_dir)
    summary = read_json(output_dir / "summary.json")
    seeds = pd.read_csv(
        output_dir / "predictions_by_seed.csv", dtype={"source_id": str, "seed": int}
    )
    ensemble = pd.read_csv(
        output_dir / "predictions_seed_ensemble.csv", dtype={"source_id": str}
    )
    target_columns = [f"prob_target_only_{label}" for label in LABELS]
    for frame_name, frame in (("seeds", seeds), ("ensemble", ensemble)):
        values = frame[target_columns].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.allclose(
            values.sum(axis=1), 1.0, atol=1e-6
        ):
            raise ValueError(f"Invalid DASH-MI target-only probabilities in {frame_name}")
        diagnostics = frame[
            [
                "context_gate_mean",
                "context_attention_entropy",
                "context_attention_max",
                "context_residual_l2",
            ]
        ].to_numpy(dtype=float)
        if not np.isfinite(diagnostics).all():
            raise ValueError(f"Non-finite DASH-MI diagnostics in {frame_name}")
    for seed, group in seeds.groupby("seed", sort=True):
        _assert_metric_close(
            evaluate_predictions(_target_only_view(group.reset_index(drop=True))),
            summary["metrics"]["target_only_ablation_per_seed"][str(seed)],
            f"dash-target-only-seed-{seed}",
        )
    _assert_metric_close(
        evaluate_predictions(_target_only_view(ensemble)),
        summary["metrics"]["target_only_ablation_seed_ensemble"],
        "dash-target-only-ensemble",
    )
    keys = ["outer_fold", "transcript_id", "utterance_id", "source_id", "label"]
    rebuilt = (
        seeds.groupby(keys, sort=False)[[f"prob_{label}" for label in LABELS]]
        .mean()
        .reset_index()
        .sort_values(keys, kind="stable")
        .reset_index(drop=True)
    )
    recorded = ensemble.sort_values(keys, kind="stable").reset_index(drop=True)
    if not np.allclose(
        rebuilt[[f"prob_{label}" for label in LABELS]].to_numpy(dtype=float),
        recorded[[f"prob_{label}" for label in LABELS]].to_numpy(dtype=float),
        atol=1e-9,
    ):
        raise ValueError("DASH-MI ensemble does not reconstruct from seed probabilities")
