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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .constants import (
    ARTIFACTS,
    FULL_DATA,
    FULL_MANIFEST,
    LABELS,
    NEURAL_CONFIG,
    PROTOCOL,
    RESEARCH_RESULTS,
    ROOT,
    SIMPLE_DATA,
    SIMPLE_MANIFEST,
)
from .data import Corpus, build_therapist_examples
from .io import canonical_json_bytes, git_commit, read_json, sha256_file, write_create_only
from .metrics import evaluate_predictions, source_balanced_weights
from .splits import fold_lookup


@dataclass(frozen=True)
class NeuralRecipe:
    recipe_id: str
    learning_rate: float
    max_length: int


@dataclass(frozen=True)
class FitOutcome:
    best_score: float | None
    best_epoch: int
    epochs_completed: int
    optimizer_steps: int
    peak_memory_bytes: int
    probabilities: np.ndarray | None = None


def _git_is_clean() -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not completed.stdout.strip()


def _require_registered_state() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = read_json(PROTOCOL)
    config = read_json(NEURAL_CONFIG)
    if config["status"] != "registered_before_neural_model_evaluation":
        raise ValueError("Neural configuration is not prospectively registered")
    if config["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("Neural configuration and research protocol disagree")
    if config["selection"]["inner_folds"] != protocol["development"]["inner_folds"]:
        raise ValueError("Neural and protocol inner-fold counts disagree")
    if config["final_seeds"] != protocol["development"]["seeds"]:
        raise ValueError("Neural and protocol seed lists disagree")
    for data_path, manifest_path in (
        (SIMPLE_DATA, SIMPLE_MANIFEST),
        (FULL_DATA, FULL_MANIFEST),
    ):
        if sha256_file(data_path) != read_json(manifest_path)["sha256"]:
            raise ValueError(f"Dataset hash mismatch: {data_path}")
    if not _git_is_clean():
        raise RuntimeError("Commit tracked code/config changes before running neural evidence")
    return protocol, config


def _recipes(config: dict[str, Any], model_name: str) -> list[NeuralRecipe]:
    try:
        values = config["models"][model_name]["recipes"]
    except KeyError as exc:
        raise ValueError(f"Unknown neural model: {model_name}") from exc
    recipes = [NeuralRecipe(**value) for value in values]
    if len(recipes) != 4 or len({recipe.recipe_id for recipe in recipes}) != len(recipes):
        raise ValueError("Each neural model must have four uniquely named recipes")
    return recipes


def _runtime_environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "compute_capability": (
            list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
        ),
        "bf16_supported": (
            bool(torch.cuda.is_bf16_supported()) if torch.cuda.is_available() else False
        ),
    }


def _require_cuda_bf16() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("The registered neural run requires CUDA, but CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The registered neural run requires BF16 support")
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


def _training_weights(frame: pd.DataFrame) -> np.ndarray:
    source_weights = source_balanced_weights(frame["source_id"])
    counts = frame["label"].value_counts()
    class_weights = (
        frame["label"]
        .map(lambda label: len(frame) / (len(LABELS) * int(counts[label])))
        .to_numpy(dtype=float)
    )
    weights = source_weights * class_weights
    return weights / weights.mean()


def _tensor_dataset(
    frame: pd.DataFrame,
    tokenizer: Any,
    text_column: str,
    max_length: int,
    include_training_weights: bool,
) -> TensorDataset:
    encoded = tokenizer(
        frame[text_column].astype(str).tolist(),
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    label_ids = torch.tensor([LABELS.index(str(value)) for value in frame["label"]])
    weights = (
        _training_weights(frame) if include_training_weights else np.ones(len(frame), dtype=float)
    )
    return TensorDataset(
        encoded["input_ids"],
        encoded["attention_mask"],
        label_ids,
        torch.tensor(weights, dtype=torch.float32),
    )


def _loader(
    dataset: TensorDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )


def _new_model(config: dict[str, Any], device: torch.device) -> nn.Module:
    encoder = config["pretrained_encoder"]
    model = AutoModelForSequenceClassification.from_pretrained(
        encoder["model_id"],
        revision=encoder["revision"],
        cache_dir=ARTIFACTS / "huggingface",
        trust_remote_code=bool(encoder["trust_remote_code"]),
        num_labels=len(LABELS),
        id2label={index: label for index, label in enumerate(LABELS)},
        label2id={label: index for index, label in enumerate(LABELS)},
        ignore_mismatched_sizes=True,
    )
    model.config.problem_type = "single_label_classification"
    return model.to(device)


@torch.inference_mode()
def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    for input_ids, attention_mask, _, _ in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(
                input_ids=input_ids.to(device, non_blocking=True),
                attention_mask=attention_mask.to(device, non_blocking=True),
            ).logits
        outputs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    probabilities = np.concatenate(outputs, axis=0)
    if not np.isfinite(probabilities).all():
        raise FloatingPointError("Neural prediction contains a non-finite probability")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise FloatingPointError("Neural prediction probabilities do not sum to one")
    return probabilities


def _validation_score(frame: pd.DataFrame, probabilities: np.ndarray) -> float:
    predicted = np.asarray(LABELS, dtype=object)[probabilities.argmax(axis=1)]
    return float(
        f1_score(
            frame["label"],
            predicted,
            labels=list(LABELS),
            average="macro",
            sample_weight=source_balanced_weights(frame["source_id"]),
            zero_division=0,
        )
    )


def _schedule_multiplier(step: int, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    remaining = max(total_steps - step, 0)
    decay_steps = max(total_steps - warmup_steps, 1)
    return float(remaining) / float(decay_steps)


def _fit_once(
    train: pd.DataFrame,
    validation_or_test: pd.DataFrame,
    tokenizer: Any,
    config: dict[str, Any],
    model_name: str,
    recipe: NeuralRecipe,
    seed: int,
    fixed_epochs: int | None = None,
) -> FitOutcome:
    device = _require_cuda_bf16()
    _seed_everything(seed)
    settings = config["training"]
    model_settings = config["models"][model_name]
    train_data = _tensor_dataset(
        train,
        tokenizer,
        model_settings["text_column"],
        recipe.max_length,
        include_training_weights=True,
    )
    eval_data = _tensor_dataset(
        validation_or_test,
        tokenizer,
        model_settings["text_column"],
        recipe.max_length,
        include_training_weights=False,
    )
    train_loader = _loader(
        train_data,
        int(model_settings["train_batch_size"]),
        shuffle=True,
        seed=seed,
    )
    eval_loader = _loader(
        eval_data,
        int(model_settings["eval_batch_size"]),
        shuffle=False,
        seed=seed,
    )
    maximum_epochs = fixed_epochs or int(settings["maximum_epochs"])
    accumulation = int(settings["gradient_accumulation_steps"])
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_steps = updates_per_epoch * maximum_epochs
    warmup_steps = round(float(settings["warmup_fraction"]) * total_steps)

    model = _new_model(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=recipe.learning_rate,
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
            for batch_index, (input_ids, attention_mask, labels, weights) in enumerate(
                train_loader, start=1
            ):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(
                        input_ids=input_ids.to(device, non_blocking=True),
                        attention_mask=attention_mask.to(device, non_blocking=True),
                    ).logits
                    losses = F.cross_entropy(
                        logits.float(),
                        labels.to(device, non_blocking=True),
                        reduction="none",
                        label_smoothing=float(settings["label_smoothing"]),
                    )
                    loss = (losses * weights.to(device, non_blocking=True)).mean() / accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError("Training produced a non-finite loss")
                loss.backward()
                running_loss += float(loss.detach()) * accumulation
                should_step = batch_index % accumulation == 0 or batch_index == len(train_loader)
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
                    f"{model_name}/{recipe.recipe_id}/seed={seed}: "
                    f"epoch {epoch}/{maximum_epochs}, train_loss="
                    f"{running_loss / len(train_loader):.4f}",
                    flush=True,
                )
                continue

            probabilities = _predict(model, eval_loader, device)
            score = _validation_score(validation_or_test, probabilities)
            print(
                f"{model_name}/{recipe.recipe_id}/seed={seed}: epoch {epoch}, "
                f"inner_source_macro_f1={score:.4f}",
                flush=True,
            )
            if best_score is None or score > best_score + 1e-8:
                best_score = score
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
            if epoch >= int(settings["minimum_epochs"]) and stale_epochs >= int(
                settings["early_stopping_patience"]
            ):
                break

        final_probabilities = (
            _predict(model, eval_loader, device) if fixed_epochs is not None else None
        )
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    finally:
        del model
        del optimizer
        del scheduler
        gc.collect()
        torch.cuda.empty_cache()

    return FitOutcome(
        best_score=best_score,
        best_epoch=best_epoch,
        epochs_completed=epochs_completed,
        optimizer_steps=optimizer_steps,
        peak_memory_bytes=peak_memory,
        probabilities=final_probabilities,
    )


def _ledger_rows(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    model_name: str,
    fold: int,
    seed: int,
    recipe: NeuralRecipe,
    epochs: int,
    train_texts: set[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    predicted = np.asarray(LABELS, dtype=object)[probabilities.argmax(axis=1)]
    ledger = frame[["transcript_id", "utterance_id", "source_id", "label"]].copy()
    ledger.insert(0, "model", model_name)
    ledger.insert(1, "seed", seed)
    ledger.insert(2, "outer_fold", fold)
    ledger["prediction"] = predicted
    for index, label in enumerate(LABELS):
        ledger[f"prob_{label}"] = probabilities[:, index]
    ledger["seen_text_in_outer_train"] = frame["normalized_text"].isin(train_texts).to_numpy()
    ledger["normalized_text_sha256"] = frame["normalized_text"].map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    ledger["selected_recipe"] = recipe.recipe_id
    ledger["max_length"] = recipe.max_length
    ledger["epochs"] = epochs
    ledger["pretrained_revision"] = config["pretrained_encoder"]["revision"]
    return ledger


def _tokenizer(config: dict[str, Any]) -> Any:
    encoder = config["pretrained_encoder"]
    return AutoTokenizer.from_pretrained(
        encoder["model_id"],
        revision=encoder["revision"],
        cache_dir=ARTIFACTS / "huggingface",
        trust_remote_code=bool(encoder["trust_remote_code"]),
        use_fast=True,
    )


def _round_median_epoch(epochs: list[int]) -> int:
    return max(1, math.floor(float(np.median(epochs)) + 0.5))


def _fold_cache_dir(model_name: str, commit: str, config_sha256: str) -> Path:
    return ARTIFACTS / "neural_v1" / f"{model_name}_{commit[:12]}_{config_sha256[:12]}"


def _load_cached_fold(
    directory: Path,
    fold: int,
    expected: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    ledger_path = directory / f"fold_{fold}_predictions.csv"
    metadata_path = directory / f"fold_{fold}_metadata.json"
    if not ledger_path.exists() or not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Cached neural fold has incompatible {key}: {metadata_path}")
    if metadata["prediction_sha256"] != sha256_file(ledger_path):
        raise ValueError(f"Cached neural fold has a ledger hash mismatch: {ledger_path}")
    return pd.read_csv(ledger_path, dtype={"source_id": str}), metadata


def _write_cached_fold(
    directory: Path,
    fold: int,
    ledger: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    buffer = io.StringIO()
    ledger.to_csv(buffer, index=False, lineterminator="\n", float_format="%.10g")
    payload = buffer.getvalue().encode("utf-8")
    metadata["prediction_sha256"] = write_create_only(
        directory / f"fold_{fold}_predictions.csv", payload
    )
    write_create_only(directory / f"fold_{fold}_metadata.json", canonical_json_bytes(metadata))


def _run_outer_fold(
    examples: pd.DataFrame,
    fold: int,
    model_name: str,
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
    for recipe in _recipes(config, model_name):
        inner_records: list[dict[str, Any]] = []
        for inner_fold, (inner_train_indices, validation_indices) in enumerate(inner_splits):
            outcome = _fit_once(
                train.iloc[inner_train_indices].reset_index(drop=True),
                train.iloc[validation_indices].reset_index(drop=True),
                tokenizer,
                config,
                model_name,
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
        for recipe in _recipes(config, model_name)
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
            model_name,
            selected_recipe,
            int(seed),
            fixed_epochs=selected_epochs,
        )
        if outcome.probabilities is None:
            raise AssertionError("Final neural training did not produce probabilities")
        ledgers.append(
            _ledger_rows(
                test,
                outcome.probabilities,
                model_name,
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
        "selected_recipe": selected_recipe.recipe_id,
        "selected_epochs": selected_epochs,
        "candidate_records": candidate_records,
        "final_seed_records": seed_records,
        "outer_test_used_for_selection": False,
        "protocol_id": protocol["protocol_id"],
    }


def _ensemble_ledger(ledger: pd.DataFrame, model_name: str) -> pd.DataFrame:
    keys = ["outer_fold", "transcript_id", "utterance_id", "source_id", "label"]
    probability_columns = [f"prob_{label}" for label in LABELS]
    invariant_columns = [
        "seen_text_in_outer_train",
        "normalized_text_sha256",
        "selected_recipe",
        "max_length",
        "epochs",
        "pretrained_revision",
    ]
    grouped = ledger.groupby(keys, sort=False, dropna=False)
    for column in invariant_columns:
        if grouped[column].nunique(dropna=False).max() != 1:
            raise ValueError(f"Seed ledgers disagree on invariant column {column}")
    ensemble = grouped[probability_columns].mean().reset_index()
    invariants = grouped[invariant_columns].first().reset_index()
    ensemble = ensemble.merge(invariants, on=keys, validate="one_to_one")
    ensemble.insert(0, "model", f"{model_name}_seed_ensemble")
    ensemble.insert(1, "seed", "mean")
    probabilities = ensemble[probability_columns].to_numpy(dtype=float)
    ensemble["prediction"] = np.asarray(LABELS, dtype=object)[probabilities.argmax(axis=1)]
    return ensemble


def run_neural(
    corpus: Corpus,
    split_manifest: dict[str, Any],
    model_name: str,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    protocol, config = _require_registered_state()
    _recipes(config, model_name)
    output_dir = output_dir or RESEARCH_RESULTS / "neural_v1" / model_name
    if (output_dir / "summary.json").exists():
        validate_neural_evidence(output_dir)
        return read_json(output_dir / "summary.json")

    device = _require_cuda_bf16()
    del device
    commit = git_commit(ROOT)
    config_sha256 = sha256_file(NEURAL_CONFIG)
    expected_cache = {
        "model": model_name,
        "code_commit": commit,
        "config_sha256": config_sha256,
        "split_manifest_sha256": split_manifest["manifest_sha256"],
    }
    examples = build_therapist_examples(
        corpus, context_turns=int(protocol["data"]["context_turns"])
    )
    examples["outer_fold"] = examples["source_id"].map(fold_lookup(split_manifest))
    if examples["outer_fold"].isna().any():
        raise ValueError("At least one source lacks an outer-fold assignment")

    tokenizer = _tokenizer(config)
    cache_dir = _fold_cache_dir(model_name, commit, config_sha256)
    fold_ledgers: list[pd.DataFrame] = []
    selections: list[dict[str, Any]] = []
    started = time.perf_counter()
    for fold in range(int(split_manifest["n_splits"])):
        cached = _load_cached_fold(cache_dir, fold, expected_cache)
        if cached is None:
            print(f"Starting {model_name} outer fold {fold}", flush=True)
            fold_started = time.perf_counter()
            fold_ledger, selection = _run_outer_fold(
                examples, fold, model_name, protocol, config, tokenizer
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
            print(f"Loaded completed {model_name} outer fold {fold} from cache", flush=True)
        fold_ledgers.append(fold_ledger)
        selections.append(selection)

    ledger = pd.concat(fold_ledgers, ignore_index=True).sort_values(
        ["seed", "outer_fold", "transcript_id", "utterance_id"], kind="stable"
    )
    ledger = ledger.reset_index(drop=True)
    expected_rows = len(examples)
    if ledger.duplicated(["seed", "transcript_id", "utterance_id"]).any():
        raise ValueError("Duplicate out-of-fold neural prediction")
    counts = ledger.groupby("seed").size()
    if len(counts) != len(config["final_seeds"]) or not counts.eq(expected_rows).all():
        raise ValueError(f"Incomplete neural out-of-fold coverage: {counts.to_dict()}")

    ensemble = _ensemble_ledger(ledger, model_name).sort_values(
        ["outer_fold", "transcript_id", "utterance_id"], kind="stable"
    )
    ensemble = ensemble.reset_index(drop=True)
    per_seed_metrics = {
        str(seed): evaluate_predictions(group.reset_index(drop=True))
        for seed, group in ledger.groupby("seed", sort=True)
    }
    ensemble_metrics = evaluate_predictions(ensemble)

    ledger_buffer = io.StringIO()
    ledger.to_csv(ledger_buffer, index=False, lineterminator="\n", float_format="%.10g")
    ensemble_buffer = io.StringIO()
    ensemble.to_csv(ensemble_buffer, index=False, lineterminator="\n", float_format="%.10g")
    ledger_hash = write_create_only(
        output_dir / "predictions_by_seed.csv", ledger_buffer.getvalue().encode("utf-8")
    )
    ensemble_hash = write_create_only(
        output_dir / "predictions_seed_ensemble.csv",
        ensemble_buffer.getvalue().encode("utf-8"),
    )
    metadata = {
        "result_id": f"annomi-{model_name}-source-cv-neural-v1",
        "protocol_id": protocol["protocol_id"],
        "model": model_name,
        "pretrained_encoder": config["pretrained_encoder"],
        "code_commit": commit,
        "config_sha256": config_sha256,
        "split_manifest_sha256": split_manifest["manifest_sha256"],
        "dataset_sha256": {
            "simple": read_json(ROOT / protocol["data"]["simple_manifest"])["sha256"],
            "full": read_json(ROOT / protocol["data"]["full_manifest"])["sha256"],
        },
        "seeds": config["final_seeds"],
        "selection": selections,
        "metrics": {
            "seed_ensemble": ensemble_metrics,
            "per_seed": per_seed_metrics,
        },
        "prediction_ledger_sha256": ledger_hash,
        "ensemble_prediction_ledger_sha256": ensemble_hash,
        "runtime_environment": _runtime_environment(),
        "elapsed_seconds_current_invocation": time.perf_counter() - started,
    }
    write_create_only(
        output_dir / "selection.json", canonical_json_bytes({"selection": selections})
    )
    write_create_only(output_dir / "summary.json", canonical_json_bytes(metadata))
    validate_neural_evidence(output_dir)
    return metadata


def run_environment_gate() -> dict[str, Any]:
    protocol, config = _require_registered_state()
    device = _require_cuda_bf16()
    output_path = RESEARCH_RESULTS / "gate1" / "gpu_environment_v1.json"
    if output_path.exists():
        return read_json(output_path)
    torch.cuda.reset_peak_memory_stats(device)
    left = torch.randn((1024, 1024), device=device, dtype=torch.bfloat16)
    right = torch.randn((1024, 1024), device=device, dtype=torch.bfloat16)
    product = left @ right
    torch.cuda.synchronize(device)
    driver = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload = {
        "gate_id": "annomi-neural-gpu-environment-v1",
        "status": "pass",
        "protocol_id": protocol["protocol_id"],
        "code_commit": git_commit(ROOT),
        "config_sha256": sha256_file(NEURAL_CONFIG),
        "uv_lock_sha256": sha256_file(ROOT / "uv.lock"),
        "runtime_environment": _runtime_environment(),
        "nvidia_smi_name_driver_memory_mib": driver,
        "bf16_matrix_multiply_shape": [1024, 1024],
        "bf16_matrix_multiply_finite": bool(torch.isfinite(product).all().item()),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "performance_result_produced": False,
        "pretrained_encoder": config["pretrained_encoder"],
    }
    del left, right, product
    torch.cuda.empty_cache()
    if not payload["bf16_matrix_multiply_finite"]:
        raise FloatingPointError("GPU environment gate produced non-finite BF16 output")
    write_create_only(output_path, canonical_json_bytes(payload))
    return payload


def _stratified_cap(frame: pd.DataFrame, rows_per_label: int, seed: int) -> pd.DataFrame:
    groups = []
    for _, group in frame.groupby("label", sort=True):
        groups.append(group.sample(n=min(rows_per_label, len(group)), random_state=seed))
    return (
        pd.concat(groups, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def run_neural_smoke(
    corpus: Corpus,
    split_manifest: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    protocol, config = _require_registered_state()
    recipes = _recipes(config, model_name)
    output_path = RESEARCH_RESULTS / "gate1" / f"{model_name}_smoke_v1.json"
    if output_path.exists():
        return read_json(output_path)
    _require_cuda_bf16()
    examples = build_therapist_examples(
        corpus, context_turns=int(protocol["data"]["context_turns"])
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
    recipe = max(recipes, key=lambda value: value.max_length)
    started = time.perf_counter()
    outcome = _fit_once(
        train,
        validation,
        _tokenizer(config),
        config,
        model_name,
        recipe,
        seed=1907,
        fixed_epochs=1,
    )
    if outcome.probabilities is None:
        raise AssertionError("Smoke training did not produce probabilities")
    payload = {
        "gate_id": f"{model_name}-cuda-smoke-v1",
        "status": "pass",
        "engineering_gate_not_performance_result": True,
        "outer_test_partition_touched": False,
        "model": model_name,
        "recipe": recipe.recipe_id,
        "code_commit": git_commit(ROOT),
        "config_sha256": sha256_file(NEURAL_CONFIG),
        "split_manifest_sha256": split_manifest["manifest_sha256"],
        "train_rows": len(train),
        "validation_rows": len(validation),
        "optimizer_steps": outcome.optimizer_steps,
        "probabilities_finite": bool(np.isfinite(outcome.probabilities).all()),
        "maximum_probability_sum_error": float(
            np.max(np.abs(outcome.probabilities.sum(axis=1) - 1.0))
        ),
        "peak_memory_bytes": outcome.peak_memory_bytes,
        "elapsed_seconds": time.perf_counter() - started,
        "runtime_environment": _runtime_environment(),
    }
    write_create_only(output_path, canonical_json_bytes(payload))
    return payload


def _assert_metric_close(rebuilt: dict[str, Any], recorded: dict[str, Any], context: str) -> None:
    for metric in (
        "utterance_macro_f1",
        "source_balanced_macro_f1",
        "source_balanced_brier",
        "source_balanced_log_loss",
    ):
        if not np.isclose(rebuilt[metric], recorded[metric], atol=1e-10):
            raise ValueError(f"Neural reconstruction mismatch for {context}/{metric}")


def validate_neural_evidence(output_dir: Path) -> None:
    summary = read_json(output_dir / "summary.json")
    seed_path = output_dir / "predictions_by_seed.csv"
    ensemble_path = output_dir / "predictions_seed_ensemble.csv"
    if sha256_file(seed_path) != summary["prediction_ledger_sha256"]:
        raise ValueError("Neural per-seed prediction-ledger hash mismatch")
    if sha256_file(ensemble_path) != summary["ensemble_prediction_ledger_sha256"]:
        raise ValueError("Neural ensemble prediction-ledger hash mismatch")
    seeds = pd.read_csv(seed_path, dtype={"source_id": str, "seed": int})
    ensemble = pd.read_csv(ensemble_path, dtype={"source_id": str})
    if seeds.duplicated(["seed", "transcript_id", "utterance_id"]).any():
        raise ValueError("Neural per-seed ledger contains duplicates")
    if ensemble.duplicated(["transcript_id", "utterance_id"]).any():
        raise ValueError("Neural ensemble ledger contains duplicates")
    for seed, group in seeds.groupby("seed", sort=True):
        _assert_metric_close(
            evaluate_predictions(group.reset_index(drop=True)),
            summary["metrics"]["per_seed"][str(seed)],
            f"seed-{seed}",
        )
    _assert_metric_close(
        evaluate_predictions(ensemble.reset_index(drop=True)),
        summary["metrics"]["seed_ensemble"],
        "seed-ensemble",
    )
