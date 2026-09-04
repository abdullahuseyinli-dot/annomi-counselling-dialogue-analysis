from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from .ac_data import SessionTurns
from .constants import ARTIFACTS, CLIENT_LABELS, LABELS, SIMPLE_DATA
from .data import Corpus
from .io import canonical_json_bytes, sha256_file
from .qtrace import _require_device, _seed_everything


class _UtteranceDataset(Dataset[dict[str, Any]]):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.reset_index(drop=True)
        source_counts = self.frame["source_id"].astype(str).value_counts()
        raw = self.frame["source_id"].astype(str).map(lambda value: 1.0 / source_counts[value])
        self.weights = (raw / raw.mean()).to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        return {
            "text": str(row["utterance_text"]),
            "role": int(row["role"]),
            "target": int(row["target"]),
            "weight": float(self.weights[index]),
        }


class _AdaptedTurnEncoder(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        encoder = config["pretrained_encoder"]
        adapter = config["turn_adapter"]
        base = AutoModel.from_pretrained(
            encoder["model_id"],
            revision=encoder["revision"],
            cache_dir=ARTIFACTS / "huggingface",
            trust_remote_code=bool(encoder["trust_remote_code"]),
        )
        layers = list(adapter.get("layers_to_transform", [])) or None
        lora = LoraConfig(
            r=int(adapter["rank"]),
            lora_alpha=int(adapter["alpha"]),
            lora_dropout=float(adapter["dropout"]),
            target_modules=list(adapter["target_modules"]),
            layers_to_transform=layers,
            layers_pattern=str(adapter["layers_pattern"]) if layers else None,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.encoder = get_peft_model(base, lora)
        hidden_size = int(base.config.hidden_size)
        self.therapist_head = nn.Linear(hidden_size, len(LABELS))
        self.client_head = nn.Linear(hidden_size, len(CLIENT_LABELS))

    @staticmethod
    def _pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask[..., None]
        return (hidden.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        pooled = self._pool(hidden, attention_mask)
        return pooled, self.therapist_head(pooled), self.client_head(pooled)


def _training_frame(corpus: Corpus, sessions: list[SessionTurns]) -> pd.DataFrame:
    transcript_ids = {session.transcript_id for session in sessions}
    frame = corpus.utterances[
        corpus.utterances["transcript_id"].isin(transcript_ids)
    ].copy()
    frame["role"] = frame["interlocutor"].map({"client": 0, "therapist": 1})
    therapist_targets = frame["main_therapist_behaviour"].map(
        {label: index for index, label in enumerate(LABELS)}
    )
    client_targets = frame["client_talk_type"].map(
        {label: index for index, label in enumerate(CLIENT_LABELS)}
    )
    frame["target"] = np.where(frame["role"].eq(1), therapist_targets, client_targets)
    if frame[["role", "target"]].isna().any().any():
        raise ValueError("An adapter-training utterance has an invalid role or label")
    frame["role"] = frame["role"].astype(int)
    frame["target"] = frame["target"].astype(int)
    return frame[
        ["transcript_id", "utterance_id", "source_id", "utterance_text", "role", "target"]
    ].reset_index(drop=True)


def _adapter_cache_paths(
    config: dict[str, Any],
    sessions: list[SessionTurns],
    fold: int,
    seed: int,
    phase: str,
) -> tuple[Path, Path]:
    identity = hashlib.sha256(
        canonical_json_bytes(
            {
                "dataset_sha256": sha256_file(SIMPLE_DATA),
                "encoder": config["pretrained_encoder"],
                "adapter": config["turn_adapter"],
                "sources": sorted({session.source_id for session in sessions}),
                "fold": fold,
                "seed": seed,
                "phase": phase,
            }
        )
    ).hexdigest()[:24]
    root = ARTIFACTS / "safe_mi_v2" / "adapted_turn_embeddings"
    return root / f"{identity}.npz", root / f"{identity}.json"


def _load_cached_embeddings(
    cache_path: Path,
    metadata_path: Path,
) -> dict[tuple[int, int], np.ndarray] | None:
    if not cache_path.exists() or not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("npz_sha256") != sha256_file(cache_path):
        raise ValueError(f"Adapted embedding cache hash mismatch: {cache_path}")
    with np.load(cache_path) as values:
        embeddings = values["embeddings"].astype(np.float32)
        transcript_ids = values["transcript_ids"].astype(int)
        utterance_ids = values["utterance_ids"].astype(int)
    return {
        (int(transcript_id), int(utterance_id)): embeddings[index]
        for index, (transcript_id, utterance_id) in enumerate(
            zip(transcript_ids, utterance_ids, strict=True)
        )
    }


def _class_weights(frame: pd.DataFrame, role: int, classes: int, device: torch.device) -> torch.Tensor:
    values = frame.loc[frame["role"].eq(role), "target"].to_numpy(dtype=int)
    counts = np.bincount(values, minlength=classes).astype(float)
    if (counts == 0).any():
        raise ValueError("Adapter fitting partition is missing a behaviour class")
    return torch.tensor(len(values) / (classes * counts), dtype=torch.float32, device=device)


def train_and_extract_adapted_embeddings(
    corpus: Corpus,
    sessions: list[SessionTurns],
    config: dict[str, Any],
    fold: int,
    seed: int,
    phase: str,
) -> dict[tuple[int, int], np.ndarray]:
    """Fit a source-restricted LoRA behaviour coder and extract every turn.

    Labels from ``sessions`` are used only during fitting.  Held-source labels are
    never passed to the adapter, and the returned mapping contains representations
    rather than predicted gold labels.
    """

    cache_path, metadata_path = _adapter_cache_paths(
        config, sessions, fold, seed, phase
    )
    cached = _load_cached_embeddings(cache_path, metadata_path)
    if cached is not None:
        print(
            f"SAFE-MI adapted embedding cache hit: fold={fold}/seed={seed}/{phase}",
            flush=True,
        )
        return cached

    device = _require_device()
    _seed_everything(seed + 10_000)
    encoder_config = config["pretrained_encoder"]
    adapter_config = config["turn_adapter"]
    tokenizer = AutoTokenizer.from_pretrained(
        encoder_config["model_id"],
        revision=encoder_config["revision"],
        cache_dir=ARTIFACTS / "huggingface",
        trust_remote_code=bool(encoder_config["trust_remote_code"]),
        use_fast=True,
    )
    training_frame = _training_frame(corpus, sessions)

    def collate(items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = tokenizer(
            [item["text"] for item in items],
            max_length=int(encoder_config["max_length"]),
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        encoded["roles"] = torch.tensor([item["role"] for item in items], dtype=torch.long)
        encoded["targets"] = torch.tensor(
            [item["target"] for item in items], dtype=torch.long
        )
        encoded["weights"] = torch.tensor(
            [item["weight"] for item in items], dtype=torch.float32
        )
        return encoded

    loader = DataLoader(
        _UtteranceDataset(training_frame),
        batch_size=int(adapter_config["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 10_000),
        num_workers=0,
        pin_memory=True,
        collate_fn=collate,
    )
    model = _AdaptedTurnEncoder(config).to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_parameter_count = int(
        sum(parameter.numel() for parameter in trainable_parameters)
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(adapter_config["learning_rate"]),
        weight_decay=float(adapter_config["weight_decay"]),
    )
    therapist_weights = _class_weights(training_frame, 1, len(LABELS), device)
    client_weights = _class_weights(training_frame, 0, len(CLIENT_LABELS), device)
    epoch_losses: list[float] = []
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for epoch in range(1, int(adapter_config["epochs"]) + 1):
            model.train()
            losses: list[float] = []
            for batch in loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                roles = batch["roles"].to(device, non_blocking=True)
                targets = batch["targets"].to(device, non_blocking=True)
                weights = batch["weights"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, therapist_logits, client_logits = model(input_ids, attention_mask)
                    row_losses = torch.zeros(len(roles), device=device)
                    therapist_mask = roles.eq(1)
                    client_mask = roles.eq(0)
                    row_losses[therapist_mask] = F.cross_entropy(
                        therapist_logits[therapist_mask],
                        targets[therapist_mask],
                        weight=therapist_weights,
                        reduction="none",
                    )
                    row_losses[client_mask] = F.cross_entropy(
                        client_logits[client_mask],
                        targets[client_mask],
                        weight=client_weights,
                        reduction="none",
                    )
                    loss = (row_losses * weights).sum() / weights.sum()
                if not torch.isfinite(loss):
                    raise FloatingPointError("SAFE-MI adapter produced a non-finite loss")
                loss.backward()
                nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    float(adapter_config["maximum_gradient_norm"]),
                )
                optimizer.step()
                losses.append(float(loss.detach()))
            mean_loss = float(np.mean(losses))
            epoch_losses.append(mean_loss)
            print(
                f"SAFE-MI adapter fold={fold}/seed={seed}/{phase}: "
                f"epoch={epoch}, loss={mean_loss:.4f}",
                flush=True,
            )

        extraction = corpus.utterances.sort_values(
            ["transcript_id", "utterance_id"], kind="stable"
        ).reset_index(drop=True)
        outputs: list[np.ndarray] = []
        model.eval()
        extraction_batch_size = int(adapter_config["extraction_batch_size"])
        for start in range(0, len(extraction), extraction_batch_size):
            texts = extraction["utterance_text"].iloc[
                start : start + extraction_batch_size
            ].astype(str).tolist()
            encoded = tokenizer(
                texts,
                max_length=int(encoder_config["max_length"]),
                padding=True,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                pooled, _, _ = model(
                    encoded["input_ids"].to(device, non_blocking=True),
                    encoded["attention_mask"].to(device, non_blocking=True),
                )
            outputs.append(pooled.cpu().numpy().astype(np.float16))
        matrix = np.concatenate(outputs, axis=0)
        peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
    finally:
        del model
        del optimizer
        del trainable_parameters
        gc.collect()
        torch.cuda.empty_cache()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            embeddings=matrix,
            transcript_ids=extraction["transcript_id"].to_numpy(dtype=np.int64),
            utterance_ids=extraction["utterance_id"].to_numpy(dtype=np.int64),
        )
    metadata_path.write_bytes(
        canonical_json_bytes(
            {
                "dataset_sha256": sha256_file(SIMPLE_DATA),
                "fold": fold,
                "seed": seed,
                "phase": phase,
                "training_sources": len({session.source_id for session in sessions}),
                "training_transcripts": len(sessions),
                "training_utterances": len(training_frame),
                "epoch_losses": epoch_losses,
                "trainable_parameters": trainable_parameter_count,
                "peak_memory_bytes": peak_memory_bytes,
                "encoder": encoder_config,
                "adapter": adapter_config,
                "rows": len(matrix),
                "dimensions": int(matrix.shape[1]),
                "npz_sha256": sha256_file(cache_path),
            }
        )
    )
    return {
        (int(row.transcript_id), int(row.utterance_id)): matrix[index].astype(np.float32)
        for index, row in enumerate(extraction.itertuples(index=False))
    }
