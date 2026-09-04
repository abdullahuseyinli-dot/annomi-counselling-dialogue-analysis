from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .constants import CLIENT_LABELS, LABELS


@dataclass(frozen=True)
class QTraceMode:
    name: str
    task_a_loss: bool
    task_c_loss: bool
    action_likelihood_quality_evidence: bool
    transition_regularization: bool


def mode_from_config(config: dict[str, Any], name: str) -> QTraceMode:
    for value in config["models"]:
        if value["model"] == name:
            return QTraceMode(
                name=str(value["model"]),
                task_a_loss=bool(value["task_a_loss"]),
                task_c_loss=bool(value["task_c_loss"]),
                action_likelihood_quality_evidence=bool(
                    value["action_likelihood_quality_evidence"]
                ),
                transition_regularization=bool(value["transition_regularization"]),
            )
    raise ValueError(f"Unknown Q-TRACE model mode: {name}")


class QTraceModel(nn.Module):
    """Causal session model with soft-state and quality-conditioned policy heads."""

    def __init__(
        self,
        embedding_size: int,
        transition_probabilities: np.ndarray,
        low_quality_prior: float,
        architecture: dict[str, Any],
        mode: QTraceMode,
    ) -> None:
        super().__init__()
        hidden_size = int(architecture["hidden_size"])
        role_size = int(architecture["role_embedding_size"])
        dropout = float(architecture["dropout"])
        self.mode = mode
        self.evidence_bound = float(architecture["quality_text_evidence_bound"])
        self.role_embedding = nn.Embedding(2, role_size)
        self.therapist_auxiliary = nn.Linear(embedding_size, len(LABELS))
        self.client_auxiliary = nn.Linear(embedding_size, len(CLIENT_LABELS))
        state_input_size = embedding_size + role_size + len(LABELS) + len(CLIENT_LABELS)
        self.input_projection = nn.Sequential(
            nn.Linear(state_input_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=int(architecture["gru_layers"]),
            batch_first=True,
            dropout=dropout if int(architecture["gru_layers"]) > 1 else 0.0,
        )
        self.state_dropout = nn.Dropout(dropout)
        self.unconditional_action = nn.Linear(hidden_size, len(LABELS))
        self.quality_action = nn.Linear(hidden_size, 2 * len(LABELS))
        self.transition_gate = nn.Linear(hidden_size, 2)
        self.text_quality_evidence = nn.Linear(hidden_size, 1)
        transitions = torch.as_tensor(transition_probabilities, dtype=torch.float32)
        if transitions.shape != (2, len(LABELS), len(CLIENT_LABELS), len(LABELS)):
            raise ValueError("Unexpected quality transition tensor shape")
        if not torch.allclose(transitions.sum(dim=-1), torch.ones_like(transitions[..., 0])):
            raise ValueError("Transition distributions do not sum to one")
        self.register_buffer("transition_probabilities", transitions)
        low_quality_prior = float(np.clip(low_quality_prior, 1e-6, 1 - 1e-6))
        self.register_buffer(
            "quality_prior_log_odds",
            torch.tensor(math.log(low_quality_prior / (1.0 - low_quality_prior))),
        )

    @staticmethod
    def _last_therapist_distribution(
        therapist_probabilities: torch.Tensor,
        roles: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, classes = therapist_probabilities.shape
        last = torch.full(
            (batch_size, classes),
            1.0 / classes,
            dtype=therapist_probabilities.dtype,
            device=therapist_probabilities.device,
        )
        history: list[torch.Tensor] = []
        for position in range(sequence_length):
            history.append(last)
            update = roles[:, position].eq(1) & valid_mask[:, position]
            last = torch.where(update[:, None], therapist_probabilities[:, position], last)
        return torch.stack(history, dim=1)

    def forward(
        self,
        embeddings: torch.Tensor,
        roles: torch.Tensor,
        lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, sequence_length, _ = embeddings.shape
        positions = torch.arange(sequence_length, device=embeddings.device)[None, :]
        valid_mask = positions < lengths[:, None]
        therapist_logits = self.therapist_auxiliary(embeddings.float())
        client_logits = self.client_auxiliary(embeddings.float())
        therapist_probabilities = torch.softmax(therapist_logits, dim=-1)
        client_probabilities = torch.softmax(client_logits, dim=-1)
        therapist_state = therapist_probabilities * roles.eq(1)[..., None]
        client_state = client_probabilities * roles.eq(0)[..., None]
        model_input = torch.cat(
            [
                embeddings,
                self.role_embedding(roles.clamp(0, 1)),
                therapist_state.to(embeddings.dtype),
                client_state.to(embeddings.dtype),
            ],
            dim=-1,
        )
        projected = self.input_projection(model_input)
        packed = pack_padded_sequence(
            projected,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_states, _ = self.gru(packed)
        states, _ = pad_packed_sequence(
            packed_states,
            batch_first=True,
            total_length=sequence_length,
        )
        states = self.state_dropout(states)

        unconditional_probabilities = torch.softmax(self.unconditional_action(states).float(), dim=-1)
        quality_logits = self.quality_action(states).reshape(
            batch_size, sequence_length, 2, len(LABELS)
        )
        last_therapist = self._last_therapist_distribution(
            therapist_probabilities, roles, valid_mask
        )
        transition_prior = torch.einsum(
            "btp,btc,qpca->btqa",
            last_therapist,
            client_probabilities,
            self.transition_probabilities,
        ).clamp_min(1e-8)
        transition_gate = torch.sigmoid(self.transition_gate(states).float())
        if self.mode.transition_regularization:
            quality_logits = quality_logits.float() + transition_gate[..., None] * torch.log(
                transition_prior
            )
        quality_action_probabilities = torch.softmax(quality_logits.float(), dim=-1)

        text_evidence = self.evidence_bound * torch.tanh(
            self.text_quality_evidence(states).squeeze(-1).float()
        )
        text_evidence = text_evidence * roles.eq(1) * valid_mask
        previous_policy = torch.roll(quality_action_probabilities, shifts=1, dims=1)
        previous_policy[:, 0] = 1.0 / len(LABELS)
        action_log_likelihood_ratio = (
            therapist_probabilities
            * (
                torch.log(previous_policy[:, :, 1].clamp_min(1e-8))
                - torch.log(previous_policy[:, :, 0].clamp_min(1e-8))
            )
        ).sum(dim=-1)
        preceded_by_client = torch.zeros_like(valid_mask)
        preceded_by_client[:, 1:] = roles[:, :-1].eq(0) & valid_mask[:, :-1]
        action_log_likelihood_ratio *= roles.eq(1) & preceded_by_client & valid_mask
        if not self.mode.action_likelihood_quality_evidence:
            action_log_likelihood_ratio = torch.zeros_like(action_log_likelihood_ratio)
        evidence = text_evidence + action_log_likelihood_ratio
        quality_log_odds = torch.clamp(
            self.quality_prior_log_odds + torch.cumsum(evidence, dim=1), -12.0, 12.0
        )
        low_probabilities = torch.sigmoid(quality_log_odds)
        online_quality_probabilities = torch.stack(
            [1.0 - low_probabilities, low_probabilities], dim=-1
        )
        if self.mode.name == "c_only":
            action_probabilities = unconditional_probabilities
        else:
            action_probabilities = (
                online_quality_probabilities[..., None] * quality_action_probabilities
            ).sum(dim=2)

        return {
            "therapist_auxiliary_logits": therapist_logits,
            "client_auxiliary_logits": client_logits,
            "unconditional_action_probabilities": unconditional_probabilities,
            "quality_action_probabilities": quality_action_probabilities,
            "action_probabilities": action_probabilities,
            "online_quality_probabilities": online_quality_probabilities,
            "transition_prior": transition_prior,
            "transition_gate": transition_gate,
            "text_quality_evidence": text_evidence,
            "action_quality_evidence": action_log_likelihood_ratio,
            "valid_mask": valid_mask,
        }


def _weighted_session_average(
    row_losses: torch.Tensor,
    mask: torch.Tensor,
    session_weights: torch.Tensor,
) -> torch.Tensor:
    counts = mask.sum(dim=1)
    active = counts.gt(0)
    if not active.any():
        return row_losses.sum() * 0.0
    per_session = (row_losses * mask).sum(dim=1) / counts.clamp_min(1)
    weights = session_weights[active]
    return (per_session[active] * weights).sum() / weights.sum()


def qtrace_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    mode: QTraceMode,
    training: dict[str, Any],
    class_weights: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    session_weights = batch["session_weights"]
    valid = output["valid_mask"]
    therapist_mask = batch["therapist_labels"].ge(0) & valid
    client_mask = batch["client_labels"].ge(0) & valid
    action_mask = batch["next_action_targets"].ge(0) & valid
    quality_mask = batch["quality_mask"] & valid

    therapist_losses = F.cross_entropy(
        output["therapist_auxiliary_logits"].transpose(1, 2),
        batch["therapist_labels"],
        weight=class_weights["therapist"],
        ignore_index=-100,
        reduction="none",
    )
    client_losses = F.cross_entropy(
        output["client_auxiliary_logits"].transpose(1, 2),
        batch["client_labels"],
        weight=class_weights["client"],
        ignore_index=-100,
        reduction="none",
    )
    auxiliary = 0.5 * (
        _weighted_session_average(therapist_losses, therapist_mask, session_weights)
        + _weighted_session_average(client_losses, client_mask, session_weights)
    )

    quality_targets = batch["quality"][:, None].expand_as(batch["next_action_targets"])
    selected_quality = output["online_quality_probabilities"].gather(
        -1, quality_targets.clamp_min(0)[..., None]
    ).squeeze(-1)
    quality_losses = -torch.log(selected_quality.clamp_min(1e-8)) * class_weights[
        "quality"
    ][quality_targets]
    quality = _weighted_session_average(
        quality_losses, quality_mask, session_weights
    )

    action_targets = batch["next_action_targets"].clamp_min(0)
    selected_action = output["action_probabilities"].gather(
        -1, action_targets[..., None]
    ).squeeze(-1)
    action_losses = -torch.log(selected_action.clamp_min(1e-8)) * class_weights["action"][
        action_targets
    ]
    action = _weighted_session_average(action_losses, action_mask, session_weights)

    quality_index = batch["quality"][:, None, None, None].expand(
        -1, output["quality_action_probabilities"].shape[1], 1, len(LABELS)
    )
    expert_probabilities = output["quality_action_probabilities"].gather(
        2, quality_index
    ).squeeze(2)
    selected_expert = expert_probabilities.gather(-1, action_targets[..., None]).squeeze(-1)
    expert_losses = -torch.log(selected_expert.clamp_min(1e-8)) * class_weights["action"][
        action_targets
    ]
    expert = _weighted_session_average(expert_losses, action_mask, session_weights)

    transition_kl_rows = (
        output["transition_prior"]
        * (
            torch.log(output["transition_prior"].clamp_min(1e-8))
            - torch.log(output["quality_action_probabilities"].clamp_min(1e-8))
        )
    ).sum(dim=(-1, -2)) / 2.0
    transition = _weighted_session_average(transition_kl_rows, action_mask, session_weights)

    total = float(training["auxiliary_behaviour_loss_weight"]) * auxiliary
    if mode.task_a_loss:
        total = total + float(training["quality_loss_weight"]) * quality
    if mode.task_c_loss:
        total = total + float(training["action_loss_weight"]) * action
        if mode.name != "c_only":
            total = total + float(training["quality_expert_action_loss_weight"]) * expert
    if mode.transition_regularization:
        total = total + float(training["transition_kl_loss_weight"]) * transition
    diagnostics = {
        "total": float(total.detach()),
        "auxiliary": float(auxiliary.detach()),
        "quality": float(quality.detach()),
        "action": float(action.detach()),
        "quality_expert_action": float(expert.detach()),
        "transition_kl": float(transition.detach()),
    }
    return total, diagnostics
