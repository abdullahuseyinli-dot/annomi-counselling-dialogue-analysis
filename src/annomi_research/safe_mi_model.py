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
class SafeMIMode:
    """One bounded SAFE-MI ablation.

    ``encoder_variant`` selects the frozen or fold-adapted turn representations in
    the execution layer.  It deliberately does not change the prediction head.
    """

    name: str
    encoder_variant: str
    context_model: str
    task_a_loss: bool
    task_c_loss: bool
    detach_task_a: bool
    discounted_task_a: bool
    action_loss: str
    transition_loss_weight: float
    transition_residual: bool


def mode_from_config(config: dict[str, Any], name: str) -> SafeMIMode:
    for value in config["models"]:
        if value["model"] != name:
            continue
        mode = SafeMIMode(
            name=str(value["model"]),
            encoder_variant=str(value["encoder_variant"]),
            context_model=str(value["context_model"]),
            task_a_loss=bool(value["task_a_loss"]),
            task_c_loss=bool(value["task_c_loss"]),
            detach_task_a=bool(value.get("detach_task_a", False)),
            discounted_task_a=bool(value.get("discounted_task_a", False)),
            action_loss=str(value.get("action_loss", "weighted_ce")),
            transition_loss_weight=float(value.get("transition_loss_weight", 0.0)),
            transition_residual=bool(value.get("transition_residual", False)),
        )
        if mode.encoder_variant not in {"frozen", "adapted"}:
            raise ValueError(f"Unknown SAFE-MI encoder variant: {mode.encoder_variant}")
        if mode.context_model not in {"gru", "client_attention"}:
            raise ValueError(f"Unknown SAFE-MI context model: {mode.context_model}")
        if mode.action_loss not in {"weighted_ce", "logit_adjusted"}:
            raise ValueError(f"Unknown SAFE-MI action loss: {mode.action_loss}")
        if mode.discounted_task_a and not mode.task_a_loss:
            raise ValueError("Discounted Task A evidence requires an active Task A loss")
        return mode
    raise ValueError(f"Unknown SAFE-MI mode: {name}")


class SafeMIModel(nn.Module):
    """Baseline-preserving causal model for Tasks A and C.

    Task C always uses one unconditional policy.  Task A can read a detached copy
    of the causal state, but it never routes, gates, or otherwise changes Task C.
    """

    def __init__(
        self,
        embedding_size: int,
        transition_probabilities: np.ndarray,
        low_quality_prior: float,
        action_class_prior: np.ndarray,
        architecture: dict[str, Any],
        mode: SafeMIMode,
    ) -> None:
        super().__init__()
        hidden_size = int(architecture["hidden_size"])
        role_size = int(architecture["role_embedding_size"])
        dropout = float(architecture["dropout"])
        self.mode = mode
        self.local_window = int(architecture["local_attention_window"])
        self.evidence_bound = float(architecture["quality_text_evidence_bound"])
        self.evidence_decay = float(architecture["quality_evidence_decay"])
        self.maximum_transition_residual = float(
            architecture["maximum_transition_residual"]
        )

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

        if mode.context_model == "gru":
            self.gru: nn.GRU | None = nn.GRU(
                hidden_size,
                hidden_size,
                num_layers=1,
                batch_first=True,
            )
            self.summary_projection: nn.Linear | None = None
            self.context_encoder: nn.TransformerEncoder | None = None
        else:
            self.gru = None
            self.summary_projection = nn.Linear(hidden_size, hidden_size)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=int(architecture["attention_heads"]),
                dim_feedforward=int(architecture["attention_feedforward_size"]),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.context_encoder = nn.TransformerEncoder(
                layer,
                num_layers=int(architecture["attention_layers"]),
                norm=nn.LayerNorm(hidden_size),
                enable_nested_tensor=False,
            )

        self.state_dropout = nn.Dropout(dropout)
        self.action_head = nn.Linear(hidden_size, len(LABELS))
        self.quality_head = nn.Linear(hidden_size, 2)
        self.text_quality_evidence = nn.Linear(hidden_size, 1)
        self.action_quality_evidence = nn.Linear(len(LABELS), 1, bias=False)
        self.action_evidence_logit = nn.Parameter(torch.tensor(-1.3862944))

        self.transition_gate = nn.Linear(hidden_size, 1)
        self.transition_residual_parameter = nn.Parameter(torch.zeros(()))
        transitions = torch.as_tensor(transition_probabilities, dtype=torch.float32)
        if transitions.shape != (len(LABELS), len(CLIENT_LABELS), len(LABELS)):
            raise ValueError("Unexpected SAFE-MI transition tensor shape")
        if not torch.allclose(transitions.sum(dim=-1), torch.ones_like(transitions[..., 0])):
            raise ValueError("SAFE-MI transition distributions do not sum to one")
        self.register_buffer("transition_probabilities", transitions)

        prior = np.asarray(action_class_prior, dtype=float)
        if prior.shape != (len(LABELS),) or (prior <= 0).any():
            raise ValueError("Action class prior must contain one positive value per class")
        prior = prior / prior.sum()
        self.register_buffer("action_log_prior", torch.log(torch.tensor(prior, dtype=torch.float32)))
        low_quality_prior = float(np.clip(low_quality_prior, 1e-6, 1 - 1e-6))
        self.register_buffer(
            "quality_prior_log_odds",
            torch.tensor(math.log(low_quality_prior / (1.0 - low_quality_prior))),
        )

    @staticmethod
    def _last_therapist_distribution(
        probabilities: torch.Tensor,
        roles: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, classes = probabilities.shape
        last = torch.full(
            (batch_size, classes),
            1.0 / classes,
            dtype=probabilities.dtype,
            device=probabilities.device,
        )
        history: list[torch.Tensor] = []
        for position in range(sequence_length):
            history.append(last)
            update = roles[:, position].eq(1) & valid_mask[:, position]
            last = torch.where(update[:, None], probabilities[:, position], last)
        return torch.stack(history, dim=1)

    def _contextualize(
        self,
        projected: torch.Tensor,
        lengths: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.gru is not None:
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
                total_length=projected.shape[1],
            )
            return states

        if self.context_encoder is None or self.summary_projection is None:
            raise AssertionError("Client-attention modules are missing")
        valid_values = valid_mask[..., None].to(projected.dtype)
        prefix_sum = torch.cumsum(projected * valid_values, dim=1)
        counts = torch.cumsum(valid_values, dim=1).clamp_min(1)
        contextual_input = projected + self.summary_projection(prefix_sum / counts)
        sequence_length = projected.shape[1]
        positions = torch.arange(sequence_length, device=projected.device)
        distance = positions[:, None] - positions[None, :]
        blocked = (distance < 0) | (distance >= self.local_window)
        states = self.context_encoder(
            contextual_input,
            mask=blocked,
            src_key_padding_mask=~valid_mask,
        )
        return states * valid_values

    def _discounted_quality_logits(
        self,
        states: torch.Tensor,
        therapist_probabilities: torch.Tensor,
        roles: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        evidence_states = states.detach() if self.mode.detach_task_a else states
        text_evidence = self.evidence_bound * torch.tanh(
            self.text_quality_evidence(evidence_states).squeeze(-1).float()
        )
        code_probabilities = (
            therapist_probabilities.detach()
            if self.mode.detach_task_a
            else therapist_probabilities
        )
        action_scale = torch.sigmoid(self.action_evidence_logit)
        action_evidence = action_scale * self.evidence_bound * torch.tanh(
            self.action_quality_evidence(code_probabilities).squeeze(-1).float()
        )
        action_evidence *= roles.eq(1)
        text_evidence *= valid_mask
        action_evidence *= valid_mask

        accumulator = torch.zeros(states.shape[0], device=states.device)
        effective_count = torch.zeros_like(accumulator)
        log_odds: list[torch.Tensor] = []
        decay = self.evidence_decay
        for position in range(states.shape[1]):
            active = valid_mask[:, position]
            update = text_evidence[:, position] + action_evidence[:, position]
            accumulator = torch.where(active, decay * accumulator + update, accumulator)
            effective_count = torch.where(
                active,
                decay * decay * effective_count + 1.0,
                effective_count,
            )
            normalized = accumulator / torch.sqrt(effective_count.clamp_min(1.0))
            log_odds.append(self.quality_prior_log_odds + normalized)
        low_log_odds = torch.stack(log_odds, dim=1).clamp(-12.0, 12.0)
        quality_logits = torch.stack([torch.zeros_like(low_log_odds), low_log_odds], dim=-1)
        return quality_logits, text_evidence, action_evidence

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
        states = self.state_dropout(self._contextualize(projected, lengths, valid_mask))

        last_therapist = self._last_therapist_distribution(
            therapist_probabilities, roles, valid_mask
        )
        transition_prior = torch.einsum(
            "btp,btc,pca->bta",
            last_therapist,
            client_probabilities,
            self.transition_probabilities,
        ).clamp_min(1e-8)
        transition_gate = torch.sigmoid(self.transition_gate(states).squeeze(-1).float())
        action_logits = self.action_head(states).float()
        transition_strength = self.maximum_transition_residual * torch.tanh(
            self.transition_residual_parameter
        )
        if self.mode.transition_residual:
            centered_log_prior = torch.log(transition_prior)
            centered_log_prior -= centered_log_prior.mean(dim=-1, keepdim=True)
            action_logits = action_logits + (
                transition_strength * transition_gate[..., None] * centered_log_prior
            )
        action_probabilities = torch.softmax(action_logits, dim=-1)

        if self.mode.discounted_task_a:
            quality_logits, text_evidence, action_evidence = self._discounted_quality_logits(
                states,
                therapist_probabilities,
                roles,
                valid_mask,
            )
        else:
            quality_states = states.detach() if self.mode.detach_task_a else states
            quality_logits = self.quality_head(quality_states).float()
            text_evidence = torch.zeros(
                (batch_size, sequence_length), device=states.device, dtype=torch.float32
            )
            action_evidence = torch.zeros_like(text_evidence)
        quality_probabilities = torch.softmax(quality_logits, dim=-1)

        return {
            "therapist_auxiliary_logits": therapist_logits,
            "client_auxiliary_logits": client_logits,
            "action_logits": action_logits,
            "action_probabilities": action_probabilities,
            "quality_logits": quality_logits,
            "online_quality_probabilities": quality_probabilities,
            "transition_prior": transition_prior,
            "transition_gate": transition_gate,
            "transition_strength": transition_strength,
            "text_quality_evidence": text_evidence,
            "action_quality_evidence": action_evidence,
            "states": states,
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


def safe_mi_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    mode: SafeMIMode,
    training: dict[str, Any],
    class_weights: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute source/session-balanced losses without an A-to-C gradient path."""

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
    quality_losses = F.cross_entropy(
        output["quality_logits"].transpose(1, 2),
        quality_targets,
        weight=class_weights["quality"],
        reduction="none",
    )
    quality = _weighted_session_average(quality_losses, quality_mask, session_weights)

    action_targets = batch["next_action_targets"].clamp_min(0)
    action_logits = output["action_logits"]
    action_weight: torch.Tensor | None = class_weights["action"]
    if mode.action_loss == "logit_adjusted":
        tau = float(training["logit_adjustment_tau"])
        action_logits = action_logits + tau * class_weights["action_log_prior"]
        action_weight = None
    action_losses = F.cross_entropy(
        action_logits.transpose(1, 2),
        action_targets,
        weight=action_weight,
        reduction="none",
    )
    action = _weighted_session_average(action_losses, action_mask, session_weights)

    probabilities = torch.softmax(output["action_logits"], dim=-1).clamp_min(1e-8)
    transition_kl_rows = (
        probabilities
        * (torch.log(probabilities) - torch.log(output["transition_prior"].clamp_min(1e-8)))
    ).sum(dim=-1)
    transition = _weighted_session_average(
        transition_kl_rows,
        action_mask,
        session_weights,
    )
    residual_penalty = output["transition_strength"].abs()

    total = float(training["auxiliary_behaviour_loss_weight"]) * auxiliary
    if mode.task_a_loss:
        total = total + float(training["quality_loss_weight"]) * quality
    if mode.task_c_loss:
        total = total + float(training["action_loss_weight"]) * action
        total = total + mode.transition_loss_weight * transition
        if mode.transition_residual:
            total = total + float(training["transition_residual_l1_weight"]) * residual_penalty
    diagnostics = {
        "total": float(total.detach()),
        "auxiliary": float(auxiliary.detach()),
        "quality": float(quality.detach()),
        "action": float(action.detach()),
        "transition_kl": float(transition.detach()),
        "transition_residual_penalty": float(residual_penalty.detach()),
    }
    return total, diagnostics
