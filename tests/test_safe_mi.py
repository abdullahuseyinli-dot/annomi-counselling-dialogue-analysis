from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from annomi_research.constants import CLIENT_LABELS, LABELS
from annomi_research.safe_mi_extension import crossfit_prediction_sets
from annomi_research.safe_mi_model import SafeMIMode, SafeMIModel, safe_mi_loss


def _architecture() -> dict[str, float | int]:
    return {
        "hidden_size": 16,
        "role_embedding_size": 4,
        "dropout": 0.0,
        "local_attention_window": 4,
        "attention_heads": 4,
        "attention_feedforward_size": 32,
        "attention_layers": 1,
        "quality_text_evidence_bound": 0.25,
        "quality_evidence_decay": 0.9,
        "maximum_transition_residual": 0.3,
    }


def _mode(**overrides: object) -> SafeMIMode:
    values: dict[str, object] = {
        "name": "test",
        "encoder_variant": "frozen",
        "context_model": "client_attention",
        "task_a_loss": True,
        "task_c_loss": True,
        "detach_task_a": True,
        "discounted_task_a": True,
        "action_loss": "weighted_ce",
        "transition_loss_weight": 0.0,
        "transition_residual": True,
    }
    values.update(overrides)
    return SafeMIMode(**values)  # type: ignore[arg-type]


def _model(mode: SafeMIMode) -> SafeMIModel:
    transitions = np.full(
        (len(LABELS), len(CLIENT_LABELS), len(LABELS)),
        1.0 / len(LABELS),
    )
    return SafeMIModel(
        embedding_size=12,
        transition_probabilities=transitions,
        low_quality_prior=0.2,
        action_class_prior=np.full(len(LABELS), 1.0 / len(LABELS)),
        architecture=_architecture(),
        mode=mode,
    )


def test_safe_mi_attention_is_causal_and_outputs_probabilities() -> None:
    model = _model(_mode())
    model.eval()
    embeddings = torch.randn(2, 6, 12)
    roles = torch.tensor([[1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 0, 0]])
    lengths = torch.tensor([6, 4])
    with torch.inference_mode():
        first = model(embeddings, roles, lengths)
        changed = embeddings.clone()
        changed[:, 4:] += 100.0
        second = model(changed, roles, lengths)
    assert first["action_probabilities"].shape == (2, 6, len(LABELS))
    assert first["online_quality_probabilities"].shape == (2, 6, 2)
    assert torch.allclose(first["action_probabilities"].sum(-1), torch.ones(2, 6))
    assert torch.allclose(first["action_probabilities"][:, :4], second["action_probabilities"][:, :4])


def test_transition_residual_starts_as_exact_noop() -> None:
    active = _model(_mode(transition_residual=True))
    inactive = _model(_mode(transition_residual=False))
    inactive.load_state_dict(active.state_dict())
    embeddings = torch.randn(2, 5, 12)
    roles = torch.tensor([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]])
    lengths = torch.tensor([5, 4])
    active.eval()
    inactive.eval()
    with torch.inference_mode():
        active_output = active(embeddings, roles, lengths)
        inactive_output = inactive(embeddings, roles, lengths)
    assert float(active_output["transition_strength"]) == 0.0
    assert torch.equal(
        active_output["action_probabilities"], inactive_output["action_probabilities"]
    )


def test_detached_task_a_loss_does_not_update_causal_trunk() -> None:
    mode = _mode(
        task_c_loss=False,
        detach_task_a=True,
        discounted_task_a=False,
        transition_residual=False,
    )
    model = _model(mode)
    embeddings = torch.randn(2, 5, 12)
    roles = torch.tensor([[1, 0, 1, 0, 1], [1, 0, 1, 0, 1]])
    lengths = torch.tensor([5, 5])
    output = model(embeddings, roles, lengths)
    batch = {
        "therapist_labels": torch.tensor([[0, -100, 1, -100, 2], [1, -100, 0, -100, 3]]),
        "client_labels": torch.tensor([[-100, 0, -100, 1, -100], [-100, 1, -100, 2, -100]]),
        "next_action_targets": torch.tensor([[-100, 1, -100, 2, -100], [-100, 0, -100, 3, -100]]),
        "quality_mask": torch.tensor([[False, False, True, False, True]] * 2),
        "quality": torch.tensor([0, 1]),
        "session_weights": torch.ones(2),
    }
    weights = {
        "therapist": torch.ones(len(LABELS)),
        "client": torch.ones(len(CLIENT_LABELS)),
        "quality": torch.ones(2),
        "action": torch.ones(len(LABELS)),
        "action_log_prior": torch.zeros(len(LABELS)),
    }
    loss, _ = safe_mi_loss(
        output,
        batch,
        mode,
        {
            "auxiliary_behaviour_loss_weight": 0.0,
            "quality_loss_weight": 1.0,
            "action_loss_weight": 0.0,
            "logit_adjustment_tau": 1.0,
            "transition_residual_l1_weight": 0.0,
        },
        weights,
    )
    loss.backward()
    assert model.quality_head.weight.grad is not None
    assert model.input_projection[0].weight.grad is None
    assert model.action_head.weight.grad is None


def test_crossfit_prediction_sets_never_use_target_fold_sources() -> None:
    rows: list[dict[str, object]] = []
    for fold in range(3):
        for index, label in enumerate(LABELS):
            probabilities = np.full(len(LABELS), 0.1)
            probabilities[index] = 0.7
            row: dict[str, object] = {
                "model": "candidate",
                "outer_fold": fold,
                "source_id": f"source-{fold}-{index}",
                "label": label,
            }
            for class_index, class_label in enumerate(LABELS):
                row[f"prob_{class_label}"] = probabilities[class_index]
            rows.append(row)
    result, records = crossfit_prediction_sets(pd.DataFrame(rows), 0.2)
    assert len(result) == len(rows)
    assert result["prediction_set_method"].eq("outer-crossfit-source-crc").all()
    assert result["set_covered"].all()
    assert all(record["calibration_sources"] == 8 for record in records)
