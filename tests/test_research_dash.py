from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn

from annomi_research.constants import LABELS
from annomi_research.dash import DashMIModel, _soft_training_targets


class _FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(16, 8)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> SimpleNamespace:
        del attention_mask
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def _config() -> dict:
    return {
        "pretrained_encoder": {
            "model_id": "fake",
            "revision": "revision",
            "trust_remote_code": False,
        },
        "architecture": {
            "dropout": 0.0,
            "attention_heads": 2,
            "initial_gate_bias": -2.0,
            "zero_initialize_context_residual_output": True,
        },
    }


def test_soft_targets_use_votes_only_when_registered() -> None:
    frame = pd.DataFrame(
        {
            "label": ["reflection", "question"],
            "vote_prob_reflection": [0.6, 0.1],
            "vote_prob_question": [0.4, 0.7],
            "vote_prob_therapist_input": [0.0, 0.1],
            "vote_prob_other": [0.0, 0.1],
        }
    )
    hard = _soft_training_targets(frame, 0.0)
    votes = _soft_training_targets(frame, 1.0)
    assert np.array_equal(hard, np.eye(len(LABELS), dtype=np.float32)[[0, 1]])
    assert np.allclose(votes, frame[[f"vote_prob_{label}" for label in LABELS]])


def test_zero_initialized_residual_starts_at_target_prediction(monkeypatch) -> None:
    monkeypatch.setattr(
        "annomi_research.dash.AutoModel.from_pretrained",
        lambda *args, **kwargs: _FakeEncoder(),
    )
    model = DashMIModel(_config()).eval()
    ids = torch.tensor([[0, 1, 2], [0, 3, 4]])
    mask = torch.ones_like(ids)
    logits, target_logits, diagnostics = model(
        ids,
        mask,
        ids,
        mask,
        torch.tensor([1.0, 0.0]),
    )
    assert torch.equal(logits, target_logits)
    assert diagnostics["context_residual_l2"].eq(0).all()
    assert diagnostics["context_gate_mean"][1].item() == 0.0
