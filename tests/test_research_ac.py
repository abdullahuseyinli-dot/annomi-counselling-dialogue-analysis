from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from annomi_research.ac_data import (
    build_session_turns,
    build_task_a_examples,
    build_task_c_examples,
)
from annomi_research.ac_metrics import (
    add_prediction_sets,
    aps_scores,
    evaluate_prediction_sets,
    source_crc_threshold,
)
from annomi_research.constants import CLIENT_LABELS, LABELS
from annomi_research.data import Corpus
from annomi_research.qtrace import estimate_quality_transitions
from annomi_research.qtrace_model import QTraceMode, QTraceModel


def _corpus() -> Corpus:
    rows = []
    specifications = [
        (1, "source-a", "high", "question", "change", "reflection"),
        (2, "source-b", "low", "therapist_input", "sustain", "other"),
    ]
    for transcript, source, quality, first_action, client_label, target_action in specifications:
        values = [
            (0, "therapist", "first therapist", first_action, ""),
            (1, "client", "observed client", "", client_label),
            (2, "therapist", "future target therapist", target_action, ""),
            (3, "client", "future client", "", "neutral"),
            (4, "therapist", "last therapist", "question", ""),
        ]
        for utterance, role, text, therapist_label, talk_type in values:
            rows.append(
                {
                    "transcript_id": transcript,
                    "utterance_id": utterance,
                    "interlocutor": role,
                    "utterance_text": text,
                    "main_therapist_behaviour": therapist_label,
                    "client_talk_type": talk_type,
                    "source_id": source,
                    "normalized_text": text.casefold(),
                    "mi_quality": quality,
                }
            )
    return Corpus(pd.DataFrame(rows), pd.DataFrame())


def test_task_c_is_a_strict_client_handoff_without_target_text() -> None:
    examples = build_task_c_examples(_corpus(), context_turns=10)
    first = examples[examples["transcript_id"].eq(1)].iloc[0]
    assert first["decision_utterance_id"] == 1
    assert first["target_utterance_id"] == 2
    assert first["label"] == "reflection"
    assert "observed client" in first["context_text"]
    assert "future target therapist" not in first["context_text"]
    assert "future client" not in first["context_text"]


def test_task_a_absolute_prefix_does_not_include_later_turns() -> None:
    examples = build_task_a_examples(_corpus(), therapist_budgets=(1, 2))
    first = examples[
        examples["transcript_id"].eq(1) & examples["checkpoint"].eq("t1")
    ].iloc[0]
    second = examples[
        examples["transcript_id"].eq(1) & examples["checkpoint"].eq("t2")
    ].iloc[0]
    assert "first therapist" in first["prefix_text"]
    assert "observed client" not in first["prefix_text"]
    assert "future target therapist" in second["prefix_text"]
    assert "future client" not in second["prefix_text"]


def test_session_builder_matches_handoff_targets_and_quality_positions() -> None:
    sessions = build_session_turns(_corpus(), therapist_budgets=(1, 2))
    assert len(sessions) == 2
    assert sum(int((session.next_action_targets >= 0).sum()) for session in sessions) == 4
    assert sessions[0].quality_positions == {"t1": 0, "t2": 2, "full": 4}
    assert sessions[0].next_action_targets[1] == LABELS.index("reflection")


def test_quality_transition_priors_are_valid() -> None:
    sessions = build_session_turns(_corpus(), therapist_budgets=(1, 2))
    values = estimate_quality_transitions(sessions, dirichlet_strength=2.0)
    assert values.shape == (2, len(LABELS), len(CLIENT_LABELS), len(LABELS))
    assert np.isfinite(values).all()
    assert (values > 0).all()
    assert np.allclose(values.sum(axis=-1), 1.0)


def test_qtrace_forward_never_accepts_target_or_quality_labels() -> None:
    transition = np.full(
        (2, len(LABELS), len(CLIENT_LABELS), len(LABELS)), 1.0 / len(LABELS)
    )
    mode = QTraceMode("qtrace_mi", True, True, True, True)
    model = QTraceModel(
        embedding_size=12,
        transition_probabilities=transition,
        low_quality_prior=0.2,
        architecture={
            "hidden_size": 16,
            "role_embedding_size": 4,
            "gru_layers": 1,
            "dropout": 0.0,
            "quality_text_evidence_bound": 0.25,
        },
        mode=mode,
    )
    output = model(
        embeddings=torch.randn(2, 5, 12),
        roles=torch.tensor([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]]),
        lengths=torch.tensor([5, 4]),
    )
    assert output["action_probabilities"].shape == (2, 5, len(LABELS))
    assert output["online_quality_probabilities"].shape == (2, 5, 2)
    assert torch.allclose(output["action_probabilities"].sum(dim=-1), torch.ones(2, 5))
    assert torch.allclose(
        output["online_quality_probabilities"].sum(dim=-1), torch.ones(2, 5)
    )


def test_source_crc_prediction_sets_are_nonempty_and_reconstruct_coverage() -> None:
    probabilities = np.asarray(
        [
            [0.70, 0.20, 0.05, 0.05],
            [0.35, 0.30, 0.20, 0.15],
            [0.10, 0.10, 0.70, 0.10],
            [0.25, 0.25, 0.25, 0.25],
        ]
    )
    labels = np.asarray([LABELS[0], LABELS[1], LABELS[2], LABELS[3]])
    sources = np.asarray(["a", "a", "b", "b"])
    scores = aps_scores(probabilities)
    true_indices = np.asarray([LABELS.index(label) for label in labels])
    calibration = source_crc_threshold(
        scores[np.arange(len(scores)), true_indices], sources, alpha=0.6
    )
    frame = pd.DataFrame({"source_id": sources, "label": labels})
    for index, label in enumerate(LABELS):
        frame[f"prob_{label}"] = probabilities[:, index]
    result = add_prediction_sets(frame, float(calibration["threshold"]))
    metrics = evaluate_prediction_sets(result)
    assert result["prediction_set_size"].between(1, len(LABELS)).all()
    assert 0 <= metrics["source_balanced_coverage"] <= 1
