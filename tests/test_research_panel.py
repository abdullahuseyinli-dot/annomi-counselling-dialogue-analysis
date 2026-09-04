from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from annomi_research.data import Corpus, build_multiannotator_task
from annomi_research.panel import PanelMIHead, evaluate_vote_predictions


def _synthetic_corpus() -> Corpus:
    utterances = pd.DataFrame(
        [
            {
                "transcript_id": 1,
                "utterance_id": 0,
                "interlocutor": "therapist",
                "utterance_text": "Would that help?",
                "normalized_text": "would that help?",
                "source_id": "source-1",
            },
            {
                "transcript_id": 2,
                "utterance_id": 0,
                "interlocutor": "therapist",
                "utterance_text": "You feel uncertain.",
                "normalized_text": "you feel uncertain.",
                "source_id": "source-2",
            },
        ]
    )
    annotations = pd.DataFrame(
        [
            {
                "transcript_id": 1,
                "utterance_id": 0,
                "annotator_id": "0",
                "main_therapist_behaviour": "question",
            },
            {
                "transcript_id": 1,
                "utterance_id": 0,
                "annotator_id": "1",
                "main_therapist_behaviour": "reflection",
            },
            {
                "transcript_id": 2,
                "utterance_id": 0,
                "annotator_id": "0",
                "main_therapist_behaviour": "reflection",
            },
            {
                "transcript_id": 2,
                "utterance_id": 0,
                "annotator_id": "1",
                "main_therapist_behaviour": "reflection",
            },
        ]
    )
    return Corpus(utterances=utterances, annotations=annotations)


def test_multiannotator_task_preserves_panel_and_vote_mass() -> None:
    task = build_multiannotator_task(
        _synthetic_corpus(),
        task="therapist",
        label_column="main_therapist_behaviour",
        labels=("reflection", "question"),
        expected_annotations_per_item=2,
        expected_items=2,
    )
    assert task.annotator_ids == ("0", "1")
    assert task.annotation_label_indices.tolist() == [[1, 0], [0, 0]]
    assert task.items["vote_prob_reflection"].tolist() == [0.5, 1.0]
    assert task.items["vote_prob_question"].tolist() == [0.5, 0.0]
    assert task.items["plurality_tie"].tolist() == [True, False]
    assert task.items["plurality_label"].tolist() == ["reflection", "reflection"]


def test_panel_deviations_are_centered_around_shared_logits() -> None:
    torch.manual_seed(3)
    model = PanelMIHead(input_size=5, n_classes=3, n_annotators=4, rank=2)
    with torch.no_grad():
        model.annotator_bias_raw.normal_()
        model.annotator_factor_raw.normal_()
    features = torch.randn(7, 5)
    logits = model(features)
    assert torch.allclose(logits.mean(dim=1), model.base(features), atol=1e-6)


def test_vote_metrics_use_transcript_balancing() -> None:
    items = pd.DataFrame(
        {
            "transcript_id": [1, 1, 2],
            "plurality_label": ["a", "a", "b"],
            "vote_entropy": [0.0, 0.0, 0.0],
            "vote_prob_a": [1.0, 1.0, 0.0],
            "vote_prob_b": [0.0, 0.0, 1.0],
        }
    )
    probabilities = np.asarray([[0.9, 0.1], [0.9, 0.1], [0.2, 0.8]])
    result = evaluate_vote_predictions(items, probabilities, ("a", "b"))
    transcript_one = -np.log(0.9)
    transcript_two = -np.log(0.8)
    assert np.isclose(
        result["transcript_balanced_vote_log_score"],
        0.5 * (transcript_one + transcript_two),
    )
    assert result["transcript_balanced_plurality_macro_f1"] == 1.0
