from __future__ import annotations

import numpy as np
import pandas as pd

from annomi_research.constants import LABELS
from annomi_research.neural import (
    _ensemble_ledger,
    _round_median_epoch,
    _schedule_multiplier,
    _training_weights,
)


def test_training_weights_are_finite_positive_and_normalized() -> None:
    frame = pd.DataFrame(
        {
            "source_id": ["long", "long", "long", "short"],
            "label": ["other", "other", "question", "reflection"],
        }
    )
    weights = _training_weights(frame)
    assert np.isfinite(weights).all()
    assert (weights > 0).all()
    assert np.isclose(weights.mean(), 1.0)


def test_schedule_warms_up_then_decays() -> None:
    values = [_schedule_multiplier(step, warmup_steps=2, total_steps=6) for step in range(6)]
    assert values[:2] == [0.5, 1.0]
    assert values[2] > values[3] > values[4] > values[5]


def test_epoch_rule_rounds_median_without_zero_epochs() -> None:
    assert _round_median_epoch([1, 2, 2]) == 2
    assert _round_median_epoch([1, 2, 3]) == 2


def test_seed_ensemble_averages_probabilities_before_argmax() -> None:
    rows = []
    for seed, reflection_probability in ((17, 0.8), (42, 0.4)):
        row = {
            "model": "candidate",
            "seed": seed,
            "outer_fold": 0,
            "transcript_id": 1,
            "utterance_id": 2,
            "source_id": "source",
            "label": "reflection",
            "prediction": "reflection",
            "seen_text_in_outer_train": False,
            "normalized_text_sha256": "digest",
            "selected_recipe": "recipe",
            "max_length": 128,
            "epochs": 2,
            "pretrained_revision": "revision",
        }
        remaining = (1.0 - reflection_probability) / 3.0
        row.update(
            {
                f"prob_{label}": (reflection_probability if label == "reflection" else remaining)
                for label in LABELS
            }
        )
        rows.append(row)
    ensemble = _ensemble_ledger(pd.DataFrame(rows), "candidate")
    assert np.isclose(ensemble["prob_reflection"].item(), 0.6)
    assert ensemble["prediction"].item() == "reflection"
