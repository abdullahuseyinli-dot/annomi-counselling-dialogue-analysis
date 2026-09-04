from __future__ import annotations

import numpy as np

from annomi_research.inference import _bootstrap_draws, _macro_f1_from_confusions


def test_macro_f1_from_confusion_supports_batches() -> None:
    confusions = np.asarray(
        [
            np.eye(4),
            np.zeros((4, 4)),
        ]
    )
    values = _macro_f1_from_confusions(confusions)
    assert values.tolist() == [1.0, 0.0]


def test_paired_bootstrap_preserves_identical_model_zero_delta() -> None:
    components = {
        "sources": ["a", "b"],
        "confusions": np.asarray([np.eye(4), np.eye(4)]),
        "brier": np.asarray([0.1, 0.2]),
        "log_loss": np.asarray([0.2, 0.3]),
    }
    draws = _bootstrap_draws(components, components, n_resamples=50, seed=7)
    assert np.allclose(draws.filter(like="delta_").to_numpy(), 0.0)
