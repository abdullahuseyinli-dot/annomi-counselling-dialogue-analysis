from __future__ import annotations

import numpy as np

from annomi_research.inference import (
    _bootstrap_draws,
    _macro_f1_from_confusions,
    _per_seed_f1_deltas,
)


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


def test_per_seed_delta_uses_matching_seed_summaries(monkeypatch) -> None:
    summaries = {
        "candidate.json": {
            "metrics": {
                "per_seed": {
                    "17": {"source_balanced_macro_f1": 0.8},
                    "42": {"source_balanced_macro_f1": 0.7},
                }
            }
        },
        "baseline.json": {
            "metrics": {
                "per_seed": {
                    "17": {"source_balanced_macro_f1": 0.6},
                    "42": {"source_balanced_macro_f1": 0.75},
                }
            }
        },
    }
    monkeypatch.setattr(
        "annomi_research.inference.read_json", lambda path: summaries[path.name]
    )
    config = {
        "baseline": {"per_seed_summary": "baseline.json"},
        "candidate": {"per_seed_summary": "candidate.json"},
    }
    deltas, contrast = _per_seed_f1_deltas(config, baseline_f1=0.5)
    assert list(deltas) == ["17", "42"]
    assert np.allclose(list(deltas.values()), [0.2, -0.05])
    assert contrast == "candidate seed minus matching baseline seed"
