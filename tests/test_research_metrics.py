from __future__ import annotations

import pandas as pd

from annomi_research.constants import LABELS
from annomi_research.metrics import evaluate_predictions, source_balanced_weights


def test_sources_receive_equal_total_weight() -> None:
    weights = source_balanced_weights(["long", "long", "long", "short"])
    assert weights[:3].sum() == weights[3]


def test_metrics_reconstruct_perfect_predictions() -> None:
    rows = []
    for index, label in enumerate(LABELS):
        row = {
            "label": label,
            "prediction": label,
            "source_id": f"source-{index}",
            "seen_text_in_outer_train": index % 2 == 0,
        }
        row.update({f"prob_{candidate}": float(candidate == label) for candidate in LABELS})
        rows.append(row)
    result = evaluate_predictions(pd.DataFrame(rows))
    assert result["source_balanced_macro_f1"] == 1.0
    assert result["utterance_macro_f1"] == 1.0
    assert result["source_balanced_brier"] == 0.0
