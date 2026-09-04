from __future__ import annotations

import numpy as np

from tools.build_ac_assets import (
    prediction_set_table,
    registered_inference_table,
    task_a_table,
    task_c_table,
)


def test_task_a_publication_table_contains_all_registered_models_and_checkpoints() -> None:
    frame = task_a_table()
    assert len(frame) == 35
    assert set(frame["checkpoint"]) == {"t3", "t5", "t10", "t20", "full"}
    primary = frame[frame["checkpoint"].eq("t10")].set_index("model")
    assert primary["source_balanced_balanced_accuracy"].idxmax() == "oracle_gold_codes"
    assert np.isclose(
        primary.loc["qtrace_mi", "source_balanced_balanced_accuracy"], 0.7
    )


def test_task_c_publication_table_retains_negative_candidate_result() -> None:
    frame = task_c_table().set_index("model")
    assert len(frame) == 6
    assert frame["source_balanced_macro_f1"].idxmax() == "c_only"
    assert (
        frame.loc["qtrace_mi", "source_balanced_macro_f1"]
        < frame.loc["c_only", "source_balanced_macro_f1"]
    )


def test_registered_inference_table_reconstructs_gate_directions() -> None:
    frame = registered_inference_table().set_index("metric")
    assert len(frame) == 3
    assert frame["passed"].eq(False).all()
    assert frame.loc["t10_source_balanced_balanced_accuracy", "point_delta"] > 0
    assert frame.loc["source_balanced_macro_f1", "ci95_high"] < 0
    assert frame.loc["source_balanced_brier", "ci95_low"] > 0


def test_prediction_set_table_records_wide_nonempty_sets() -> None:
    frame = prediction_set_table().set_index("model")
    assert frame["source_balanced_coverage"].between(0, 1).all()
    assert frame["source_balanced_mean_set_size"].between(1, 4).all()
    assert frame.loc["qtrace_mi", "source_balanced_singleton_rate"] == 0
