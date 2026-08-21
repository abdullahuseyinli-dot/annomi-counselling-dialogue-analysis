from pathlib import Path

from annomi_portfolio.evidence import read_csv, read_json, validate_evidence

ROOT = Path(__file__).resolve().parents[1]


def test_complete_evidence_contract() -> None:
    checks = validate_evidence(ROOT)
    assert len(checks) == 8


def test_roberta_improves_primary_metrics() -> None:
    baseline, roberta = read_csv("results/main/model_comparison.csv", ROOT)
    assert float(roberta["accuracy"]) > float(baseline["accuracy"])
    assert float(roberta["f1_macro"]) > float(baseline["f1_macro"])
    assert float(roberta["brier_multiclass"]) < float(baseline["brier_multiclass"])


def test_temperature_scaling_improves_probability_metrics() -> None:
    metrics = read_json("results/main/calibration_summary.json", ROOT)
    assert metrics["roberta_calibrated_ece"] < metrics["roberta_uncalibrated_ece"]
    assert metrics["roberta_calibrated_brier"] < metrics["roberta_uncalibrated_brier"]


def test_summary_winner_is_data_driven() -> None:
    rows = read_csv("results/summarisation/method_means.csv", ROOT)
    scores = {row["Method"]: float(row["Overall Usefulness"]) for row in rows}
    assert scores["Pre-BERTopic KMeans + MMR"] > scores["BERTopic + MMR"]
