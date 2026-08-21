from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(__file__).resolve().parents[2]


def read_csv(relative_path: str, root: Path = DEFAULT_ROOT) -> list[dict[str, str]]:
    with (root / relative_path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(relative_path: str, root: Path = DEFAULT_ROOT) -> Any:
    return json.loads((root / relative_path).read_text(encoding="utf-8"))


def _number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _require_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label}: expected {expected}, found {actual}")


def validate_evidence(root: Path = DEFAULT_ROOT) -> list[str]:
    checks: list[str] = []

    models = read_csv("results/main/model_comparison.csv", root)
    if len(models) != 2:
        raise ValueError("Headline comparison must contain exactly two models")
    baseline, roberta = models
    if "Elastic-net" not in baseline["model"] or "roberta-base" not in roberta["model"]:
        raise ValueError("Unexpected model order in headline comparison")
    _require_close(_number(baseline, "accuracy"), 0.7769784172661871, "baseline accuracy")
    _require_close(_number(baseline, "f1_macro"), 0.7358350528097619, "baseline macro-F1")
    _require_close(_number(roberta, "accuracy"), 0.8180883864337102, "RoBERTa accuracy")
    _require_close(_number(roberta, "f1_macro"), 0.7744389233180415, "RoBERTa macro-F1")
    checks.append("headline metrics match the locked evidence")

    deltas = {row["metric"]: row for row in read_csv("results/main/metric_deltas.csv", root)}
    for metric in ("accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"):
        expected = _number(roberta, metric) - _number(baseline, metric)
        _require_close(
            _number(deltas[metric], "delta_roberta_minus_baseline"),
            expected,
            f"{metric} delta",
        )
    expected_brier = _number(roberta, "brier_multiclass") - _number(baseline, "brier_multiclass")
    _require_close(
        _number(deltas["brier_multiclass"], "delta_roberta_minus_baseline"),
        expected_brier,
        "Brier delta",
    )
    checks.append("reported deltas reproduce direct arithmetic")

    significance = read_csv("results/main/grouped_significance.csv", root)
    if {row["metric"] for row in significance} != {"accuracy", "f1_macro"}:
        raise ValueError("Grouped significance table has unexpected metrics")
    for row in significance:
        if _number(row, "bootstrap_ci95_low") <= 0.0:
            raise ValueError(f"{row['metric']} grouped interval does not exclude zero")
        if _number(row, "permutation_p_two_sided") >= 0.05:
            raise ValueError(f"{row['metric']} grouped permutation result is not significant")
    checks.append("transcript-grouped intervals and permutation tests support the gain")

    mcnemar_rows = read_csv("results/main/mcnemar_summary.csv", root)
    mcnemar = {row["Measure"]: row["Value"] for row in mcnemar_rows}
    if int(mcnemar["RoBERTa-only correct"]) + int(mcnemar["Baseline-only correct"]) != int(
        mcnemar["Discordant pairs"]
    ):
        raise ValueError("McNemar discordant counts do not sum")
    if float(mcnemar["Exact p-value"]) >= 0.05:
        raise ValueError("McNemar result is not significant")
    checks.append("paired correctness counts are internally consistent")

    calibration = read_json("results/main/calibration_summary.json", root)
    if calibration["roberta_calibrated_brier"] >= calibration["roberta_uncalibrated_brier"]:
        raise ValueError("Temperature scaling did not improve Brier score")
    if calibration["roberta_calibrated_ece"] >= calibration["roberta_uncalibrated_ece"]:
        raise ValueError("Temperature scaling did not improve ECE")
    checks.append("calibration improvement matches the recorded temperature fit")

    summaries = read_csv("results/summarisation/method_means.csv", root)
    winner = max(summaries, key=lambda row: _number(row, "Overall Usefulness"))
    if winner["Method"] != "Pre-BERTopic KMeans + MMR":
        raise ValueError("Summarisation winner does not match the aggregate evidence")
    checks.append("summarisation narrative matches the aggregate ranking")

    split = read_json("results/protocol/official_split.json", root)
    train = set(split["train_transcripts"])
    test = set(split["test_transcripts"])
    if train & test:
        raise ValueError("Official transcript split overlaps")
    if len(train) != split["n_train_transcripts"] or len(test) != split["n_test_transcripts"]:
        raise ValueError("Official split counts do not match transcript lists")
    if len(train | test) != 133:
        raise ValueError("Official split does not cover all 133 transcripts")
    checks.append("official transcript split is disjoint and exhaustive")

    manifest = read_json("data/source_manifest.json", root)
    if manifest["sha256"] != "b178db3b0b9858a0fa4ed670dabeccd63e975ba67e141d7ae16f2e8214f78e61":
        raise ValueError("Dataset digest is not the verified AnnoMI-simple digest")
    if manifest["rows"] != 9_699 or manifest["transcripts"] != 133:
        raise ValueError("Dataset manifest cardinality is incorrect")
    checks.append("dataset source is commit-pinned and checksum-locked")

    return checks
