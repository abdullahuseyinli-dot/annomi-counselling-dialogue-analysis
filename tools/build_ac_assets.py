from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from annomi_research.io import canonical_json_bytes, sha256_file, write_create_only

ASSET_DIR = ROOT / "assets" / "research"
TABLE_DIR = ROOT / "results" / "research" / "publication_ac_v1"
BASELINE_SUMMARY = ROOT / "results" / "research" / "ac_v1" / "baselines" / "summary.json"
QTRACE_DIR = ROOT / "results" / "research" / "ac_v1" / "qtrace_mi"
QTRACE_SUMMARY = QTRACE_DIR / "summary.json"

MODEL_LABELS = {
    "class_prior": "Class prior",
    "structure_only": "Structure only",
    "tfidf_raw_prefix": "TF-IDF prefix",
    "oracle_gold_codes": "Gold-code oracle",
    "a_only": "A-only neural",
    "c_only": "C-only neural",
    "joint_no_transition": "Joint, no transition",
    "qtrace_mi": "Q-TRACE-MI",
    "tfidf_causal10": "TF-IDF causal",
    "oracle_gold_markov": "Gold-history oracle",
}
COLORS = {
    "classical": "#6B7280",
    "single_task": "#2563EB",
    "joint": "#0F766E",
    "candidate": "#7C3AED",
    "oracle": "#D97706",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _family(model: str) -> str:
    if model.startswith("oracle_"):
        return "oracle"
    if model == "qtrace_mi":
        return "candidate"
    if model == "joint_no_transition":
        return "joint"
    if model in {"a_only", "c_only"}:
        return "single_task"
    return "classical"


def task_a_table() -> pd.DataFrame:
    baseline = _read_json(BASELINE_SUMMARY)["task_a_metrics"]
    neural = _read_json(QTRACE_SUMMARY)["task_a_metrics"]
    rows: list[dict[str, Any]] = []
    for model, checkpoints in {**baseline, **neural}.items():
        for checkpoint, values in checkpoints.items():
            rows.append(
                {
                    "model": model,
                    "display_name": MODEL_LABELS[model],
                    "family": _family(model),
                    "checkpoint": checkpoint,
                    "n_transcripts": values["n_transcripts"],
                    "n_sources": values["n_sources"],
                    "low_transcripts": values["low_transcripts"],
                    "source_balanced_balanced_accuracy": values[
                        "source_balanced_balanced_accuracy"
                    ],
                    "source_balanced_low_auprc": values["source_balanced_low_auprc"],
                    "source_balanced_macro_f1": values["source_balanced_macro_f1"],
                    "source_balanced_brier": values["source_balanced_brier"],
                    "source_balanced_log_loss": values["source_balanced_log_loss"],
                    "equal_frequency_ece_10": values["equal_frequency_ece_10"],
                }
            )
    checkpoint_order = {name: index for index, name in enumerate(["t3", "t5", "t10", "t20", "full"])}
    frame = pd.DataFrame(rows)
    frame["checkpoint_order"] = frame["checkpoint"].map(checkpoint_order)
    return frame.sort_values(["model", "checkpoint_order"], kind="stable").drop(
        columns="checkpoint_order"
    )


def task_c_table() -> pd.DataFrame:
    baseline = _read_json(BASELINE_SUMMARY)["task_c_metrics"]
    neural = _read_json(QTRACE_SUMMARY)["task_c_metrics"]
    rows: list[dict[str, Any]] = []
    for model, values in {**baseline, **neural}.items():
        row = {
            "model": model,
            "display_name": MODEL_LABELS[model],
            "family": _family(model),
            "n_predictions": values["n_predictions"],
            "n_sources": values["n_sources"],
            "source_balanced_macro_f1": values["source_balanced_macro_f1"],
            "utterance_macro_f1": values["utterance_macro_f1"],
            "source_balanced_brier": values["source_balanced_brier"],
            "source_balanced_log_loss": values["source_balanced_log_loss"],
            "equal_frequency_ece_10": values["equal_frequency_ece_10"],
            "worst_20pct_source_log_loss_cvar": values[
                "worst_20pct_source_log_loss_cvar"
            ],
        }
        for label, metrics in values["per_class"].items():
            row[f"f1_{label}"] = metrics["f1"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("model", kind="stable")


def registered_inference_table() -> pd.DataFrame:
    summary = _read_json(QTRACE_SUMMARY)
    intervals = summary["paired_source_bootstrap"]["intervals"]
    task_a = summary["task_a_metrics"]
    task_c = summary["task_c_metrics"]
    specifications = [
        {
            "task": "A",
            "comparison": "qtrace_mi - a_only",
            "metric": "t10_source_balanced_balanced_accuracy",
            "favorable_direction": "positive",
            "point_delta": task_a["qtrace_mi"]["t10"][
                "source_balanced_balanced_accuracy"
            ]
            - task_a["a_only"]["t10"]["source_balanced_balanced_accuracy"],
            "interval_key": "task_a_t10_balanced_accuracy_delta",
            "registered_limit": 0.0,
            "passed": summary["candidate_success_gate"]["task_a_positive_interval"],
        },
        {
            "task": "C",
            "comparison": "qtrace_mi - c_only",
            "metric": "source_balanced_macro_f1",
            "favorable_direction": "positive",
            "point_delta": task_c["qtrace_mi"]["source_balanced_macro_f1"]
            - task_c["c_only"]["source_balanced_macro_f1"],
            "interval_key": "task_c_macro_f1_delta",
            "registered_limit": 0.02,
            "passed": summary["candidate_success_gate"]["task_c_minimum_delta"]
            and summary["candidate_success_gate"]["task_c_positive_interval"],
        },
        {
            "task": "C",
            "comparison": "qtrace_mi - c_only",
            "metric": "source_balanced_brier",
            "favorable_direction": "negative",
            "point_delta": task_c["qtrace_mi"]["source_balanced_brier"]
            - task_c["c_only"]["source_balanced_brier"],
            "interval_key": "task_c_brier_delta",
            "registered_limit": 0.01,
            "passed": summary["candidate_success_gate"]["task_c_brier_within_limit"],
        },
    ]
    rows = []
    for specification in specifications:
        interval = intervals[specification.pop("interval_key")]
        rows.append(
            {
                **specification,
                "bootstrap_mean_delta": interval["mean"],
                "ci95_low": interval["low"],
                "ci95_high": interval["high"],
            }
        )
    return pd.DataFrame(rows)


def prediction_set_table() -> pd.DataFrame:
    metrics = _read_json(QTRACE_SUMMARY)["prediction_set_metrics"]
    return pd.DataFrame(
        [
            {
                "model": model,
                "display_name": MODEL_LABELS[model],
                "source_balanced_coverage": values["source_balanced_coverage"],
                "source_balanced_mean_set_size": values[
                    "source_balanced_mean_set_size"
                ],
                "source_balanced_singleton_rate": values[
                    "source_balanced_singleton_rate"
                ],
                **{
                    f"coverage_{label}": coverage
                    for label, coverage in values["per_class_coverage"].items()
                },
            }
            for model, values in metrics.items()
        ]
    ).sort_values("model", kind="stable")


def _write_csv(path: Path, frame: pd.DataFrame) -> str:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, lineterminator="\n", float_format="%.10g")
    return write_create_only(path, buffer.getvalue().encode("utf-8"))


def _save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for extension in ("png", "svg"):
        buffer = io.BytesIO()
        metadata = {"Software": "annomi-qtrace-ac-v1"} if extension == "png" else {"Date": None}
        fig.savefig(
            buffer,
            format=extension,
            dpi=180,
            bbox_inches="tight",
            facecolor="white",
            metadata=metadata,
        )
        payload = buffer.getvalue()
        if extension == "svg":
            text = payload.decode("utf-8")
            payload = ("\n".join(line.rstrip() for line in text.splitlines()) + "\n").encode()
        hashes[extension] = write_create_only(ASSET_DIR / f"{stem}.{extension}", payload)
    plt.close(fig)
    return hashes


def _bar_panel(
    axis: plt.Axes,
    frame: pd.DataFrame,
    order: list[str],
    metric: str,
    title: str,
    xlabel: str,
) -> None:
    values = frame.set_index("model").loc[order]
    positions = np.arange(len(values))
    bars = axis.barh(
        positions,
        values[metric],
        color=[COLORS[family] for family in values["family"]],
        height=0.64,
    )
    for bar, family in zip(bars, values["family"], strict=True):
        if family == "oracle":
            bar.set_hatch("///")
            bar.set_edgecolor("#92400E")
    axis.set_yticks(positions, values["display_name"])
    axis.invert_yaxis()
    axis.set_xlim(0, max(0.8, float(values[metric].max()) * 1.16))
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def build_results_figure(task_a: pd.DataFrame, task_c: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), constrained_layout=True)
    _bar_panel(
        axes[0],
        task_a[task_a["checkpoint"].eq("t10")],
        [
            "class_prior",
            "tfidf_raw_prefix",
            "structure_only",
            "a_only",
            "joint_no_transition",
            "qtrace_mi",
            "oracle_gold_codes",
        ],
        "source_balanced_balanced_accuracy",
        "A  Quality after 10 therapist turns",
        "Source-balanced balanced accuracy",
    )
    _bar_panel(
        axes[1],
        task_c,
        [
            "class_prior",
            "tfidf_causal10",
            "joint_no_transition",
            "qtrace_mi",
            "oracle_gold_markov",
            "c_only",
        ],
        "source_balanced_macro_f1",
        "B  Next therapist-action forecasting",
        "Source-balanced macro-F1",
    )
    fig.suptitle(
        "Task-specific gains, but the registered Q-TRACE-MI joint gate failed",
        fontsize=14,
        fontweight="bold",
    )
    return fig


def build_interval_figure(inference: pd.DataFrame) -> plt.Figure:
    fig, axis = plt.subplots(figsize=(8.7, 3.7), constrained_layout=True)
    labels = [
        "Task A balanced accuracy",
        "Task C macro-F1",
        "Task C Brier (lower is better)",
    ]
    positions = np.arange(len(inference))
    colors = np.where(inference["passed"].astype(bool), "#0F766E", "#B91C1C")
    for position, (_, row), color in zip(positions, inference.iterrows(), colors, strict=True):
        axis.plot([row["ci95_low"], row["ci95_high"]], [position, position], color=color, lw=2.5)
        axis.scatter(row["point_delta"], position, color=color, s=48, zorder=3)
    axis.axvline(0, color="#111827", linewidth=1, linestyle="--")
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Q-TRACE-MI minus matched single-task model (paired-source 95% interval)")
    axis.set_title("Registered candidate effects")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    return fig


def main() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "svg.hashsalt": "annomi-qtrace-ac-publication-v1",
        }
    )
    task_a = task_a_table()
    task_c = task_c_table()
    inference = registered_inference_table()
    prediction_sets = prediction_set_table()
    table_hashes = {
        "task_a_summary.csv": _write_csv(TABLE_DIR / "task_a_summary.csv", task_a),
        "task_c_summary.csv": _write_csv(TABLE_DIR / "task_c_summary.csv", task_c),
        "registered_inference_summary.csv": _write_csv(
            TABLE_DIR / "registered_inference_summary.csv", inference
        ),
        "prediction_set_summary.csv": _write_csv(
            TABLE_DIR / "prediction_set_summary.csv", prediction_sets
        ),
    }
    figure_hashes = {
        "qtrace_ac_results": _save_figure(
            build_results_figure(task_a, task_c), "qtrace_ac_results"
        ),
        "qtrace_ac_intervals": _save_figure(
            build_interval_figure(inference), "qtrace_ac_intervals"
        ),
    }
    source_paths = [
        BASELINE_SUMMARY,
        QTRACE_SUMMARY,
        QTRACE_DIR / "bootstrap_draws.csv",
        QTRACE_DIR / "calibration.json",
    ]
    manifest = {
        "manifest_id": "annomi-qtrace-ac-publication-assets-v1",
        "source_sha256": {
            path.relative_to(ROOT).as_posix(): sha256_file(path) for path in source_paths
        },
        "table_sha256": table_hashes,
        "figure_sha256": figure_hashes,
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write_create_only(TABLE_DIR / "manifest.json", canonical_json_bytes(manifest))
    for filename, digest in table_hashes.items():
        print(f"Wrote/verified {filename}: {digest}")
    for stem, formats in figure_hashes.items():
        print(f"Wrote/verified {stem}: {formats}")


if __name__ == "__main__":
    main()
