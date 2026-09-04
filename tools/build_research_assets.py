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
TABLE_DIR = ROOT / "results" / "research" / "publication_v1"
PANEL_SUMMARY = ROOT / "results" / "research" / "multiannotator_v1" / "panel_mi" / "summary.json"

MODEL_LABELS = {
    "tfidf_elasticnet_utterance": "TF-IDF",
    "roberta_utterance": "RoBERTa target",
    "roberta_flat_causal10": "RoBERTa causal",
    "dash_mi": "DASH-MI",
    "transcript_balanced_prior": "Transcript prior",
    "hard_linear": "Hard linear",
    "soft_linear": "Soft linear",
    "panel_mi": "PANEL-MI",
}
COLORS = {
    "TF-IDF": "#6B7280",
    "RoBERTa target": "#2563EB",
    "RoBERTa causal": "#0F766E",
    "DASH-MI": "#7C3AED",
    "Transcript prior": "#9CA3AF",
    "Hard linear": "#D97706",
    "Soft linear": "#2563EB",
    "PANEL-MI": "#7C3AED",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def classification_table() -> pd.DataFrame:
    baseline = _read_json(ROOT / "results" / "research" / "baseline_v1" / "summary.json")
    locations = {
        "roberta_utterance": ROOT
        / "results"
        / "research"
        / "neural_v1"
        / "roberta_utterance"
        / "summary.json",
        "roberta_flat_causal10": ROOT
        / "results"
        / "research"
        / "neural_v1"
        / "roberta_flat_causal10"
        / "summary.json",
        "dash_mi": ROOT / "results" / "research" / "neural_v1" / "dash_mi" / "summary.json",
    }
    metrics = {"tfidf_elasticnet_utterance": baseline["metrics"]["tfidf_elasticnet_utterance"]}
    for model, path in locations.items():
        metrics[model] = _read_json(path)["metrics"]["seed_ensemble"]
    rows = []
    for model, values in metrics.items():
        rows.append(
            {
                "model": model,
                "display_name": MODEL_LABELS[model],
                "source_balanced_macro_f1": values["source_balanced_macro_f1"],
                "utterance_macro_f1": values["utterance_macro_f1"],
                "source_balanced_brier": values["source_balanced_brier"],
                "source_balanced_log_loss": values["source_balanced_log_loss"],
                "equal_frequency_ece_10": values["equal_frequency_ece_10"],
                "worst_20pct_source_log_loss_cvar": values["worst_20pct_source_log_loss_cvar"],
            }
        )
    return pd.DataFrame(rows)


def multiannotator_table(panel_summary: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for task, models in panel_summary["metrics"]["seed_ensemble"].items():
        for model, values in models.items():
            rows.append(
                {
                    "task": task,
                    "model": model,
                    "display_name": MODEL_LABELS[model],
                    "transcript_balanced_vote_log_score": values[
                        "transcript_balanced_vote_log_score"
                    ],
                    "transcript_balanced_vote_brier": values["transcript_balanced_vote_brier"],
                    "transcript_balanced_jensen_shannon_divergence": values[
                        "transcript_balanced_jensen_shannon_divergence"
                    ],
                    "transcript_balanced_plurality_macro_f1": values[
                        "transcript_balanced_plurality_macro_f1"
                    ],
                    "vote_entropy_prediction_spearman": values["vote_entropy_prediction_spearman"],
                    "vote_entropy_mean_absolute_error": values["vote_entropy_mean_absolute_error"],
                }
            )
    return pd.DataFrame(rows).sort_values(["task", "model"], kind="stable")


def inference_table(panel_summary: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    classification_specs = [
        (
            "target RoBERTa - TF-IDF",
            "roberta_utterance_vs_tfidf_v1",
        ),
        (
            "causal RoBERTa - target RoBERTa",
            "roberta_causal10_vs_utterance_v1",
        ),
        (
            "DASH-MI - causal RoBERTa",
            "dash_mi_vs_causal10_v1",
        ),
    ]
    for label, directory in classification_specs:
        summary = _read_json(
            ROOT / "results" / "research" / "comparisons" / directory / "summary.json"
        )
        interval = summary["bootstrap"]["intervals"]["source_balanced_macro_f1"]
        delta = summary["point_deltas_candidate_minus_baseline"]["source_balanced_macro_f1"]
        rows.append(
            {
                "study": "classification",
                "task": "therapist",
                "comparison": label,
                "metric": "source_balanced_macro_f1",
                "favorable_direction": "positive",
                "delta": delta,
                "ci_low": interval["low"],
                "ci_high": interval["high"],
                "interval_excludes_zero_in_favorable_direction": interval["low"] > 0,
                "exact_one_sided_sign_flip_p": np.nan,
                "improved_clusters": np.nan,
                "n_clusters": summary["candidate_metrics"]["n_sources"],
            }
        )

    selected_panel_comparisons = [
        "therapist:soft_linear_vs_transcript_balanced_prior",
        "therapist:panel_mi_vs_soft_linear",
        "client:soft_linear_vs_transcript_balanced_prior",
        "client:panel_mi_vs_soft_linear",
    ]
    for key in selected_panel_comparisons:
        result = panel_summary["inference"]["comparisons"][key]
        interval = result["cluster_bootstrap_intervals"]["vote_log_score"]
        rows.append(
            {
                "study": "multiannotator",
                "task": result["task"],
                "comparison": key.split(":", maxsplit=1)[1].replace("_vs_", " - "),
                "metric": "transcript_balanced_vote_log_score",
                "favorable_direction": "negative",
                "delta": result["point_deltas"]["vote_log_score"],
                "ci_low": interval["low"],
                "ci_high": interval["high"],
                "interval_excludes_zero_in_favorable_direction": interval["high"] < 0,
                "exact_one_sided_sign_flip_p": result["exact_one_sided_sign_flip_p_vote_log_score"],
                "improved_clusters": result["improved_transcripts_vote_log_score"],
                "n_clusters": result["n_transcript_clusters"],
            }
        )
    return pd.DataFrame(rows)


def _write_csv(path: Path, frame: pd.DataFrame) -> str:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, lineterminator="\n", float_format="%.10g")
    return write_create_only(path, buffer.getvalue().encode("utf-8"))


def _save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for extension in ("png", "svg"):
        buffer = io.BytesIO()
        metadata = {"Software": "annomi-research-v1"} if extension == "png" else {"Date": None}
        fig.savefig(
            buffer,
            format=extension,
            dpi=180,
            bbox_inches="tight",
            facecolor="white",
            metadata=metadata,
        )
        hashes[extension] = write_create_only(ASSET_DIR / f"{stem}.{extension}", buffer.getvalue())
    plt.close(fig)
    return hashes


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)


def build_overview_figure(
    classification: pd.DataFrame,
    multiannotator: pd.DataFrame,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4), constrained_layout=True)
    ordered_classification = classification.set_index("display_name").loc[
        ["TF-IDF", "RoBERTa target", "RoBERTa causal", "DASH-MI"]
    ]
    axis = axes[0]
    names = ordered_classification.index.tolist()
    values = ordered_classification["source_balanced_macro_f1"].to_numpy()
    bars = axis.barh(
        names,
        values,
        color=[COLORS[name] for name in names],
        height=0.62,
    )
    axis.invert_yaxis()
    axis.set_xlim(0.68, 0.83)
    axis.set_xlabel("Source-balanced macro-F1 ↑")
    axis.set_title("A  Therapist coding")
    axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    _style_axis(axis)

    for axis, task, title in zip(
        axes[1:],
        ("therapist", "client"),
        ("B  Therapist vote distribution", "C  Client vote distribution"),
        strict=True,
    ):
        subset = (
            multiannotator[multiannotator["task"].eq(task)]
            .set_index("display_name")
            .loc[["Transcript prior", "Hard linear", "Soft linear", "PANEL-MI"]]
        )
        names = subset.index.tolist()
        values = subset["transcript_balanced_vote_log_score"].to_numpy()
        bars = axis.barh(
            names,
            values,
            color=[COLORS[name] for name in names],
            height=0.62,
        )
        axis.invert_yaxis()
        axis.set_xlabel("Transcript-balanced vote log score ↓")
        axis.set_title(title)
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        axis.set_xlim(0, max(values) * 1.14)
        _style_axis(axis)
    fig.suptitle("Leakage-controlled AnnoMI results", fontsize=14, fontweight="semibold")
    return fig


def _interval_panel(
    axis: plt.Axes,
    rows: pd.DataFrame,
    labels: list[str],
    title: str,
    xlabel: str,
) -> None:
    positions = np.arange(len(rows))
    supported = rows["interval_excludes_zero_in_favorable_direction"].astype(bool).to_numpy()
    colors = np.where(supported, "#0F766E", "#6B7280")
    for position, (_, row), color in zip(positions, rows.iterrows(), colors, strict=True):
        axis.plot([row["ci_low"], row["ci_high"]], [position, position], color=color, lw=2.4)
        axis.scatter(row["delta"], position, color=color, s=42, zorder=3)
    axis.axvline(0, color="#111827", lw=1, linestyle="--")
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)


def build_interval_figure(inference: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.0), constrained_layout=True)
    classification = inference[inference["study"].eq("classification")]
    _interval_panel(
        axes[0],
        classification,
        ["Target − TF-IDF", "Causal − target", "DASH-MI − causal"],
        "A  Therapist macro-F1 contrasts",
        "Paired source-bootstrap delta (positive favors candidate)",
    )
    multiannotator = inference[inference["study"].eq("multiannotator")]
    _interval_panel(
        axes[1],
        multiannotator,
        [
            "Therapist: soft − prior",
            "Therapist: panel − soft",
            "Client: soft − prior",
            "Client: panel − soft",
        ],
        "B  Vote-distribution log-score contrasts",
        "Paired transcript-bootstrap delta (negative favors candidate)",
    )
    fig.suptitle("Registered effect estimates with 95% cluster intervals", fontsize=14)
    return fig


def main() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "semibold",
            "svg.hashsalt": "annomi-research-publication-v1",
        }
    )
    panel_summary = _read_json(PANEL_SUMMARY)
    classification = classification_table()
    multiannotator = multiannotator_table(panel_summary)
    inference = inference_table(panel_summary)

    table_hashes = {
        "classification_summary.csv": _write_csv(
            TABLE_DIR / "classification_summary.csv", classification
        ),
        "multiannotator_summary.csv": _write_csv(
            TABLE_DIR / "multiannotator_summary.csv", multiannotator
        ),
        "registered_inference_summary.csv": _write_csv(
            TABLE_DIR / "registered_inference_summary.csv", inference
        ),
    }
    figure_hashes = {
        "research_overview": _save_figure(
            build_overview_figure(classification, multiannotator), "research_overview"
        ),
        "registered_effect_intervals": _save_figure(
            build_interval_figure(inference), "registered_effect_intervals"
        ),
    }
    source_paths = [
        ROOT / "results" / "research" / "baseline_v1" / "summary.json",
        ROOT / "results" / "research" / "neural_v1" / "roberta_utterance" / "summary.json",
        ROOT / "results" / "research" / "neural_v1" / "roberta_flat_causal10" / "summary.json",
        ROOT / "results" / "research" / "neural_v1" / "dash_mi" / "summary.json",
        PANEL_SUMMARY,
    ]
    manifest = {
        "manifest_id": "annomi-research-publication-assets-v1",
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
