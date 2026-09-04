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
TABLE_DIR = ROOT / "results" / "research" / "publication_safe_mi_v2"
QTRACE_SUMMARY = ROOT / "results" / "research" / "ac_v1" / "qtrace_mi" / "summary.json"
SAFE_SUMMARY = ROOT / "results" / "research" / "safe_mi_v2" / "summary.json"
EXTENSION_SUMMARY = ROOT / "results" / "research" / "safe_mi_v2_1" / "summary.json"
EXTERNAL_AUDIT = (
    ROOT / "results" / "research" / "mi_tags_external_v1" / "sample_overlap_audit.json"
)

DISPLAY_NAMES = {
    "a_only": "A-only neural",
    "joint_no_transition": "Joint, no transition",
    "qtrace_mi": "Q-TRACE-MI",
    "oracle_gold_codes": "Gold-code oracle",
    "c_only": "Earlier C-only",
    "c0_frozen_gru": "Frozen-GRU baseline",
    "c1_adapted_gru": "Adapted-GRU",
    "r1_prototype": "Safe prototype retrieval",
    "m1_shared": "Shared multitask",
    "m2_oneway": "One-way multitask",
    "m3_discounted": "Discounted one-way",
    "c2_frozen_attention": "Frozen attention",
    "c3_adapted_attention": "Adapted attention",
    "c4_adapted_attention_logit": "Adapted attention + logit",
    "t1_loss_only": "Transition loss only",
    "t2_safe_residual": "Zero-start safe residual",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_row(model: str, values: dict[str, Any], role: str) -> dict[str, Any]:
    return {
        "model": model,
        "display_name": DISPLAY_NAMES[model],
        "study_role": role,
        "n_predictions": values["n_predictions"],
        "n_sources": values["n_sources"],
        "source_balanced_macro_f1": values["source_balanced_macro_f1"],
        "utterance_macro_f1": values["utterance_macro_f1"],
        "source_balanced_brier": values["source_balanced_brier"],
        "source_balanced_log_loss": values["source_balanced_log_loss"],
        "equal_frequency_ece_10": values["equal_frequency_ece_10"],
        "seen_text_macro_f1": values["seen_text_macro_f1"],
        "unseen_text_macro_f1": values["unseen_text_macro_f1"],
    }


def task_a_table() -> pd.DataFrame:
    qtrace = _read_json(QTRACE_SUMMARY)["task_a_metrics"]
    extension = _read_json(EXTENSION_SUMMARY)["task_a_metrics"]
    specifications = [
        ("a_only", qtrace["a_only"]["t10"], "registered baseline"),
        (
            "joint_no_transition",
            qtrace["joint_no_transition"]["t10"],
            "registered ablation",
        ),
        ("qtrace_mi", qtrace["qtrace_mi"]["t10"], "registered candidate"),
        ("m2_oneway", extension["m2_oneway"]["t10"], "posthoc exploratory candidate"),
    ]
    rows = []
    for model, values, role in specifications:
        rows.append(
            {
                "model": model,
                "display_name": DISPLAY_NAMES[model],
                "study_role": role,
                "checkpoint": "t10",
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
    return pd.DataFrame(rows)


def task_c_table() -> pd.DataFrame:
    qtrace = _read_json(QTRACE_SUMMARY)["task_c_metrics"]
    safe = _read_json(SAFE_SUMMARY)["final"]["task_c_metrics"]
    extension = _read_json(EXTENSION_SUMMARY)["task_c_metrics"]
    specifications = [
        ("c_only", qtrace["c_only"], "earlier registered baseline"),
        ("c0_frozen_gru", safe["c0_frozen_gru"], "matched exploratory baseline"),
        ("c1_adapted_gru", extension["c1_adapted_gru"], "posthoc exploratory baseline"),
        ("r1_prototype", safe["r1_prototype"], "exploratory candidate"),
        ("m2_oneway", extension["m2_oneway"], "posthoc exploratory candidate"),
        ("m3_discounted", safe["m3_discounted"], "exploratory candidate"),
    ]
    return pd.DataFrame([_metric_row(model, values, role) for model, values, role in specifications])


def screen_table() -> pd.DataFrame:
    screen = _read_json(SAFE_SUMMARY)["screen"]
    rows = []
    for model, values in screen["task_c_metrics"].items():
        task_a = screen["task_a_metrics"].get(model, {}).get("t10", {})
        rows.append(
            {
                "model": model,
                "display_name": DISPLAY_NAMES[model],
                "source_balanced_task_c_macro_f1": values[
                    "source_balanced_macro_f1"
                ],
                "source_balanced_task_c_brier": values["source_balanced_brier"],
                "source_balanced_task_c_log_loss": values[
                    "source_balanced_log_loss"
                ],
                "source_balanced_task_a_t10_balanced_accuracy": task_a.get(
                    "source_balanced_balanced_accuracy"
                ),
                "source_balanced_task_a_t10_brier": task_a.get("source_balanced_brier"),
            }
        )
    return pd.DataFrame(rows)


def inference_table() -> pd.DataFrame:
    safe = _read_json(SAFE_SUMMARY)["final"]
    extension = _read_json(EXTENSION_SUMMARY)
    rows: list[dict[str, Any]] = []

    def add(
        *,
        stage: str,
        task: str,
        candidate: str,
        baseline: str,
        metric: str,
        point_delta: float,
        interval: dict[str, float],
        favorable_direction: str,
    ) -> None:
        favorable_interval = (
            interval["low"] > 0
            if favorable_direction == "positive"
            else interval["high"] < 0
        )
        rows.append(
            {
                "stage": stage,
                "task": task,
                "comparison": f"{candidate} - {baseline}",
                "metric": metric,
                "favorable_direction": favorable_direction,
                "point_delta": point_delta,
                "bootstrap_mean_delta": interval["mean"],
                "ci95_low": interval["low"],
                "ci95_high": interval["high"],
                "interval_favorable": favorable_interval,
            }
        )

    for candidate in ("r1_prototype", "m3_discounted"):
        gate = safe["exploratory_gates"][candidate]
        intervals = safe["paired_source_bootstrap"][candidate]["intervals"]
        add(
            stage="SAFE-MI v2 exploratory",
            task="C",
            candidate=candidate,
            baseline="c0_frozen_gru",
            metric="source_balanced_macro_f1",
            point_delta=gate["task_c_macro_f1_delta"],
            interval=intervals["task_c_macro_f1_delta"],
            favorable_direction="positive",
        )
        add(
            stage="SAFE-MI v2 exploratory",
            task="C",
            candidate=candidate,
            baseline="c0_frozen_gru",
            metric="source_balanced_brier",
            point_delta=gate["task_c_brier_delta"],
            interval=intervals["task_c_brier_delta"],
            favorable_direction="negative",
        )
    m3_gate = safe["exploratory_gates"]["m3_discounted"]
    m3_intervals = safe["paired_source_bootstrap"]["m3_discounted"]["intervals"]
    for metric, key, direction in (
        ("t10_source_balanced_balanced_accuracy", "task_a_t10_balanced_accuracy_delta", "positive"),
        ("t10_source_balanced_brier", "task_a_t10_brier_delta", "negative"),
    ):
        add(
            stage="SAFE-MI v2 exploratory",
            task="A",
            candidate="m3_discounted",
            baseline="a_only",
            metric=metric,
            point_delta=m3_gate[key],
            interval=m3_intervals[key],
            favorable_direction=direction,
        )

    for candidate in ("c1_adapted_gru", "m2_oneway"):
        gate = extension["descriptive_gates"][candidate]
        intervals = extension["paired_source_bootstrap"][candidate]["intervals"]
        add(
            stage="SAFE-MI v2.1 posthoc",
            task="C",
            candidate=candidate,
            baseline="c0_frozen_gru",
            metric="source_balanced_macro_f1",
            point_delta=gate["task_c_macro_f1_delta"],
            interval=intervals["task_c_macro_f1_delta"],
            favorable_direction="positive",
        )
        add(
            stage="SAFE-MI v2.1 posthoc",
            task="C",
            candidate=candidate,
            baseline="c0_frozen_gru",
            metric="source_balanced_brier",
            point_delta=gate["task_c_brier_delta"],
            interval=intervals["task_c_brier_delta"],
            favorable_direction="negative",
        )
    m2_gate = extension["descriptive_gates"]["m2_oneway"]
    m2_intervals = extension["paired_source_bootstrap"]["m2_oneway"]["intervals"]
    for metric, key, direction in (
        ("t10_source_balanced_balanced_accuracy", "task_a_t10_balanced_accuracy_delta", "positive"),
        ("t10_source_balanced_brier", "task_a_t10_brier_delta", "negative"),
    ):
        add(
            stage="SAFE-MI v2.1 posthoc",
            task="A",
            candidate="m2_oneway",
            baseline="a_only",
            metric=metric,
            point_delta=m2_gate[key],
            interval=m2_intervals[key],
            favorable_direction=direction,
        )
    return pd.DataFrame(rows)


def prediction_set_table() -> pd.DataFrame:
    safe_sets = _read_json(SAFE_SUMMARY)["final"]["prediction_set_metrics"]
    extension = _read_json(EXTENSION_SUMMARY)
    rows = []

    def append(model: str, scheme: str, values: dict[str, Any]) -> None:
        rows.append(
            {
                "model": model,
                "display_name": DISPLAY_NAMES[model],
                "calibration_scheme": scheme,
                "n_decisions": values["n_decisions"],
                "n_sources": values["n_sources"],
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
        )

    for model, values in safe_sets.items():
        append(model, "registered split-source", values)
    for model in ("c1_adapted_gru", "m2_oneway"):
        append(model, "posthoc split-source", extension["split_prediction_set_metrics"][model])
    for model, values in extension["crossfit_prediction_set_metrics"].items():
        append(model, "posthoc outer-crossfit", values)
    return pd.DataFrame(rows)


def external_audit_table() -> pd.DataFrame:
    audit = _read_json(EXTERNAL_AUDIT)
    summary = audit["summary"]
    return pd.DataFrame(
        [
            {
                "audit_id": audit["audit_id"],
                "upstream_commit": audit["upstream_commit"],
                "sample_records": summary["sample_records"],
                "quarantined_records": summary["quarantined_records"],
                "clear_records": summary["clear_records"],
                "locked_test_records_before_quarantine": summary[
                    "partition_counts_before_quarantine"
                ]["test"],
                "minimum_test_groups_required": summary["minimum_test_groups_required"],
                "sample_sufficient_for_external_evaluation": summary[
                    "sample_sufficient_for_external_evaluation"
                ],
                "performance_claim_permitted": audit["performance_claim_permitted"],
                "status": audit["status"],
            }
        ]
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> str:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, lineterminator="\n", float_format="%.10g")
    return write_create_only(path, buffer.getvalue().encode("utf-8"))


def _save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    hashes = {}
    for extension in ("png", "svg"):
        buffer = io.BytesIO()
        metadata = {"Software": "annomi-safe-mi-v2"} if extension == "png" else {"Date": None}
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


def build_results_figure(task_a: pd.DataFrame, task_c: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True)
    panels = [
        (
            axes[0],
            task_a,
            "source_balanced_balanced_accuracy",
            "A  Quality after 10 therapist turns",
            "Source-balanced balanced accuracy",
        ),
        (
            axes[1],
            task_c,
            "source_balanced_macro_f1",
            "B  Strict next-action forecasting",
            "Source-balanced macro-F1",
        ),
    ]
    for axis, frame, metric, title, xlabel in panels:
        positions = np.arange(len(frame))
        colors = [
            "#D97706" if model == "m2_oneway" else "#0F766E" if "safe" in role or "exploratory" in role else "#64748B"
            for model, role in zip(frame["model"], frame["study_role"], strict=True)
        ]
        bars = axis.barh(positions, frame[metric], color=colors, height=0.64)
        axis.set_yticks(positions, frame["display_name"])
        axis.invert_yaxis()
        axis.set_xlim(0, max(float(frame[metric].max()) * 1.19, 0.5))
        axis.set_xlabel(xlabel)
        axis.set_title(title)
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "SAFE-MI raises the Task A point estimate; Task C candidates do not beat their baseline",
        fontsize=13.5,
        fontweight="bold",
    )
    return fig


def build_interval_figure(inference: pd.DataFrame) -> plt.Figure:
    selected = inference[inference["metric"].isin(
        ["source_balanced_macro_f1", "t10_source_balanced_balanced_accuracy"]
    )].copy()
    selected["label"] = selected.apply(
        lambda row: f"Task {row['task']}: {DISPLAY_NAMES[row['comparison'].split(' - ')[0]]}",
        axis=1,
    )
    fig, axis = plt.subplots(figsize=(9.4, 5.2), constrained_layout=True)
    positions = np.arange(len(selected))
    colors = ["#0F766E" if task == "C" else "#D97706" for task in selected["task"]]
    for position, (_, row), color in zip(positions, selected.iterrows(), colors, strict=True):
        axis.plot([row["ci95_low"], row["ci95_high"]], [position, position], color=color, lw=2.4)
        axis.scatter(row["point_delta"], position, color=color, s=46, zorder=3)
    axis.axvline(0, color="#111827", linewidth=1, linestyle="--")
    axis.set_yticks(positions, selected["label"])
    axis.invert_yaxis()
    axis.set_xlabel("Candidate minus matched baseline (paired-source 95% interval)")
    axis.set_title("Exploratory and post-hoc primary-score effects")
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
            "svg.hashsalt": "annomi-safe-mi-publication-v2",
        }
    )
    task_a = task_a_table()
    task_c = task_c_table()
    inference = inference_table()
    tables = {
        "task_a_t10_summary.csv": task_a,
        "task_c_summary.csv": task_c,
        "screen_summary.csv": screen_table(),
        "paired_inference_summary.csv": inference,
        "prediction_set_summary.csv": prediction_set_table(),
        "external_overlap_summary.csv": external_audit_table(),
    }
    table_hashes = {
        filename: _write_csv(TABLE_DIR / filename, frame) for filename, frame in tables.items()
    }
    figure_hashes = {
        "safe_mi_results": _save_figure(
            build_results_figure(task_a, task_c), "safe_mi_results"
        ),
        "safe_mi_effect_intervals": _save_figure(
            build_interval_figure(inference), "safe_mi_effect_intervals"
        ),
    }
    source_paths = [
        QTRACE_SUMMARY,
        SAFE_SUMMARY,
        EXTENSION_SUMMARY,
        EXTERNAL_AUDIT,
        ROOT / "configs" / "research" / "protocol_safe_mi_v2.json",
        ROOT / "configs" / "research" / "protocol_safe_mi_v2_1.json",
        ROOT / "configs" / "research" / "protocol_mi_tags_external_v1.json",
    ]
    manifest = {
        "manifest_id": "annomi-safe-mi-publication-assets-v2",
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
