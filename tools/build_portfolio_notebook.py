from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "annomi_counselling_dialogue_analysis.ipynb"


def build() -> None:
    cells = [
        new_markdown_cell(
            """# AnnoMI counselling dialogue analysis

This notebook is the compact, executable view of the repository's verified evidence. It
loads only tracked aggregate results; raw counselling text and model checkpoints are not
required."""
        ),
        new_code_cell(
            """from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

ROOT = Path.cwd()
if not (ROOT / "results").is_dir():
    raise RuntimeError("Run this notebook from the repository root.")
sys.path.insert(0, str(ROOT / "src"))
from annomi_portfolio.evidence import validate_evidence

pd.set_option("display.max_columns", 30)
pd.set_option("display.precision", 4)
plt.style.use("seaborn-v0_8-whitegrid")"""
        ),
        new_markdown_cell(
            """## Data and split contract

AnnoMI-simple contains 9,699 utterances from 133 transcripts. The primary classifier uses
102 transcripts for training and reserves 31 entire transcripts for final evaluation, preventing
utterances from the same dialogue from crossing the boundary."""
        ),
        new_code_cell(
            """manifest = json.loads((ROOT / "data/source_manifest.json").read_text())
split = json.loads((ROOT / "results/protocol/official_split.json").read_text())
display(pd.DataFrame({
    "Dataset": [manifest["dataset"]],
    "Utterances": [manifest["rows"]],
    "Transcripts": [manifest["transcripts"]],
    "Train transcripts": [split["n_train_transcripts"]],
    "Held-out transcripts": [split["n_test_transcripts"]],
}))"""
        ),
        new_markdown_cell(
            """## Primary comparison

The context-aware RoBERTa model improves both accuracy and macro-F1 over the sparse
elastic-net baseline on 973 therapist turns from unseen transcripts."""
        ),
        new_code_cell(
            """comparison = pd.read_csv(ROOT / "results/main/model_comparison.csv")
headline = comparison[["model", "accuracy", "f1_macro", "f1_weighted", "brier_multiclass"]].copy()
headline.columns = ["Model", "Accuracy", "Macro-F1", "Weighted F1", "Brier"]
display(headline.style.format({
    "Accuracy": "{:.2%}", "Macro-F1": "{:.2%}", "Weighted F1": "{:.2%}", "Brier": "{:.4f}"
}).highlight_max(subset=["Accuracy", "Macro-F1", "Weighted F1"], color="#DCFCE7")
  .highlight_min(subset=["Brier"], color="#DCFCE7"))"""
        ),
        new_markdown_cell("![Held-out model comparison](assets/model_comparison.png)"),
        new_markdown_cell(
            """## Grouped uncertainty and paired correctness

Resampling is performed at transcript level. Both 95% gain intervals exclude zero, and exact
McNemar testing isolates the paired item-level difference."""
        ),
        new_code_cell(
            """significance = pd.read_csv(ROOT / "results/main/grouped_significance.csv")
sig_view = significance[[
    "metric", "delta_roberta_minus_baseline", "bootstrap_ci95_low", "bootstrap_ci95_high",
    "permutation_p_two_sided"
]].copy()
sig_view.columns = ["Metric", "Gain", "95% low", "95% high", "Two-sided p"]
display(sig_view.style.format({
    "Gain": "{:+.2%}", "95% low": "{:+.2%}", "95% high": "{:+.2%}", "Two-sided p": "{:.4f}"
}))

mcnemar = pd.read_csv(ROOT / "results/main/mcnemar_summary.csv")
display(mcnemar[mcnemar["Measure"].isin([
    "Held-out items", "Discordant pairs", "RoBERTa-only correct", "Baseline-only correct", "Exact p-value"
])])"""
        ),
        new_markdown_cell("![Grouped confidence intervals](assets/significance_intervals.png)"),
        new_markdown_cell(
            """## Class-level performance

Per-class metrics keep minority behaviour performance visible instead of relying only on a
single aggregate score."""
        ),
        new_code_cell(
            """per_class = pd.read_csv(ROOT / "results/main/per_class_metrics.csv")
f1_by_class = per_class[per_class["metric"] == "f1"].pivot(
    index="label", columns="model", values="score"
)
display(f1_by_class.style.format("{:.2%}").highlight_max(axis=1, color="#DCFCE7"))"""
        ),
        new_markdown_cell("![Per-class F1](assets/per_class_f1.png)"),
        new_markdown_cell(
            """## Probability calibration

Temperature scaling improves RoBERTa's Brier score and expected calibration error while leaving
top-1 predictions unchanged. The sparse baseline still has the lowest ECE."""
        ),
        new_code_cell(
            """calibration = pd.read_csv(ROOT / "results/main/calibration.csv")
display(calibration[["model", "accuracy", "f1_macro", "brier_multiclass", "ece"]]
        .style.format({"accuracy": "{:.2%}", "f1_macro": "{:.2%}",
                       "brier_multiclass": "{:.4f}", "ece": "{:.4f}"}))"""
        ),
        new_markdown_cell(
            """## Extractive summarisation

The aggregate evidence selects **KMeans + MMR** on overall usefulness (2.790 versus 2.445).
BERTopic + MMR is stronger on average faithfulness/support and coverage, but its lower specificity
reduces the combined score."""
        ),
        new_code_cell(
            """summary_methods = pd.read_csv(ROOT / "results/summarisation/method_means.csv")
rubric_columns = ["Method", "Faithfulness / Support", "Coverage", "Specificity",
                  "Non-redundancy", "Overall Usefulness"]
display(summary_methods[rubric_columns].style.format({column: "{:.3f}" for column in rubric_columns[1:]})
        .highlight_max(subset=rubric_columns[1:], color="#DCFCE7"))"""
        ),
        new_markdown_cell("![Summarisation comparison](assets/summarisation_comparison.png)"),
        new_markdown_cell(
            """## Extensions

Transcript-quality classification and next-behaviour forecasting are separate experiments. They
broaden the analysis but do not participate in selection of the primary classifier."""
        ),
        new_code_cell(
            """quality = json.loads((ROOT / "results/extensions/transcript_classification.json").read_text())
quality_view = pd.DataFrame.from_dict(quality, orient="index")[["accuracy", "f1", "roc_auc"]]
quality_view.index.name = "Model"
forecast = pd.read_csv(ROOT / "results/extensions/next_behaviour_forecasting_topk.csv")
display(quality_view.style.format("{:.3f}"))
display(forecast.style.format({"top_1_accuracy": "{:.2%}", "top_2_accuracy": "{:.2%}",
                               "top_3_accuracy": "{:.2%}"}))"""
        ),
        new_markdown_cell(
            """## Evidence validation

The final cell rechecks headline values, arithmetic deltas, grouped intervals, paired counts,
calibration, summarisation ranking, split integrity, and dataset provenance."""
        ),
        new_code_cell(
            """checks = validate_evidence(ROOT)
for check in checks:
    print(f"PASS  {check}")"""
        ),
        new_markdown_cell(
            """## Scope

These results support method comparison on the pinned benchmark. They do not establish clinical
validity, demographic fairness, or safe use in patient-facing systems. See
`docs/MODEL_CARD.md` for the full limitations statement."""
        ),
    ]

    notebook = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
    )
    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    nbformat.write(notebook, OUTPUT)
    print(f"Built and executed {OUTPUT.name}")


if __name__ == "__main__":
    build()
