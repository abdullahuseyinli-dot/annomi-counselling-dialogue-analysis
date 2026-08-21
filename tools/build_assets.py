from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ASSETS = ROOT / "assets"
BLUE = "#2563EB"
GREEN = "#059669"
SLATE = "#64748B"
TEXT = "#172033"


def configure() -> None:
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelcolor": TEXT,
            "text.color": TEXT,
            "font.size": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS / f"{name}.png", dpi=180, bbox_inches="tight")
    svg_path = ASSETS / f"{name}.svg"
    fig.savefig(
        svg_path,
        bbox_inches="tight",
        metadata={"Date": None},
    )
    svg = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    plt.close(fig)


def model_comparison() -> None:
    frame = pd.read_csv(RESULTS / "main" / "model_comparison.csv")
    labels = ["Elastic-net", "RoBERTa-base"]
    metrics = [("accuracy", "Accuracy"), ("f1_macro", "Macro-F1")]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = range(len(labels))
    width = 0.34
    for offset, (column, label), color in zip(
        (-width / 2, width / 2), metrics, (SLATE, BLUE), strict=True
    ):
        values = frame[column].mul(100)
        bars = ax.bar(
            [position + offset for position in x], values, width, label=label, color=color
        )
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Held-out score")
    ax.set_ylim(0, 90)
    ax.set_title("Context-aware RoBERTa improves held-out classification")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    fig.tight_layout()
    save(fig, "model_comparison")


def per_class_f1() -> None:
    frame = pd.read_csv(RESULTS / "main" / "per_class_metrics.csv")
    frame = frame[frame["metric"] == "f1"].copy()
    frame["model"] = frame["model"].map(
        lambda value: "RoBERTa-base" if "roberta-base" in value else "Elastic-net"
    )
    order = ["reflection", "question", "therapist_input", "other"]
    pivot = frame.pivot(index="label", columns="model", values="score").reindex(order)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    pivot[["Elastic-net", "RoBERTa-base"]].mul(100).plot.bar(ax=ax, color=[SLATE, BLUE], width=0.72)
    ax.set_xticklabels(["Reflection", "Question", "Therapist input", "Other"], rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 100)
    ax.set_title("Held-out F1 by therapist-behaviour class")
    ax.legend(title="", frameon=False)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    fig.tight_layout()
    save(fig, "per_class_f1")


def significance_intervals() -> None:
    frame = pd.read_csv(RESULTS / "main" / "grouped_significance.csv")
    frame["label"] = frame["metric"].map({"accuracy": "Accuracy", "f1_macro": "Macro-F1"})
    center = frame["delta_roberta_minus_baseline"].mul(100)
    low = frame["bootstrap_ci95_low"].mul(100)
    high = frame["bootstrap_ci95_high"].mul(100)
    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    ax.errorbar(
        center,
        frame["label"],
        xerr=[center - low, high - center],
        fmt="o",
        color=BLUE,
        ecolor=BLUE,
        capsize=5,
        markersize=7,
    )
    ax.axvline(0, color=SLATE, linewidth=1, linestyle="--")
    ax.set_xlabel("RoBERTa gain (percentage points), transcript-grouped 95% interval")
    ax.set_title("Both primary gains remain positive under grouped resampling")
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.8)
    fig.tight_layout()
    save(fig, "significance_intervals")


def summarisation_comparison() -> None:
    frame = pd.read_csv(RESULTS / "summarisation" / "method_means.csv").set_index("Method")
    columns = [
        "Faithfulness / Support",
        "Coverage",
        "Specificity",
        "Non-redundancy",
        "Overall Usefulness",
    ]
    display = frame[columns].T
    display.columns = ["BERTopic + MMR", "KMeans + MMR"]
    fig, ax = plt.subplots(figsize=(9.2, 5.1))
    display[["KMeans + MMR", "BERTopic + MMR"]].plot.barh(ax=ax, color=[BLUE, GREEN], width=0.72)
    ax.set_xlim(0, 4)
    ax.set_xlabel("Aggregate rubric score")
    ax.set_ylabel("")
    ax.set_title("Summary methods trade faithfulness for specificity")
    ax.legend(title="", frameon=False, loc="lower right")
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.8)
    fig.tight_layout()
    save(fig, "summarisation_comparison")


def main() -> None:
    configure()
    model_comparison()
    per_class_f1()
    significance_intervals()
    summarisation_comparison()
    print("Built four chart pairs in assets/.")


if __name__ == "__main__":
    main()
