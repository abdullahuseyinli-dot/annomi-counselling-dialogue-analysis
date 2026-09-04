from __future__ import annotations

import argparse
from pathlib import Path

from .audit import build_data_audit
from .baselines import run_baselines
from .constants import FULL_DATA, RESEARCH_RESULTS, SIMPLE_DATA
from .dash import run_dash_mi, run_dash_smoke
from .data import load_corpus
from .inference import DEFAULT_CONFIG, DEFAULT_OUTPUT, run_comparison
from .io import read_json, write_json_create_only
from .neural import run_environment_gate, run_neural, run_neural_smoke
from .splits import build_source_folds
from .validation import legacy_inventory, validate_research

GATE0 = RESEARCH_RESULTS / "gate0"
DEFAULT_SPLITS = GATE0 / "source_folds_v1.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="annomi-research")
    parser.add_argument("--simple-data", type=Path, default=SIMPLE_DATA)
    parser.add_argument("--full-data", type=Path, default=FULL_DATA)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit-data", help="Build the privacy-safe Gate 0 audit")
    subparsers.add_parser("make-splits", help="Freeze five source-disjoint outer folds")
    subparsers.add_parser("validate", help="Validate the research evidence contract")
    baseline = subparsers.add_parser("run-baselines", help="Run nested source-grouped baselines")
    baseline.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    subparsers.add_parser("check-neural-env", help="Verify and record the CUDA/BF16 gate")
    smoke = subparsers.add_parser("smoke-neural", help="Run the CUDA neural engineering gate")
    smoke.add_argument(
        "--model",
        choices=("roberta_utterance", "roberta_flat_causal10"),
        required=True,
    )
    smoke.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    neural = subparsers.add_parser("run-neural", help="Run nested source-grouped RoBERTa")
    neural.add_argument(
        "--model",
        choices=("roberta_utterance", "roberta_flat_causal10"),
        required=True,
    )
    neural.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    neural.add_argument("--output-dir", type=Path)
    dash_smoke = subparsers.add_parser(
        "smoke-dash", help="Run the CUDA DASH-MI engineering gate"
    )
    dash_smoke.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    dash = subparsers.add_parser(
        "run-dash", help="Run nested source-grouped DASH-MI"
    )
    dash.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    dash.add_argument("--output-dir", type=Path)
    comparison = subparsers.add_parser(
        "compare-models", help="Run the registered paired source bootstrap"
    )
    comparison.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    comparison.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    corpus = load_corpus(args.simple_data, args.full_data)
    if args.command == "audit-data":
        audit_hash = write_json_create_only(GATE0 / "data_audit.json", build_data_audit(corpus))
        inventory_hash = write_json_create_only(
            GATE0 / "legacy_inventory.json", legacy_inventory()
        )
        print(f"Wrote/verified Gate 0 audit: {audit_hash}")
        print(f"Wrote/verified legacy inventory: {inventory_hash}")
        return 0
    if args.command == "make-splits":
        digest = write_json_create_only(DEFAULT_SPLITS, build_source_folds(corpus))
        print(f"Wrote/verified source folds: {digest}")
        return 0
    if args.command == "validate":
        for check in validate_research(corpus):
            print(f"PASS  {check}")
        return 0
    if args.command == "run-baselines":
        result = run_baselines(corpus, read_json(args.splits))
        for model, metrics in result["metrics"].items():
            print(
                f"{model}: source-balanced macro-F1="
                f"{metrics['source_balanced_macro_f1']:.4f}; "
                f"ordinary macro-F1={metrics['utterance_macro_f1']:.4f}"
            )
        return 0
    if args.command == "check-neural-env":
        result = run_environment_gate()
        print(
            f"PASS CUDA/BF16 environment: {result['runtime_environment']['gpu']} "
            f"with torch {result['runtime_environment']['torch']}"
        )
        return 0
    if args.command == "smoke-neural":
        result = run_neural_smoke(corpus, read_json(args.splits), args.model)
        print(
            f"PASS {args.model} CUDA smoke: "
            f"peak={result['peak_memory_bytes'] / (1024**3):.2f} GiB; "
            f"steps={result['optimizer_steps']}"
        )
        return 0
    if args.command == "run-neural":
        result = run_neural(
            corpus,
            read_json(args.splits),
            args.model,
            output_dir=args.output_dir,
        )
        metrics = result["metrics"]["seed_ensemble"]
        print(
            f"{args.model}: source-balanced macro-F1="
            f"{metrics['source_balanced_macro_f1']:.4f}; "
            f"ordinary macro-F1={metrics['utterance_macro_f1']:.4f}"
        )
        return 0
    if args.command == "smoke-dash":
        result = run_dash_smoke(corpus, read_json(args.splits))
        print(
            f"PASS DASH-MI CUDA smoke: "
            f"peak={result['peak_memory_bytes'] / (1024**3):.2f} GiB; "
            f"steps={result['optimizer_steps']}"
        )
        return 0
    if args.command == "run-dash":
        result = run_dash_mi(
            corpus,
            read_json(args.splits),
            output_dir=args.output_dir,
        )
        metrics = result["metrics"]["seed_ensemble"]
        print(
            f"dash_mi: source-balanced macro-F1="
            f"{metrics['source_balanced_macro_f1']:.4f}; "
            f"ordinary macro-F1={metrics['utterance_macro_f1']:.4f}"
        )
        return 0
    if args.command == "compare-models":
        result = run_comparison(args.config, args.output_dir)
        delta = result["point_deltas_candidate_minus_baseline"][
            "source_balanced_macro_f1"
        ]
        interval = result["bootstrap"]["intervals"]["source_balanced_macro_f1"]
        print(
            f"paired source macro-F1 delta={delta:.4f}; "
            f"95% CI=[{interval['low']:.4f}, {interval['high']:.4f}]; "
            f"gate_pass={result['candidate_success_gate']['pass']}"
        )
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")
