from __future__ import annotations

import argparse
from pathlib import Path

from .ac_baselines import run_ac_baselines
from .audit import build_data_audit
from .baselines import run_baselines
from .constants import FULL_DATA, RESEARCH_RESULTS, SIMPLE_DATA
from .data import load_corpus
from .inference import DEFAULT_CONFIG, DEFAULT_OUTPUT, run_comparison
from .io import read_json, write_json_create_only
from .mi_tags_external import run_mi_tags_sample_audit
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
    subparsers.add_parser(
        "audit-mi-tags", help="Audit the pinned official MI-TAGS public samples for overlap"
    )
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
    dash_smoke = subparsers.add_parser("smoke-dash", help="Run the CUDA DASH-MI engineering gate")
    dash_smoke.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    dash = subparsers.add_parser("run-dash", help="Run nested source-grouped DASH-MI")
    dash.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    dash.add_argument("--output-dir", type=Path)
    subparsers.add_parser("smoke-panel", help="Run the frozen-encoder PANEL-MI engineering gate")
    panel = subparsers.add_parser(
        "run-panel", help="Run the seven-transcript multi-annotator study"
    )
    panel.add_argument("--output-dir", type=Path)
    ac_baselines = subparsers.add_parser(
        "run-ac-baselines", help="Run source-disjoint Task A and Task C baselines"
    )
    ac_baselines.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ac_baselines.add_argument("--output-dir", type=Path)
    qtrace_smoke = subparsers.add_parser(
        "smoke-qtrace", help="Run the Q-TRACE-MI CUDA engineering gate"
    )
    qtrace_smoke.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    qtrace = subparsers.add_parser(
        "run-qtrace", help="Run source-disjoint joint Task A/C Q-TRACE-MI"
    )
    qtrace.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    qtrace.add_argument("--output-dir", type=Path)
    safe_smoke = subparsers.add_parser(
        "smoke-safe-mi", help="Run the SAFE-MI v2 CUDA engineering gate"
    )
    safe_smoke.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    safe = subparsers.add_parser(
        "run-safe-mi", help="Run the exploratory SAFE-MI staged Task A/C campaign"
    )
    safe.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    safe.add_argument("--output-dir", type=Path)
    safe_extension = subparsers.add_parser(
        "run-safe-mi-extension",
        help="Run the registered posthoc SAFE-MI v2.1 reporting extension",
    )
    safe_extension.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    safe_extension.add_argument("--output-dir", type=Path)
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
        inventory_hash = write_json_create_only(GATE0 / "legacy_inventory.json", legacy_inventory())
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
    if args.command == "audit-mi-tags":
        result = run_mi_tags_sample_audit(corpus)
        summary = result["summary"]
        print(
            "MI-TAGS sample audit: "
            f"records={summary['sample_records']}; "
            f"quarantined={summary['quarantined_records']}; "
            "full_external_evaluation=blocked"
        )
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
        from .neural import run_environment_gate

        result = run_environment_gate()
        print(
            f"PASS CUDA/BF16 environment: {result['runtime_environment']['gpu']} "
            f"with torch {result['runtime_environment']['torch']}"
        )
        return 0
    if args.command == "smoke-neural":
        from .neural import run_neural_smoke

        result = run_neural_smoke(corpus, read_json(args.splits), args.model)
        print(
            f"PASS {args.model} CUDA smoke: "
            f"peak={result['peak_memory_bytes'] / (1024**3):.2f} GiB; "
            f"steps={result['optimizer_steps']}"
        )
        return 0
    if args.command == "run-neural":
        from .neural import run_neural

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
        from .dash import run_dash_smoke

        result = run_dash_smoke(corpus, read_json(args.splits))
        print(
            f"PASS DASH-MI CUDA smoke: "
            f"peak={result['peak_memory_bytes'] / (1024**3):.2f} GiB; "
            f"steps={result['optimizer_steps']}"
        )
        return 0
    if args.command == "run-dash":
        from .dash import run_dash_mi

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
    if args.command == "smoke-panel":
        from .panel import run_panel_smoke

        result = run_panel_smoke(corpus)
        print(
            "PASS PANEL-MI smoke: "
            f"encoder_dimensions={result['embedding_dimensions']}; "
            f"projected_dimensions={result['projected_dimensions']}; "
            f"epochs={result['optimizer_epochs']}"
        )
        return 0
    if args.command == "run-panel":
        from .panel import run_panel_mi

        result = run_panel_mi(corpus, output_dir=args.output_dir)
        primary = result["metrics"]["seed_ensemble"]["therapist"]["panel_mi"]
        gate = result["inference"]["candidate_success_gate"]["pass"]
        print(
            "panel_mi therapist: transcript-balanced vote log score="
            f"{primary['transcript_balanced_vote_log_score']:.4f}; "
            f"candidate_gate_pass={gate}"
        )
        return 0
    if args.command == "run-ac-baselines":
        result = run_ac_baselines(
            corpus,
            read_json(args.splits),
            output_dir=args.output_dir,
        )
        task_a = result["task_a_metrics"]["tfidf_raw_prefix"]["t10"]
        task_c = result["task_c_metrics"]["tfidf_causal10"]
        print(
            "Task A TF-IDF t10 balanced accuracy="
            f"{task_a['source_balanced_balanced_accuracy']:.4f}; "
            "Task C TF-IDF source-balanced macro-F1="
            f"{task_c['source_balanced_macro_f1']:.4f}"
        )
        return 0
    if args.command == "smoke-qtrace":
        from .qtrace import run_qtrace_smoke

        result = run_qtrace_smoke(corpus, read_json(args.splits))
        print(
            "PASS Q-TRACE-MI CUDA smoke: "
            f"loss={result['loss']:.4f}; "
            f"peak={result['peak_memory_bytes'] / (1024**3):.2f} GiB"
        )
        return 0
    if args.command == "run-qtrace":
        from .qtrace import run_qtrace

        result = run_qtrace(
            corpus,
            read_json(args.splits),
            output_dir=args.output_dir,
        )
        task_a = result["task_a_metrics"]["qtrace_mi"]["t10"]
        task_c = result["task_c_metrics"]["qtrace_mi"]
        print(
            "Q-TRACE-MI: Task A t10 balanced accuracy="
            f"{task_a['source_balanced_balanced_accuracy']:.4f}; "
            "Task C source-balanced macro-F1="
            f"{task_c['source_balanced_macro_f1']:.4f}; "
            f"joint_gate_pass={result['candidate_success_gate']['pass']}"
        )
        return 0
    if args.command == "smoke-safe-mi":
        from .safe_mi import run_safe_mi_smoke

        result = run_safe_mi_smoke(corpus, read_json(args.splits))
        print(
            "PASS SAFE-MI CUDA smoke: "
            f"loss={result['loss']:.4f}; "
            f"peak={result['peak_memory_bytes'] / (1024**3):.2f} GiB"
        )
        return 0
    if args.command == "run-safe-mi":
        from .safe_mi import run_safe_mi

        result = run_safe_mi(
            corpus,
            read_json(args.splits),
            output_dir=args.output_dir,
        )
        finalists = result["finalists"]
        metrics = result["final"]["task_c_metrics"]
        rendered = ", ".join(
            f"{model}={metrics[model]['source_balanced_macro_f1']:.4f}" for model in finalists
        )
        print(f"SAFE-MI final Task C macro-F1: {rendered}")
        return 0
    if args.command == "run-safe-mi-extension":
        from .safe_mi_extension import run_safe_mi_extension

        result = run_safe_mi_extension(
            corpus,
            read_json(args.splits),
            output_dir=args.output_dir,
        )
        task_a = result["task_a_metrics"]["m2_oneway"]["t10"]
        task_c = result["task_c_metrics"]["c1_adapted_gru"]
        print(
            "SAFE-MI v2.1 posthoc audit: "
            f"m2 Task A t10 balanced accuracy={task_a['source_balanced_balanced_accuracy']:.4f}; "
            f"c1 Task C macro-F1={task_c['source_balanced_macro_f1']:.4f}"
        )
        return 0
    if args.command == "compare-models":
        result = run_comparison(args.config, args.output_dir)
        delta = result["point_deltas_candidate_minus_baseline"]["source_balanced_macro_f1"]
        interval = result["bootstrap"]["intervals"]["source_balanced_macro_f1"]
        print(
            f"paired source macro-F1 delta={delta:.4f}; "
            f"95% CI=[{interval['low']:.4f}, {interval['high']:.4f}]; "
            f"gate_pass={result['candidate_success_gate']['pass']}"
        )
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")
