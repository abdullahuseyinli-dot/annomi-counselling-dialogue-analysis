from __future__ import annotations

import argparse
from pathlib import Path

from .audit import build_data_audit
from .baselines import run_baselines
from .constants import FULL_DATA, RESEARCH_RESULTS, SIMPLE_DATA
from .data import load_corpus
from .io import read_json, write_json_create_only
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
    raise AssertionError(f"Unhandled command: {args.command}")
