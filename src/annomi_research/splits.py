from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .constants import LABELS, PROTOCOL
from .data import Corpus, build_therapist_examples
from .io import canonical_json_bytes, read_json


def _manifest_digest(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "manifest_sha256"}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def build_source_folds(corpus: Corpus, protocol_path: Path = PROTOCOL) -> dict[str, Any]:
    protocol = read_json(protocol_path)
    n_splits = int(protocol["data"]["outer_folds"])
    seed = int(protocol["data"]["fold_seed"])
    examples = build_therapist_examples(corpus, context_turns=0)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[dict[str, Any]] = []
    all_indices = np.arange(len(examples))

    for fold_id, (_, test_index) in enumerate(
        splitter.split(all_indices, examples["label"], groups=examples["source_id"])
    ):
        test = examples.iloc[test_index]
        test_sources = sorted(test["source_id"].unique().tolist())
        test_transcripts = sorted(int(value) for value in test["transcript_id"].unique())
        label_counts = Counter(test["label"])
        folds.append(
            {
                "fold": fold_id,
                "test_source_ids": test_sources,
                "test_transcript_ids": test_transcripts,
                "test_sources": len(test_sources),
                "test_transcripts": len(test_transcripts),
                "test_therapist_utterances": len(test),
                "test_label_counts": {label: int(label_counts[label]) for label in LABELS},
            }
        )

    manifest: dict[str, Any] = {
        "split_id": "annomi-source-grouped-5fold-v1",
        "protocol_id": protocol["protocol_id"],
        "algorithm": "StratifiedGroupKFold over therapist labels with normalized video_url groups",
        "n_splits": n_splits,
        "seed": seed,
        "source_id_definition": "sha256(casefold(strip_trailing_slash(video_url)))",
        "all_folds_are_reported": True,
        "folds": folds,
    }
    manifest["manifest_sha256"] = _manifest_digest(manifest)
    validate_source_folds(corpus, manifest)
    return manifest


def validate_source_folds(corpus: Corpus, manifest: dict[str, Any]) -> None:
    if manifest.get("manifest_sha256") != _manifest_digest(manifest):
        raise ValueError("Source-fold manifest self-hash mismatch")
    expected_sources = set(corpus.utterances["source_id"].unique())
    expected_transcripts = {int(value) for value in corpus.utterances["transcript_id"].unique()}
    observed_sources: list[str] = []
    observed_transcripts: list[int] = []
    for fold in manifest["folds"]:
        sources = fold["test_source_ids"]
        transcripts = fold["test_transcript_ids"]
        if len(sources) != len(set(sources)) or len(transcripts) != len(set(transcripts)):
            raise ValueError(f"Duplicates inside fold {fold['fold']}")
        observed_sources.extend(sources)
        observed_transcripts.extend(int(value) for value in transcripts)
        actual = corpus.utterances[corpus.utterances["transcript_id"].isin(transcripts)][
            "source_id"
        ]
        if set(actual) != set(sources):
            raise ValueError(f"Transcript/source mismatch in fold {fold['fold']}")
    if len(observed_sources) != len(set(observed_sources)):
        raise ValueError("A source occurs in more than one outer test fold")
    if len(observed_transcripts) != len(set(observed_transcripts)):
        raise ValueError("A transcript occurs in more than one outer test fold")
    if set(observed_sources) != expected_sources:
        raise ValueError("Outer folds do not cover every source exactly once")
    if set(observed_transcripts) != expected_transcripts:
        raise ValueError("Outer folds do not cover every transcript exactly once")


def fold_lookup(manifest: dict[str, Any]) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for fold in manifest["folds"]:
        for source in fold["test_source_ids"]:
            if source in lookup:
                raise ValueError(f"Repeated source in manifest: {source}")
            lookup[source] = int(fold["fold"])
    return lookup
