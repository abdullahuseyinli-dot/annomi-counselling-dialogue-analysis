from __future__ import annotations

from pathlib import Path
from typing import Any

from .constants import FULL_MANIFEST, LABELS, LEGACY_SPLIT, SIMPLE_MANIFEST
from .data import Corpus
from .io import read_json, sha256_file


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _disagreement_summary(
    corpus: Corpus,
    interlocutor: str,
    label_column: str,
) -> dict[str, Any]:
    keys = corpus.utterances.loc[
        corpus.utterances["interlocutor"].eq(interlocutor), ["transcript_id", "utterance_id"]
    ]
    rated = corpus.annotations.merge(keys, on=["transcript_id", "utterance_id"], how="inner")
    counts = rated.groupby(["transcript_id", "utterance_id"])["annotator_id"].nunique()
    multi_keys = counts[counts > 1].index
    multi = rated.set_index(["transcript_id", "utterance_id"]).loc[multi_keys].reset_index()
    unique_labels = multi.groupby(["transcript_id", "utterance_id"])[label_column].nunique()
    disagreements = int(unique_labels.gt(1).sum())
    total = len(unique_labels)
    return {
        "items": total,
        "items_with_any_disagreement": disagreements,
        "rate": _rate(disagreements, total),
    }


def build_data_audit(corpus: Corpus, legacy_split_path: Path = LEGACY_SPLIT) -> dict[str, Any]:
    utterances = corpus.utterances
    therapist = utterances[utterances["interlocutor"].eq("therapist")].copy()
    legacy = read_json(legacy_split_path)
    train_transcripts = {int(value) for value in legacy["train_transcripts"]}
    test_transcripts = {int(value) for value in legacy["test_transcripts"]}

    transcript_sources = utterances[["transcript_id", "source_id"]].drop_duplicates()
    train_sources = set(
        transcript_sources.loc[
            transcript_sources["transcript_id"].isin(train_transcripts), "source_id"
        ]
    )
    test_sources = set(
        transcript_sources.loc[
            transcript_sources["transcript_id"].isin(test_transcripts), "source_id"
        ]
    )
    crossing_sources = train_sources & test_sources
    affected_test = therapist[
        therapist["transcript_id"].isin(test_transcripts)
        & therapist["source_id"].isin(crossing_sources)
    ]
    test_therapist = therapist[therapist["transcript_id"].isin(test_transcripts)]

    text_groups = therapist.groupby("normalized_text", sort=False)
    cross_transcript_texts = {
        text for text, group in text_groups if group["transcript_id"].nunique() > 1
    }
    train_texts = set(
        therapist.loc[therapist["transcript_id"].isin(train_transcripts), "normalized_text"]
    )
    test_texts = set(test_therapist["normalized_text"])
    crossing_texts = train_texts & test_texts
    conflicting = 0
    for text in cross_transcript_texts:
        group = therapist[therapist["normalized_text"].eq(text)]
        if group["main_therapist_behaviour"].nunique() > 1:
            conflicting += 1

    annotation_counts = corpus.annotations.groupby(["transcript_id", "utterance_id"])[
        "annotator_id"
    ].nunique()
    multi_keys = annotation_counts[annotation_counts > 1].index
    multi_transcripts = sorted({int(key[0]) for key in multi_keys})

    simple_manifest = read_json(SIMPLE_MANIFEST)
    full_manifest = read_json(FULL_MANIFEST)
    return {
        "audit_id": "annomi-source-and-annotation-audit-v1",
        "data": {
            "simple_sha256": sha256_file(
                Path(SIMPLE_MANIFEST).parent / "raw" / "AnnoMI" / "dataset.csv"
            ),
            "full_sha256": sha256_file(
                Path(FULL_MANIFEST).parent / "raw" / "AnnoMI" / "AnnoMI-full.csv"
            ),
            "expected_simple_sha256": simple_manifest["sha256"],
            "expected_full_sha256": full_manifest["sha256"],
            "utterances": len(utterances),
            "therapist_utterances": len(therapist),
            "transcripts": int(utterances["transcript_id"].nunique()),
            "source_video_urls": int(utterances["source_id"].nunique()),
            "class_counts": {
                label: int(therapist["main_therapist_behaviour"].eq(label).sum())
                for label in LABELS
            },
        },
        "legacy_project_split": {
            "status": "development_consumed",
            "train_transcripts": len(train_transcripts),
            "test_transcripts": len(test_transcripts),
            "source_ids_crossing_boundary": len(crossing_sources),
            "affected_transcripts": int(
                transcript_sources[transcript_sources["source_id"].isin(crossing_sources)][
                    "transcript_id"
                ].nunique()
            ),
            "affected_test_transcripts": int(affected_test["transcript_id"].nunique()),
            "affected_test_therapist_utterances": len(affected_test),
            "test_therapist_utterances": len(test_therapist),
            "affected_test_therapist_rate": _rate(len(affected_test), len(test_therapist)),
        },
        "repeated_text": {
            "unique_normalized_therapist_texts": int(therapist["normalized_text"].nunique()),
            "cross_transcript_text_groups": len(cross_transcript_texts),
            "train_test_crossing_text_groups": len(crossing_texts),
            "test_rows_with_text_seen_in_train": int(
                test_therapist["normalized_text"].isin(train_texts).sum()
            ),
            "test_seen_text_rate": _rate(
                int(test_therapist["normalized_text"].isin(train_texts).sum()),
                len(test_therapist),
            ),
            "cross_transcript_groups_with_conflicting_labels": conflicting,
            "interpretation": (
                "Repeated short phrases are not automatically leakage; report seen/unseen and "
                "conflicting-label slices and keep source grouping primary."
            ),
        },
        "multi_annotator": {
            "unique_items": len(multi_keys),
            "transcripts": len(multi_transcripts),
            "annotations_per_item": sorted(
                {int(value) for value in annotation_counts.loc[multi_keys].unique()}
            ),
            "transcripts_in_legacy_test": len(set(multi_transcripts) & test_transcripts),
            "therapist_main_behaviour": _disagreement_summary(
                corpus, "therapist", "main_therapist_behaviour"
            ),
            "client_talk_type": _disagreement_summary(corpus, "client", "client_talk_type"),
            "claim_boundary": "Only seven transcripts independently identify disagreement effects.",
        },
    }
