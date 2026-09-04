from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any

import pandas as pd

from .constants import MI_TAGS_EXTERNAL_PROTOCOL, RESEARCH_RESULTS, ROOT
from .data import Corpus
from .io import canonical_json_bytes, git_commit, read_json, sha256_file, write_create_only

MI_TAGS_SAMPLE_ROOT = ROOT / "data" / "raw" / "MI-TAGS"
MI_TAGS_SAMPLE_MANIFEST = ROOT / "data" / "mi_tags_sample_manifest.json"
MI_TAGS_EXTERNAL_RESULTS = RESEARCH_RESULTS / "mi_tags_external_v1"


def normalize_overlap_text(value: str) -> str:
    """Normalize titles/transcripts exactly as locked by the external protocol."""

    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"\s+", " ", "".join(char if char.isalnum() else " " for char in normalized)).strip()


def token_set_ratio(first: str, second: str) -> float:
    """Dependency-free equivalent of the standard fuzzy token-set ratio."""

    first_tokens = set(normalize_overlap_text(first).split())
    second_tokens = set(normalize_overlap_text(second).split())
    if not first_tokens and not second_tokens:
        return 1.0
    if not first_tokens or not second_tokens:
        return 0.0
    intersection = first_tokens & second_tokens
    first_only = first_tokens - intersection
    second_only = second_tokens - intersection
    common = " ".join(sorted(intersection))
    combined_first = " ".join(sorted(intersection | first_only))
    combined_second = " ".join(sorted(intersection | second_only))
    pairs = [(combined_first, combined_second)]
    if common:
        pairs.extend([(common, combined_first), (common, combined_second)])
    return max(SequenceMatcher(None, left, right, autojunk=False).ratio() for left, right in pairs)


def _shingles(value: str, width: int = 5) -> set[tuple[str, ...]]:
    tokens = normalize_overlap_text(value).split()
    if not tokens:
        return set()
    if len(tokens) < width:
        return {tuple(tokens)}
    return {tuple(tokens[index : index + width]) for index in range(len(tokens) - width + 1)}


def shingle_jaccard(first: str, second: str, width: int = 5) -> float:
    first_shingles = _shingles(first, width)
    second_shingles = _shingles(second, width)
    union = first_shingles | second_shingles
    return len(first_shingles & second_shingles) / len(union) if union else 1.0


def locked_partition(group_key: str) -> str:
    remainder = int(hashlib.sha256(normalize_overlap_text(group_key).encode()).hexdigest(), 16) % 5
    if remainder in {0, 1, 2}:
        return "train"
    return "calibration" if remainder == 3 else "test"


def _verify_sample_files() -> dict[str, Any]:
    manifest = read_json(MI_TAGS_SAMPLE_MANIFEST)
    for name, record in manifest["files"].items():
        path = MI_TAGS_SAMPLE_ROOT / name
        if not path.exists():
            raise FileNotFoundError(f"Missing official MI-TAGS sample: {path}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"MI-TAGS sample hash mismatch: {name}")
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"MI-TAGS sample byte count mismatch: {name}")
    return manifest


def _sample_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    utterances = pd.read_csv(MI_TAGS_SAMPLE_ROOT / "sample_utterances.csv")
    globals_frame = pd.read_csv(MI_TAGS_SAMPLE_ROOT / "sample_global_mitis.csv")
    expected_utterance = {
        "id",
        "Video Title",
        "Turn",
        "Speaker",
        "Text",
        "Code",
        "Annotator",
        "Normalized Turn",
    }
    expected_global = {
        "id",
        "Video Title",
        "Annotator",
        "Empathy",
        "SofteningSustainTalk",
        "CultivatingChangeTalk",
        "Partnership",
        "Only Text",
        "Tagged Text",
        "Only Tags",
    }
    if set(utterances.columns) != expected_utterance:
        raise ValueError("Unexpected MI-TAGS utterance sample schema")
    if set(globals_frame.columns) != expected_global:
        raise ValueError("Unexpected MI-TAGS global sample schema")
    records: list[dict[str, Any]] = []
    for title, frame in utterances.groupby("Video Title", sort=True):
        frame = frame.sort_values("Turn", kind="stable")
        records.append(
            {
                "record_type": "utterance_sample_fragment",
                "title": str(title),
                "text": " ".join(frame["Text"].astype(str)),
                "rows": len(frame),
            }
        )
    for _, row in globals_frame.iterrows():
        records.append(
            {
                "record_type": "global_sample_session",
                "title": str(row["Video Title"]),
                "text": str(row["Only Text"]),
                "rows": 1,
            }
        )
    schema = {
        "utterance_rows": len(utterances),
        "utterance_titles": int(utterances["Video Title"].nunique()),
        "global_rows": len(globals_frame),
        "global_titles": int(globals_frame["Video Title"].nunique()),
        "utterance_columns": list(utterances.columns),
        "global_columns": list(globals_frame.columns),
        "observed_utterance_codes": sorted(utterances["Code"].astype(str).unique()),
        "global_score_ranges": {
            column: [int(globals_frame[column].min()), int(globals_frame[column].max())]
            for column in (
                "Empathy",
                "SofteningSustainTalk",
                "CultivatingChangeTalk",
                "Partnership",
            )
        },
    }
    return records, schema


def _annomi_records(corpus: Corpus) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for transcript_id, frame in corpus.utterances.groupby("transcript_id", sort=True):
        frame = frame.sort_values("utterance_id", kind="stable")
        records.append(
            {
                "transcript_id": int(transcript_id),
                "source_id": str(frame.iloc[0]["source_id"]),
                "title": str(frame.iloc[0]["video_title"]),
                "text": " ".join(frame["utterance_text"].astype(str)),
                "utterance_hashes": {
                    hashlib.sha256(normalize_overlap_text(value).encode()).hexdigest()
                    for value in frame["utterance_text"].astype(str)
                },
            }
        )
    return records


def build_mi_tags_sample_audit(corpus: Corpus) -> dict[str, Any]:
    """Audit the official public samples; never estimate performance from them."""

    protocol = read_json(MI_TAGS_EXTERNAL_PROTOCOL)
    if protocol["status"] != "registered_before_mi_tags_sample_or_full_data_access":
        raise ValueError("MI-TAGS external protocol is not registered")
    manifest = _verify_sample_files()
    samples, schema = _sample_records()
    annomi = _annomi_records(corpus)
    title_threshold = float(
        protocol["overlap_audit"]["fuzzy_title_token_set_ratio_threshold"]
    )
    shingle_threshold = float(
        protocol["overlap_audit"]["five_word_shingle_jaccard_threshold"]
    )
    rows: list[dict[str, Any]] = []
    for sample in samples:
        normalized_title = normalize_overlap_text(sample["title"])
        normalized_text = normalize_overlap_text(sample["text"])
        sample_text_hash = hashlib.sha256(normalized_text.encode()).hexdigest()
        best_title = max(
            annomi,
            key=lambda record: token_set_ratio(sample["title"], record["title"]),
        )
        title_score = token_set_ratio(sample["title"], best_title["title"])
        exact_title = normalized_title == normalize_overlap_text(best_title["title"])
        best_text = max(
            annomi,
            key=lambda record: shingle_jaccard(sample["text"], record["text"]),
        )
        text_score = shingle_jaccard(sample["text"], best_text["text"])
        best_text_hash = hashlib.sha256(
            normalize_overlap_text(best_text["text"]).encode()
        ).hexdigest()
        exact_session_text = sample_text_hash == best_text_hash
        exact_fragment = any(
            sample_text_hash in record["utterance_hashes"] for record in annomi
        )
        quarantined = bool(
            exact_title
            or title_score >= title_threshold
            or exact_session_text
            or text_score >= shingle_threshold
            or exact_fragment
        )
        rows.append(
            {
                "sample_record_type": sample["record_type"],
                "sample_title_sha256": hashlib.sha256(normalized_title.encode()).hexdigest(),
                "sample_text_sha256": sample_text_hash,
                "sample_rows": int(sample["rows"]),
                "locked_partition_if_released": locked_partition(sample["title"]),
                "best_title_token_set_ratio": title_score,
                "best_title_annomi_transcript_id": int(best_title["transcript_id"]),
                "best_title_annomi_source_id": best_title["source_id"],
                "exact_normalized_title_match": exact_title,
                "best_five_word_shingle_jaccard": text_score,
                "best_text_annomi_transcript_id": int(best_text["transcript_id"]),
                "best_text_annomi_source_id": best_text["source_id"],
                "exact_normalized_session_text_match": exact_session_text,
                "supplemental_exact_utterance_fragment_match": exact_fragment,
                "quarantined": quarantined,
            }
        )
    partitions = pd.Series([row["locked_partition_if_released"] for row in rows]).value_counts()
    quarantined = sum(bool(row["quarantined"]) for row in rows)
    return {
        "audit_id": "annomi-safe-mi-mi-tags-public-sample-overlap-v1",
        "status": "sample_audit_complete_full_external_evaluation_blocked",
        "performance_claim_permitted": False,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(MI_TAGS_EXTERNAL_PROTOCOL),
        "protocol_commit_before_sample_retrieval": manifest[
            "protocol_commit_before_retrieval"
        ],
        "code_commit": git_commit(ROOT),
        "upstream_repository": manifest["upstream_repository"],
        "upstream_commit": manifest["upstream_commit"],
        "sample_file_sha256": {
            name: value["sha256"] for name, value in manifest["files"].items()
        },
        "schema": schema,
        "thresholds": protocol["overlap_audit"],
        "supplemental_safety_check": (
            "Exact sample-fragment-to-AnnoMI-utterance hashes also trigger quarantine."
        ),
        "records": rows,
        "summary": {
            "sample_records": len(rows),
            "quarantined_records": quarantined,
            "clear_records": len(rows) - quarantined,
            "partition_counts_before_quarantine": {
                str(name): int(value) for name, value in partitions.items()
            },
            "minimum_test_groups_required": int(
                protocol["locked_split"]["minimum_test_groups"]
            ),
            "sample_sufficient_for_external_evaluation": False,
        },
        "full_external_stage": {
            "completed": False,
            "blocker": (
                "The official repository exposes only 10 isolated utterances and two global-score "
                "sessions. The 242-session corpus requires the author-hosted access request form."
            ),
            "next_required_input": "Researcher-approved access to the official full MI-TAGS files.",
        },
    }


def run_mi_tags_sample_audit(corpus: Corpus) -> dict[str, Any]:
    output = MI_TAGS_EXTERNAL_RESULTS / "sample_overlap_audit.json"
    if output.exists():
        return read_json(output)
    payload = build_mi_tags_sample_audit(corpus)
    write_create_only(output, canonical_json_bytes(payload))
    return payload


def validate_mi_tags_sample_audit(corpus: Corpus) -> None:
    output = MI_TAGS_EXTERNAL_RESULTS / "sample_overlap_audit.json"
    if not output.exists():
        return
    payload = read_json(output)
    if payload["performance_claim_permitted"]:
        raise ValueError("MI-TAGS sample evidence cannot support performance claims")
    if payload["summary"]["sample_records"] != len(payload["records"]):
        raise ValueError("MI-TAGS sample audit record count mismatch")
    quarantined = sum(bool(row["quarantined"]) for row in payload["records"])
    if quarantined != int(payload["summary"]["quarantined_records"]):
        raise ValueError("MI-TAGS sample quarantine count mismatch")
    for row in payload["records"]:
        numeric = (
            float(row["best_title_token_set_ratio"]),
            float(row["best_five_word_shingle_jaccard"]),
        )
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in numeric):
            raise ValueError("MI-TAGS overlap score is outside [0, 1]")
    reconstructed = build_mi_tags_sample_audit(corpus)
    for key in (
        "protocol_id",
        "protocol_sha256",
        "upstream_commit",
        "sample_file_sha256",
        "schema",
        "thresholds",
        "records",
        "summary",
        "full_external_stage",
    ):
        if payload[key] != reconstructed[key]:
            raise ValueError(f"MI-TAGS sample audit reconstruction mismatch: {key}")
