from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import FULL_DATA, KEY_COLUMNS, LABEL_COLUMNS, LABELS, SIMPLE_DATA
from .io import sha256_text


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"\s+", " ", value).strip()


def normalize_url(value: str) -> str:
    return str(value).strip().rstrip("/").casefold()


def source_id(value: str) -> str:
    return sha256_text(normalize_url(value))


@dataclass(frozen=True)
class Corpus:
    utterances: pd.DataFrame
    annotations: pd.DataFrame


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run tools/download_dataset.py for both simple and full variants."
        )
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    frame["transcript_id"] = frame["transcript_id"].astype(int)
    frame["utterance_id"] = frame["utterance_id"].astype(int)
    return frame


def load_corpus(
    simple_path: Path = SIMPLE_DATA,
    full_path: Path = FULL_DATA,
) -> Corpus:
    simple = _read_csv(simple_path)
    full = _read_csv(full_path)

    if simple.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("AnnoMI-simple contains duplicate utterance keys")
    if full.duplicated([*KEY_COLUMNS, "annotator_id"]).any():
        raise ValueError("AnnoMI-full contains duplicate utterance/annotator keys")

    expected_keys = set(map(tuple, simple.loc[:, KEY_COLUMNS].to_numpy()))
    full_keys = set(map(tuple, full.loc[:, KEY_COLUMNS].to_numpy()))
    if expected_keys != full_keys:
        raise ValueError("Simple and full releases do not contain the same utterance keys")

    invariant_columns = [
        "mi_quality",
        "video_title",
        "video_url",
        "topic",
        "interlocutor",
        "timestamp",
        "utterance_text",
    ]
    inconsistent = full.groupby(list(KEY_COLUMNS), sort=False)[invariant_columns].nunique()
    if (inconsistent > 1).any(axis=None):
        raise ValueError("Metadata differs across annotations for the same utterance")

    full_first = full.drop_duplicates(list(KEY_COLUMNS), keep="first")
    merged = simple.merge(
        full_first[[*KEY_COLUMNS, *invariant_columns]],
        on=list(KEY_COLUMNS),
        suffixes=("", "_full"),
        validate="one_to_one",
    )
    for column in invariant_columns:
        other = f"{column}_full"
        if not merged[column].equals(merged[other]):
            raise ValueError(f"Simple/full metadata mismatch in {column}")
        merged = merged.drop(columns=other)

    merged["source_id"] = merged["video_url"].map(source_id)
    merged["normalized_text"] = merged["utterance_text"].map(normalize_text)
    annotations = full[[*KEY_COLUMNS, "annotator_id", *LABEL_COLUMNS]].copy()
    return Corpus(utterances=merged, annotations=annotations)


def build_therapist_examples(corpus: Corpus, context_turns: int = 10) -> pd.DataFrame:
    if context_turns < 0:
        raise ValueError("context_turns must be non-negative")
    examples: list[dict[str, object]] = []
    for _, dialogue in corpus.utterances.groupby("transcript_id", sort=True):
        dialogue = dialogue.sort_values("utterance_id", kind="stable")
        history: list[str] = []
        for row in dialogue.itertuples(index=False):
            role = str(row.interlocutor).upper()
            rendered = f"[{role}] {row.utterance_text}"
            if row.interlocutor == "therapist":
                prior = history[-context_turns:] if context_turns else []
                context = "\n".join([*prior, f"[TARGET_THERAPIST] {row.utterance_text}"])
                target_first_context = "\n".join(
                    [
                        f"[TARGET_THERAPIST] {row.utterance_text}",
                        *reversed(prior),
                    ]
                )
                recent_history = "\n".join(reversed(prior))
                examples.append(
                    {
                        "transcript_id": int(row.transcript_id),
                        "utterance_id": int(row.utterance_id),
                        "source_id": str(row.source_id),
                        "label": str(row.main_therapist_behaviour),
                        "utterance_text": str(row.utterance_text),
                        "context_text": context,
                        "target_first_context_text": target_first_context,
                        "recent_history_text": recent_history,
                        "normalized_text": str(row.normalized_text),
                    }
                )
            history.append(rendered)
    result = pd.DataFrame(examples)
    if not result["label"].isin({"reflection", "question", "therapist_input", "other"}).all():
        raise ValueError("Unexpected therapist label")
    return result


def add_therapist_vote_distributions(
    examples: pd.DataFrame,
    annotations: pd.DataFrame,
) -> pd.DataFrame:
    """Attach empirical annotator distributions without exposing annotator identity."""
    valid = annotations[annotations["main_therapist_behaviour"].isin(LABELS)].copy()
    if valid.empty:
        raise ValueError("No therapist-behaviour annotations are available")
    counts = (
        valid.groupby([*KEY_COLUMNS, "main_therapist_behaviour"], sort=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=LABELS, fill_value=0)
    )
    annotation_count = counts.sum(axis=1)
    if (annotation_count <= 0).any():
        raise ValueError("A therapist utterance has no valid annotations")
    probabilities = counts.div(annotation_count, axis=0)
    vote_frame = probabilities.rename(
        columns={label: f"vote_prob_{label}" for label in LABELS}
    ).reset_index()
    vote_frame["annotation_count"] = annotation_count.to_numpy(dtype=int)
    probability_values = probabilities.to_numpy(dtype=float)
    log_probability = np.zeros_like(probability_values)
    np.log(probability_values, out=log_probability, where=probability_values > 0)
    vote_frame["vote_entropy"] = -(
        probability_values * log_probability
    ).sum(axis=1) / np.log(len(LABELS))

    result = examples.merge(vote_frame, on=list(KEY_COLUMNS), how="left", validate="one_to_one")
    vote_columns = [f"vote_prob_{label}" for label in LABELS]
    if result[vote_columns].isna().any(axis=None):
        raise ValueError("At least one therapist example lacks an annotation distribution")
    if not np.allclose(result[vote_columns].sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("Therapist annotation distributions do not sum to one")
    label_indices = np.asarray([LABELS.index(label) for label in result["label"]])
    result["hard_label_vote_probability"] = result[vote_columns].to_numpy()[
        np.arange(len(result)), label_indices
    ]
    result["annotator_disagreement"] = result[vote_columns].max(axis=1).lt(1.0)
    return result
