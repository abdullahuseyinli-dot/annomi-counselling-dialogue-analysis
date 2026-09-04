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


@dataclass(frozen=True)
class MultiAnnotatorTask:
    task: str
    labels: tuple[str, ...]
    items: pd.DataFrame
    annotation_label_indices: np.ndarray
    annotator_ids: tuple[str, ...]


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
    vote_frame["vote_entropy"] = -(probability_values * log_probability).sum(axis=1) / np.log(
        len(LABELS)
    )

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


def build_multiannotator_task(
    corpus: Corpus,
    task: str,
    label_column: str,
    labels: tuple[str, ...],
    expected_annotations_per_item: int = 10,
    expected_items: int | None = None,
) -> MultiAnnotatorTask:
    """Build one speaker-specific task from items with a complete annotation panel."""
    if task not in {"therapist", "client"}:
        raise ValueError(f"Unexpected multi-annotator task: {task}")
    if label_column not in corpus.annotations.columns:
        raise ValueError(f"Missing annotation label column: {label_column}")
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Registered task labels must be non-empty and unique")

    annotation_counts = corpus.annotations.groupby(list(KEY_COLUMNS), sort=True).size()
    complete_keys = annotation_counts[annotation_counts.eq(expected_annotations_per_item)].index
    utterances = corpus.utterances[corpus.utterances["interlocutor"].eq(task)].copy()
    utterance_index = pd.MultiIndex.from_frame(utterances.loc[:, KEY_COLUMNS])
    utterances = utterances.loc[utterance_index.isin(complete_keys)].copy()
    utterances = utterances.sort_values(list(KEY_COLUMNS), kind="stable").reset_index(drop=True)
    if expected_items is not None and len(utterances) != expected_items:
        raise ValueError(
            f"Expected {expected_items} complete {task} items, found {len(utterances)}"
        )
    if utterances.empty:
        raise ValueError(f"No complete multi-annotator items for {task}")

    item_keys = pd.MultiIndex.from_frame(utterances.loc[:, KEY_COLUMNS])
    annotation_index = pd.MultiIndex.from_frame(corpus.annotations.loc[:, KEY_COLUMNS])
    annotations = corpus.annotations.loc[annotation_index.isin(item_keys)].copy()
    annotations["annotator_id"] = annotations["annotator_id"].astype(str)
    if not annotations[label_column].isin(labels).all():
        unexpected = sorted(set(annotations[label_column]) - set(labels))
        raise ValueError(f"Unexpected {task} annotation labels: {unexpected}")

    annotator_sets = annotations.groupby(list(KEY_COLUMNS), sort=True)["annotator_id"].agg(
        lambda values: tuple(sorted(values))
    )
    if annotator_sets.nunique() != 1:
        raise ValueError(f"Complete {task} items do not share one annotation panel")
    annotator_ids = tuple(annotator_sets.iloc[0])
    if len(annotator_ids) != expected_annotations_per_item:
        raise ValueError(f"The {task} annotation panel has an unexpected size")

    label_to_index = {label: index for index, label in enumerate(labels)}
    matrix = annotations.pivot(
        index=list(KEY_COLUMNS), columns="annotator_id", values=label_column
    ).reindex(index=item_keys, columns=annotator_ids)
    if matrix.isna().any(axis=None):
        raise ValueError(f"At least one complete {task} item lacks an annotator label")
    annotation_label_indices = matrix.map(label_to_index.__getitem__).to_numpy(dtype=int)
    vote_probabilities = np.stack(
        [(annotation_label_indices == index).mean(axis=1) for index in range(len(labels))],
        axis=1,
    )
    if not np.allclose(vote_probabilities.sum(axis=1), 1.0, atol=1e-12):
        raise AssertionError("Multi-annotator vote probabilities do not sum to one")

    items = utterances[
        [
            "transcript_id",
            "utterance_id",
            "source_id",
            "utterance_text",
            "normalized_text",
        ]
    ].copy()
    items["task"] = task
    items["role_prefixed_text"] = task + ": " + items["utterance_text"].astype(str)
    items["annotation_count"] = expected_annotations_per_item
    for index, label in enumerate(labels):
        items[f"vote_prob_{label}"] = vote_probabilities[:, index]
    log_votes = np.zeros_like(vote_probabilities)
    np.log(vote_probabilities, out=log_votes, where=vote_probabilities > 0)
    items["vote_entropy"] = -(vote_probabilities * log_votes).sum(axis=1) / np.log(len(labels))
    maxima = vote_probabilities.max(axis=1, keepdims=True)
    items["plurality_tie"] = np.isclose(vote_probabilities, maxima).sum(axis=1) > 1
    items["plurality_label"] = np.asarray(labels, dtype=object)[vote_probabilities.argmax(axis=1)]

    return MultiAnnotatorTask(
        task=task,
        labels=labels,
        items=items,
        annotation_label_indices=annotation_label_indices,
        annotator_ids=annotator_ids,
    )
