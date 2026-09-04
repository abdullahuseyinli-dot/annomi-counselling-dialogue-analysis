from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .constants import CLIENT_LABELS, LABELS, QUALITY_LABELS
from .data import Corpus


@dataclass(frozen=True)
class SessionTurns:
    """One dialogue represented without any information from another dialogue."""

    transcript_id: int
    source_id: str
    quality: int
    utterance_ids: np.ndarray
    texts: tuple[str, ...]
    roles: np.ndarray
    therapist_labels: np.ndarray
    client_labels: np.ndarray
    next_action_targets: np.ndarray
    quality_positions: dict[str, int]


def _label_index(value: Any, labels: tuple[str, ...]) -> int:
    text = str(value)
    return labels.index(text) if text in labels else -100


def _render(role: str, text: str) -> str:
    return f"[{str(role).upper()}] {text}"


def build_task_c_examples(corpus: Corpus, context_turns: int = 10) -> pd.DataFrame:
    """Build strict client-to-therapist handoff forecasts from observed prefixes.

    The target therapist utterance and all later rows are deliberately absent from
    ``context_text``. Gold historical codes are retained only in explicitly named
    oracle columns; confirmatory models must not consume those columns.
    """
    if context_turns < 1:
        raise ValueError("context_turns must be positive")
    rows: list[dict[str, Any]] = []
    for transcript_id, dialogue in corpus.utterances.groupby("transcript_id", sort=True):
        dialogue = dialogue.sort_values("utterance_id", kind="stable").reset_index(drop=True)
        history: list[str] = []
        previous_therapist_label: str | None = None
        for index, current in dialogue.iterrows():
            rendered = _render(str(current["interlocutor"]), str(current["utterance_text"]))
            history.append(rendered)
            if current["interlocutor"] == "therapist":
                previous_therapist_label = str(current["main_therapist_behaviour"])
                continue
            if index + 1 >= len(dialogue):
                continue
            following = dialogue.iloc[index + 1]
            if following["interlocutor"] != "therapist":
                continue
            label = str(following["main_therapist_behaviour"])
            if label not in LABELS:
                raise ValueError("A therapist handoff has an invalid target label")
            rows.append(
                {
                    "transcript_id": int(transcript_id),
                    "decision_utterance_id": int(current["utterance_id"]),
                    "target_utterance_id": int(following["utterance_id"]),
                    "source_id": str(current["source_id"]),
                    "label": label,
                    "mi_quality": str(current["mi_quality"]),
                    "context_text": "\n".join(history[-context_turns:]),
                    "current_client_text": str(current["utterance_text"]),
                    "normalized_text": str(current["normalized_text"]),
                    "oracle_current_client_label": str(current["client_talk_type"]),
                    "oracle_previous_therapist_label": previous_therapist_label,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No client-to-therapist handoffs were found")
    if not result["label"].isin(LABELS).all():
        raise ValueError("Unexpected Task C label")
    if result.duplicated(["transcript_id", "target_utterance_id"]).any():
        raise ValueError("Task C contains duplicate targets")
    return result


def _prefix_features(prefix: pd.DataFrame) -> dict[str, float]:
    therapist = prefix["interlocutor"].eq("therapist")
    client = prefix["interlocutor"].eq("client")
    words = prefix["utterance_text"].astype(str).str.split().map(len)
    switches = prefix["interlocutor"].ne(prefix["interlocutor"].shift()).iloc[1:]
    features: dict[str, float] = {
        "observed_turns": float(len(prefix)),
        "observed_words": float(words.sum()),
        "mean_words": float(words.mean()),
        "client_turns": float(client.sum()),
        "role_switch_rate": float(switches.mean()) if len(switches) else 0.0,
    }
    for label in LABELS:
        features[f"oracle_therapist_prop_{label}"] = float(
            prefix.loc[therapist, "main_therapist_behaviour"].eq(label).mean()
        )
    for label in CLIENT_LABELS:
        features[f"oracle_client_prop_{label}"] = float(
            prefix.loc[client, "client_talk_type"].eq(label).mean()
        )
    return features


def build_task_a_examples(
    corpus: Corpus,
    therapist_budgets: tuple[int, ...] = (3, 5, 10, 20),
) -> pd.DataFrame:
    """Build one quality example per available absolute prefix budget and session endpoint."""
    if not therapist_budgets or any(value < 1 for value in therapist_budgets):
        raise ValueError("Therapist budgets must be positive")
    if tuple(sorted(set(therapist_budgets))) != therapist_budgets:
        raise ValueError("Therapist budgets must be unique and increasing")
    rows: list[dict[str, Any]] = []
    for transcript_id, dialogue in corpus.utterances.groupby("transcript_id", sort=True):
        dialogue = dialogue.sort_values("utterance_id", kind="stable").reset_index(drop=True)
        therapist_total = int(dialogue["interlocutor"].eq("therapist").sum())
        checkpoints: list[tuple[str, int]] = []
        therapist_seen = 0
        for position, role in enumerate(dialogue["interlocutor"]):
            if role != "therapist":
                continue
            therapist_seen += 1
            if therapist_seen in therapist_budgets:
                checkpoints.append((f"t{therapist_seen}", position))
        last_therapist_position = int(
            np.flatnonzero(dialogue["interlocutor"].eq("therapist").to_numpy())[-1]
        )
        checkpoints.append(("full", last_therapist_position))
        seen: set[tuple[str, int]] = set()
        for checkpoint, position in checkpoints:
            key = (checkpoint, position)
            if key in seen:
                continue
            seen.add(key)
            prefix = dialogue.iloc[: position + 1]
            rendered = [
                _render(str(row.interlocutor), str(row.utterance_text))
                for row in prefix.itertuples(index=False)
            ]
            record: dict[str, Any] = {
                "transcript_id": int(transcript_id),
                "source_id": str(dialogue.iloc[0]["source_id"]),
                "label": str(dialogue.iloc[0]["mi_quality"]),
                "checkpoint": checkpoint,
                "last_utterance_id": int(dialogue.iloc[position]["utterance_id"]),
                "observed_therapist_turns": (
                    therapist_total if checkpoint == "full" else int(checkpoint[1:])
                ),
                "prefix_text": "\n".join(rendered),
            }
            record.update(_prefix_features(prefix))
            rows.append(record)
    result = pd.DataFrame(rows)
    if not result["label"].isin(QUALITY_LABELS).all():
        raise ValueError("Unexpected Task A quality label")
    if result.duplicated(["transcript_id", "checkpoint"]).any():
        raise ValueError("Task A contains duplicate transcript/checkpoint rows")
    return result


def build_session_turns(
    corpus: Corpus,
    therapist_budgets: tuple[int, ...] = (3, 5, 10, 20),
) -> list[SessionTurns]:
    """Build causal session tensors; labels are targets, never model inputs."""
    sessions: list[SessionTurns] = []
    for transcript_id, dialogue in corpus.utterances.groupby("transcript_id", sort=True):
        dialogue = dialogue.sort_values("utterance_id", kind="stable").reset_index(drop=True)
        roles = dialogue["interlocutor"].map({"client": 0, "therapist": 1})
        if roles.isna().any():
            raise ValueError("Unexpected dialogue role")
        therapist_labels = np.asarray(
            [_label_index(value, LABELS) for value in dialogue["main_therapist_behaviour"]],
            dtype=np.int64,
        )
        client_labels = np.asarray(
            [_label_index(value, CLIENT_LABELS) for value in dialogue["client_talk_type"]],
            dtype=np.int64,
        )
        next_actions = np.full(len(dialogue), -100, dtype=np.int64)
        for position in range(len(dialogue) - 1):
            if roles.iloc[position] == 0 and roles.iloc[position + 1] == 1:
                next_actions[position] = therapist_labels[position + 1]
        therapist_positions = np.flatnonzero(roles.to_numpy(dtype=np.int64) == 1)
        quality_positions = {
            f"t{budget}": int(therapist_positions[budget - 1])
            for budget in therapist_budgets
            if len(therapist_positions) >= budget
        }
        quality_positions["full"] = int(therapist_positions[-1])
        quality = QUALITY_LABELS.index(str(dialogue.iloc[0]["mi_quality"]))
        sessions.append(
            SessionTurns(
                transcript_id=int(transcript_id),
                source_id=str(dialogue.iloc[0]["source_id"]),
                quality=quality,
                utterance_ids=dialogue["utterance_id"].to_numpy(dtype=np.int64),
                texts=tuple(dialogue["utterance_text"].astype(str)),
                roles=roles.to_numpy(dtype=np.int64),
                therapist_labels=therapist_labels,
                client_labels=client_labels,
                next_action_targets=next_actions,
                quality_positions=quality_positions,
            )
        )
    if sum(int((session.next_action_targets >= 0).sum()) for session in sessions) != len(
        build_task_c_examples(corpus)
    ):
        raise AssertionError("Session and tabular Task C builders disagree")
    return sessions
