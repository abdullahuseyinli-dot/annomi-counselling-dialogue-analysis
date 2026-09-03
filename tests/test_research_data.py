from __future__ import annotations

import pandas as pd

from annomi_research.data import Corpus, build_therapist_examples, normalize_text, source_id


def test_normalization_is_conservative_and_stable() -> None:
    assert normalize_text("  I\u00a0AGREE.  ") == "i agree."
    assert source_id(" HTTPS://EXAMPLE.test/video/ ") == source_id(
        "https://example.test/video"
    )


def test_context_is_causal_and_dialogue_local() -> None:
    utterances = pd.DataFrame(
        [
            {
                "transcript_id": 1,
                "utterance_id": 0,
                "interlocutor": "client",
                "utterance_text": "past client",
                "main_therapist_behaviour": "n/a",
                "source_id": "source-a",
                "normalized_text": "past client",
            },
            {
                "transcript_id": 1,
                "utterance_id": 1,
                "interlocutor": "therapist",
                "utterance_text": "current therapist",
                "main_therapist_behaviour": "reflection",
                "source_id": "source-a",
                "normalized_text": "current therapist",
            },
            {
                "transcript_id": 1,
                "utterance_id": 2,
                "interlocutor": "client",
                "utterance_text": "future client",
                "main_therapist_behaviour": "n/a",
                "source_id": "source-a",
                "normalized_text": "future client",
            },
            {
                "transcript_id": 2,
                "utterance_id": 0,
                "interlocutor": "therapist",
                "utterance_text": "other dialogue",
                "main_therapist_behaviour": "question",
                "source_id": "source-b",
                "normalized_text": "other dialogue",
            },
        ]
    )
    examples = build_therapist_examples(Corpus(utterances, pd.DataFrame()), context_turns=10)
    context = examples.loc[examples["transcript_id"].eq(1), "context_text"].item()
    target_first = examples.loc[
        examples["transcript_id"].eq(1), "target_first_context_text"
    ].item()
    assert "past client" in context
    assert "current therapist" in context
    assert "future client" not in context
    assert "other dialogue" not in context
    assert target_first.startswith("[TARGET_THERAPIST] current therapist")
    assert "past client" in target_first
    assert "future client" not in target_first
    assert "other dialogue" not in target_first
