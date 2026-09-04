from __future__ import annotations

from annomi_research.mi_tags_external import (
    locked_partition,
    normalize_overlap_text,
    shingle_jaccard,
    token_set_ratio,
)


def test_overlap_normalization_and_token_set_ratio() -> None:
    assert normalize_overlap_text("  MI—Role_Play!  ") == "mi role play"
    assert token_set_ratio("Motivational Interview Role Play", "Role Play Motivational Interview") == 1
    assert token_set_ratio("unrelated title", "motivational interview") < 0.8


def test_shingle_jaccard_is_symmetric_and_bounded() -> None:
    first = "one two three four five six seven"
    second = "zero two three four five six eight"
    value = shingle_jaccard(first, second)
    assert 0 < value < 1
    assert value == shingle_jaccard(second, first)


def test_locked_partition_is_deterministic() -> None:
    value = locked_partition("Example Video")
    assert value in {"train", "calibration", "test"}
    assert value == locked_partition(" example---video ")
