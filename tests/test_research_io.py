from __future__ import annotations

from pathlib import Path

import pytest

from annomi_research.io import write_create_only


def test_evidence_is_create_only(tmp_path: Path) -> None:
    target = tmp_path / "evidence.json"
    digest = write_create_only(target, b"first\n")
    assert write_create_only(target, b"first\n") == digest
    with pytest.raises(FileExistsError):
        write_create_only(target, b"different\n")
