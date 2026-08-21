from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "source_manifest.json"
DEFAULT_DESTINATION = ROOT / "data" / "raw" / "AnnoMI" / "dataset.csv"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(path: Path, manifest: dict) -> None:
    if path.stat().st_size != manifest["bytes"]:
        raise ValueError(f"Unexpected byte count: {path.stat().st_size} != {manifest['bytes']}")
    digest = file_sha256(path)
    if digest != manifest["sha256"]:
        raise ValueError(f"Unexpected SHA-256: {digest}")

    transcript_ids: set[str] = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != manifest["expected_columns"]:
            raise ValueError(f"Unexpected columns: {reader.fieldnames}")
        row_count = 0
        for row_count, row in enumerate(reader, start=1):
            transcript_ids.add(row["transcript_id"])
    if row_count != manifest["rows"]:
        raise ValueError(f"Unexpected row count: {row_count}")
    if len(transcript_ids) != manifest["transcripts"]:
        raise ValueError(f"Unexpected transcript count: {len(transcript_ids)}")


def download(destination: Path, force: bool = False) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        validate(destination, manifest)
        print("Existing dataset passed checksum and schema validation.")
        return

    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        manifest["download_url"],
        headers={"User-Agent": "annomi-reproducibility-downloader/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
        validate(partial, manifest)
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    print("Dataset downloaded and validated successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and verify AnnoMI-simple")
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download(args.destination.resolve(), args.force)
