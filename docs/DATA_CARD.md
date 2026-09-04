# Data card

## Sources

| Source | Pinned record | Use in this repository |
|---|---|---|
| AnnoMI-simple | 9,699 rows, 133 transcripts, upstream commit and SHA-256 in `data/source_manifest.json` | Main therapist-behaviour and Task A/C corpus |
| AnnoMI-full | 13,551 annotation rows, 133 transcripts, ten annotators, upstream commit and SHA-256 in `data/source_manifest_full.json` | Existing multi-annotator labels and fine-grained fields |
| MI-TAGS public samples | Ten utterance rows and two global-score rows, pinned in `data/mi_tags_sample_manifest.json` | Schema and possible-overlap audit only |

The AnnoMI records are professionally transcribed and annotated demonstration dialogues. They are
not presented here as private therapy sessions or a representative sample of clinical care.

## Acquisition and storage

The downloader fetches both AnnoMI variants from a commit-pinned upstream URL and verifies byte
count, SHA-256, columns, row count, and transcript count before use:

```bash
uv run python tools/download_dataset.py --variant simple
uv run python tools/download_dataset.py --variant full
```

Validated files are stored under ignored `data/raw/AnnoMI/`. Raw dialogue text, video metadata
extracts, MI-TAGS samples, and derivatives containing source text must not be committed. Tracked
results contain aggregate metrics, hashes, non-textual identifiers, and probability ledgers without
utterance text.

## Analysis populations

- The main benchmark uses 4,882 therapist utterances grouped into 119 normalized video sources.
- The multi-annotator analysis uses 216 therapist and 212 client utterances from seven transcripts,
  with ten existing annotations per utterance.
- Task A's primary ten-turn checkpoint covers 115 eligible transcripts from 108 sources.
- Task C covers 4,743 strict client-to-therapist handoffs from 119 sources.

These are task-specific filtered populations; they should not be mistaken for the raw-file row
counts.

## MI-TAGS boundary

The locked public-sample audit marks 9 of 12 sample records as possible AnnoMI overlaps and leaves
too few test groups for evaluation. Those samples support schema and leakage checks only. Full
external performance evaluation is blocked until a researcher obtains authorized access to the
official complete files; results must not be inferred from the samples.

## Licensing and responsible reuse

The repository's MIT license covers its original code and documentation, not third-party data,
source media, or model weights. No separate dataset licence was located in the pinned upstream
repository when this card was prepared. Public availability is not itself a licence grant, so the
current upstream repository, source-media permissions, and applicable terms must be checked before
redistribution or downstream use. This repository does not resolve that uncertainty and therefore
redistributes no raw dialogues. See
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

No demographic fields are available for a meaningful subgroup-fairness evaluation. Dataset labels
must not be reinterpreted as diagnoses, outcomes, or independent assessments of therapist quality.
