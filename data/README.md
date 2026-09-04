# Data

The raw AnnoMI counselling text is deliberately excluded from Git. The tracked manifests pin the
official sources and record expected checksums, schemas, row counts, and transcript counts:

- `source_manifest.json`: simplified AnnoMI table;
- `source_manifest_full.json`: full AnnoMI table used for source grouping and Task A/C metadata;
- `mi_tags_sample_manifest.json`: official MI-TAGS public samples used only for the overlap audit.

Download and validate the data with:

```bash
uv run python tools/download_dataset.py --variant simple
uv run python tools/download_dataset.py --variant full
```

Validated files are written below `data/raw/`, which is ignored by Git. The MI-TAGS full corpus is
not downloaded by this tool and requires separate authorized access. Do not commit derived tables
containing utterance text, source URLs, or other source records. Privacy-reduced prediction ledgers,
aggregate metrics, and non-textual protocol identifiers belong under `results/`.

The upstream AnnoMI repository does not currently provide a separate dataset licence in this
project's recorded provenance. Review [the data card](../docs/DATA_CARD.md) and
[third-party notices](../THIRD_PARTY_NOTICES.md) before reuse or redistribution.
