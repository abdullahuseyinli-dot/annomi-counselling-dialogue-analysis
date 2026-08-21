# Data

The raw AnnoMI counselling text is deliberately excluded from Git. The tracked
`source_manifest.json` pins the official source and records the expected checksum,
schema, row count, and transcript count.

Download and validate the data with:

```bash
python tools/download_dataset.py
```

The validated file is written to `data/raw/AnnoMI/dataset.csv`, which is ignored by Git.
Do not commit derived tables containing utterance text, video metadata, or other source
records. Aggregate metrics and non-textual protocol identifiers belong under `results/`.
