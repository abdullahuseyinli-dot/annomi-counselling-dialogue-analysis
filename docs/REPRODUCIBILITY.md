# Reproducibility

The project separates quick software verification from data-dependent reconstruction and expensive
neural reproduction. Run commands from the repository root. `uv.lock` is the maintained dependency
resolution; the requirements files are preserved environment records rather than the release lock.

## 1. Software verification

This level does not need raw dialogues or a GPU:

```bash
uv sync --locked --extra dev --extra neural-cpu
uv run python -m pytest --cov=annomi_research --cov-report=term-missing
uv run ruff check src tests tools
uv run ruff format --check src tests tools
uv run mypy
uv run python tools/validate_repository.py
```

Passing these checks establishes software and repository consistency only. It does not reproduce a
performance claim.

## 2. Data and retained-evidence validation

Download both pinned AnnoMI variants, then reconstruct the tracked evidence contracts:

```bash
uv run python tools/download_dataset.py --variant simple
uv run python tools/download_dataset.py --variant full
uv run python -m annomi_research audit-data
uv run python -m annomi_research make-splits
uv run python -m annomi_research validate
```

Existing evidence is create-only. An identical rerun is a no-op; a changed payload must use a new
lineage rather than overwrite a prior result.

## 3. Classical reproduction

The nested source-grouped sparse baselines run through:

```bash
uv run python -m annomi_research run-baselines
uv run python -m annomi_research run-ac-baselines
```

The label-distribution study uses frozen neural embeddings followed by low-dimensional heads. It is
not a data-free software test and may need the `neural` extra and access to the pinned pretrained
encoder even though the downstream heads can run on CPU after embedding extraction.

## 4. Neural reproduction

Install the locked neural environment and pass the recorded CUDA/BF16 engineering gate before full
runs:

```bash
uv sync --locked --extra dev --extra neural
uv run python -m annomi_research check-neural-env
uv run python -m annomi_research smoke-neural --model roberta_utterance
uv run python -m annomi_research smoke-neural --model roberta_flat_causal10
uv run python -m annomi_research smoke-dash
uv run python -m annomi_research smoke-panel
uv run python -m annomi_research smoke-qtrace
uv run python -m annomi_research smoke-safe-mi
```

After the smoke gates pass, the full entry points are:

```bash
uv run annomi-research run-neural --model roberta_utterance
uv run annomi-research run-neural --model roberta_flat_causal10
uv run annomi-research run-dash
uv run annomi-research run-panel
uv run annomi-research run-qtrace
uv run annomi-research run-safe-mi
uv run annomi-research run-safe-mi-extension
```

These commands perform nested folds and fixed multi-seed evaluation and can be expensive. Run them
only against a fresh output lineage if the recorded environment or payload would differ. Hardware
observations from the completed campaign are in [HARDWARE.md](HARDWARE.md); they are provenance,
not a portable minimum specification.

## 5. Publication assets

The three builders derive compact tables and figures from retained summaries and record input and
output hashes:

```bash
uv run python tools/build_research_assets.py
uv run python tools/build_ac_assets.py
uv run python tools/build_safe_mi_assets.py
```

Then rerun `annomi_research validate`, the test suite, and repository validation. Do not manually
edit generated tables or figures.

## 6. Optional MI-TAGS evaluation

The tracked public-sample audit can be reconstructed only when the exact manifest-matched sample
files are present under ignored `data/raw/MI-TAGS/`. It is not a performance evaluation. The full
external protocol remains blocked pending authorized access to the complete MI-TAGS corpus, and its
frozen overlap, split, mapping, and no-retuning rules must be applied before any external metric is
reported.
