# Contributing

Contributions should preserve the separation between software changes, commit-locked protocols,
and completed evidence. Start by reading the [documentation index](docs/README.md), the
[locked protocol](docs/research/LOCKED_PROTOCOL.md), and the
[result-lineage policy](docs/RESULT_LINEAGE.md).

## Local setup

Use Python 3.11 or 3.12 and the checked-in `uv.lock`:

```bash
uv sync --locked --extra dev --extra neural-cpu
uv run python -m pytest --cov=annomi_research --cov-report=term-missing
uv run ruff check src tests tools
uv run ruff format --check src tests tools
uv run mypy
uv run python tools/validate_repository.py
```

Use the `analysis` extra for notebooks and the CUDA-backed `neural` extra for full model runs. The
two requirements files are preserved environment records; `pyproject.toml` and `uv.lock` define
the maintained environment.

## Evidence rules

- Do not commit raw dialogue text, source metadata extracts, embeddings, checkpoints, model weights,
  credentials, or local machine paths.
- Do not overwrite a completed protocol, probability ledger, summary, manifest, or failure record.
  A materially different run needs a new named lineage and an accompanying comparison.
- Keep split construction, preprocessing, selection, calibration, and inference group-safe. Test
  sources must not influence training or model choice.
- Record failed runs and failed acceptance gates when they affect interpretation.
- Treat performance numbers as claims that require a tracked ledger or deterministic manifest.
- Do not weaken clinical-use, licensing, or external-validation limitations to improve presentation.

## Pull-request checklist

- Tests, lint, repository validation, and relevant evidence reconstruction pass.
- New commands and configuration fields have focused tests.
- Local links and documented commands are current.
- Data and model dependencies are pinned where practical and credited in
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- The change is classified as confirmatory, exploratory, post-hoc, or engineering-only.
- Any result change is added as a new lineage rather than replacing prior evidence.

Small documentation and test improvements are welcome. Changes that introduce a new experiment
should include a machine-readable protocol, stopping rule, comparison target, and claim boundary
before outcomes are inspected.
