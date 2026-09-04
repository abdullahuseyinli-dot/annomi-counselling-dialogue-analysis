# Changelog

This changelog records material changes to the research package. It is not a release announcement.
The current repository state is a development candidate: it has no matching release tag, DOI, or
public archive deposit.

## Unreleased

### Added

- A five-fold, source-disjoint benchmark for therapist-behaviour classification, including sparse
  and neural systems, paired source-cluster inference, calibration, and row-level probability
  ledgers.
- A leave-one-transcript-out study of label-distribution learning from the existing ten-annotator
  subset.
- DASH-MI, PANEL-MI, and Q-TRACE-MI evaluations whose protocols were fixed in Git before testing,
  including results for candidates that did not meet their prespecified acceptance criteria.
- Exploratory SAFE-MI Task A/C experiments covering asymmetric transfer, fold-local adaptation,
  source-safe prototype retrieval, and cross-fitted prediction sets.
- An MI-TAGS public-sample overlap audit conducted under a protocol written before data access. Full
  external evaluation remains blocked pending authorized access to the complete corpus.
- Deterministic publication tables, figures, manifests, repository validation, and the documentation
  set indexed in `docs/README.md`.

### Changed

- The new benchmark groups data by normalized source-video URL. The older transcript-level
  analysis remains available for reference.
- Results are separated into supported improvements, numerical-only rankings, exploratory findings,
  and failed gates.
- Raw dialogue text, embeddings, checkpoints, and model weights remain outside version control.

## Earlier portfolio snapshot

The original portfolio state remains available at commit `e3ff100`. Its notebook and aggregate
exports use the earlier evaluation design.
