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
- Commit-locked DASH-MI, PANEL-MI, and Q-TRACE-MI evaluations, with unsuccessful acceptance gates
  retained as negative results.
- Exploratory SAFE-MI Task A/C experiments covering asymmetric transfer, fold-local adaptation,
  source-safe prototype retrieval, and cross-fitted prediction sets.
- A protocol-bound MI-TAGS public-sample overlap audit. Full external evaluation remains blocked
  pending authorized access to the complete corpus.
- Deterministic publication tables, figures, manifests, repository validation, and the documentation
  set indexed in `docs/README.md`.

### Changed

- The earlier transcript-level portfolio analysis is now explicitly development-consumed; new
  headline claims use normalized source video URL as the dependency group.
- Results are separated into supported improvements, numerical-only rankings, exploratory findings,
  and failed gates.
- Raw dialogue text, embeddings, checkpoints, and model weights remain outside version control.

## Preserved portfolio lineage

The original portfolio state is preserved by the annotated tag
`portfolio-v0.1.0-development-consumed`. Its notebook and aggregate exports remain available for
audit, but they are not the confirmatory endpoint of the source-disjoint study.
