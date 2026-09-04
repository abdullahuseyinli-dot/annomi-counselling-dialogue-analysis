# Result lineage

The public evidence is a curated subset of the completed run artifacts. File-level hashes
are recorded in `results/provenance.json`; the executed source pipeline is identified only by its
SHA-256 digest, so no machine-specific path or personal identifier is required.

The headline lineage is:

1. `results/protocol/official_split.json` fixes transcript membership.
2. `results/protocol/roberta_config.json` fixes the selected encoder configuration.
3. `results/main/model_comparison.csv` fixes held-out metrics.
4. `results/main/metric_deltas.csv` records direct arithmetic differences.
5. `results/main/grouped_significance.csv` records transcript-grouped uncertainty.
6. `results/main/mcnemar_summary.csv` records paired item correctness.
7. `results/main/calibration.csv` and `calibration_summary.json` record probability quality.

A stale narrative sentence had described BERTopic + MMR as the overall summarisation winner.
The exported aggregate table shows the opposite: KMeans + MMR scores 2.790 overall
usefulness versus 2.445. The repository text follows the recorded table and retains both
methods' component scores so the trade-off remains auditable.

## Source-disjoint research lineage

The `results/research/` tree is a second, independent evidence lineage governed by
`configs/research/protocol_v1.json`. Its headline chain is:

1. `gate0/data_audit.json` and `gate0/source_folds_v1.json` bind the pinned data to 119 source
   groups and five exhaustive outer folds.
2. `baseline_v1/predictions.csv` reconstructs all nested sparse baselines.
3. `neural_v1/{roberta_utterance,roberta_flat_causal10,dash_mi}/` retains per-seed and seed-ensemble
   out-of-source probabilities, selection traces, code commits, configurations, and metrics.
4. `comparisons/*/` retains 10,000 paired source-bootstrap draws and per-source effects for every
   registered comparison.
5. `multiannotator_v1/panel_mi/` retains the seven-transcript PANEL-MI selection trace, all final
   seed predictions, ensemble predictions, distribution metrics, paired cluster intervals, and all
   128 sign-flip assignments summarized exactly.
6. `publication_v1/` contains compact tables derived from those immutable summaries. Its manifest
   binds every table and figure to its input hashes and builder hash.
7. `ac_v1/baselines/` and `ac_v1/qtrace_mi/` retain the registered Task A/C baselines, 100 neural
   fits summarized as per-seed and ensemble out-of-source ledgers, calibration records, prediction
   sets, and 5,000 paired source-bootstrap draws.
8. `publication_ac_v1/` derives exact Task A/C tables and figures from the immutable summaries. Its
   independent manifest binds the inputs, tables, figures, and builder.
9. `safe_mi_v2/` retains the registered exploratory staged screen, 110 neural fits, five-seed final
   ledgers, source-safe prototype selections, calibrated prediction sets, and 5,000-draw
   paired-source inference. Failed final gates are immutable outputs.
10. `safe_mi_v2_1/` retains the separately registered post-hoc Task A/C audit, including 20 missing
    one-way-model fits and outer-cross-fitted prediction sets. Its stopping rule forbids further
    AnnoMI architecture selection.
11. `mi_tags_external_v1/` retains the protocol-bound public-sample overlap audit. The raw official
    samples remain ignored local evidence; the tracked audit records their hashes and quarantines.
    `publication_safe_mi_v2/` derives compact tables and figures from these immutable summaries.

Research validators recompute metrics from row-level ledgers rather than trusting aggregate JSON.
The prediction ledgers deliberately exclude utterance text and anonymous annotator identity. Raw
data, embeddings, and checkpoints remain ignored local artifacts.

## Interpretation lineage

Three distinctions are intentionally preserved:

- **Supported improvement:** target-only RoBERTa over TF-IDF; soft-linear label-distribution
  prediction over a transcript-balanced prior on both multi-annotator tasks.
- **Numerical-only leader:** causal-history RoBERTa has the highest observed classification
  macro-F1, but its interval versus target-only includes zero and its log loss is worse.
- **Negative registered candidates:** DASH-MI does not add supported contextual value, and PANEL-MI
  does not improve its primary therapist vote log score despite better JSD/entropy diagnostics.
  Q-TRACE-MI does not pass its joint Task A/C gate: the Task A point gain is uncertain and the
  Task C effect is reliably negative relative to C-only.
- **Task-specific Task A/C leaders:** Q-TRACE-MI is the numerical non-oracle Task A leader at
  0.7000 t10 balanced accuracy in the registered Q-TRACE study; the post-hoc one-way SAFE-MI model
  later reaches 0.7389 but fails its calibration constraint and has an interval crossing zero
  against A-only. The updated frozen-GRU Task C baseline reaches 0.4359; no SAFE-MI candidate beats
  it on macro-F1.
- **External boundary:** 9 of 12 official MI-TAGS public-sample records trigger the locked possible
  AnnoMI-overlap quarantine. The sample is insufficient for performance evaluation, so external
  confirmation remains explicitly incomplete pending authorized full-corpus access.

No aggregate ranking may erase these inference and calibration qualifications.
