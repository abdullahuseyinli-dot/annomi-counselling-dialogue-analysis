# Validity audit of the portfolio benchmark

The earlier repository state is available at commit
`e3ff100b3866a283146e4af41596f5a837153818`. Its aggregate results informed the later study design
and are therefore treated as exploratory.

## Strengths retained

- The upstream AnnoMI commit, checksum, schema, row count, and transcript count were pinned.
- Dialogue transcripts, rather than utterance rows, defined the original train/test boundary.
- The original analysis reported class-specific metrics, calibration, grouped uncertainty, and
  explicit clinical-use limitations.
- Raw counselling text was excluded from version control.

## Threats requiring a new protocol

1. The project-created transcript partition was called an official split, although upstream AnnoMI
   does not define it. The partition selected one balanced candidate from five label-aware folds.
2. AnnoMI's 133 transcripts come from fewer source video URLs. Grouping only by transcript allows
   a source video and its production or speaker characteristics to occur on both sides.
3. Repeated normalized utterance text crosses the portfolio split. This does not prove invalidity,
   because short conversational phrases genuinely repeat, but it requires seen/unseen and
   conflicting-label analyses.
4. The full data's annotator-level votes and fine-grained hierarchy were collapsed or unused.
5. The public evidence lacks row-level held-out probabilities, so reported inferential and
   calibration results cannot be independently reconstructed from retained predictions.
6. The `human/manual` summarisation rubric was computed by deterministic lexical and embedding
   heuristics. It is an automated composite heuristic, not human evaluation.
7. Forecasting features used eventual transcript length and gold historical labels. Such results
   are oracle/offline analyses rather than strictly causal forecasts.
8. The full experiment is a monolithic notebook and is not executed by continuous integration.
9. The topic-shift cell contains a trailing-comma defect hidden by cached aggregate exports.
10. True-label error slices used unrestricted macro-F1, which is not meaningful for a slice with
    only one supported true class; recall is the appropriate label-slice statistic.

## Consequence

The earlier result tables are unchanged. Source-disjoint results are reported from separate
versioned outputs with row-level out-of-fold predictions.
