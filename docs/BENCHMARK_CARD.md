# Benchmark card

## Scope

The benchmark evaluates research coding of public motivational-interviewing demonstrations. It does
not measure clinical outcomes, treatment quality, or readiness for live use. Raw dialogue text and
model weights are not distributed.

| Track | Evaluation unit and split | Primary measure | Recorded result |
|---|---|---|---|
| Therapist behaviour | 4,882 utterances; five outer folds over 119 normalized video sources | Source-balanced macro-F1 | Target-only RoBERTa 0.8108; causal-history RoBERTa 0.8163 numerical maximum |
| Existing vote distributions | 428 utterances, ten votes each; leave one of seven transcripts out | Transcript-balanced vote log score | Soft-linear improves over the transcript prior for therapist and client turns |
| Task A: early quality metadata | 115 eligible transcripts from 108 sources at ten therapist turns | Source-balanced balanced accuracy | Post-hoc one-way model 0.7389; uncertainty crosses zero and Brier constraint fails |
| Task C: next therapist action | 4,743 source-grouped client-to-therapist handoffs | Source-balanced macro-F1 | Frozen-GRU baseline 0.4359; no SAFE-MI candidate establishes superiority |

## Main interpretation

Target-only RoBERTa improves therapist-behaviour macro-F1 over TF-IDF by 0.0935, with a paired
source-bootstrap 95% interval of [0.0778, 0.1097]. This is the supported classification improvement.
Causal history is the numerical leader at 0.8163, but its difference from target-only is uncertain
and its log loss is worse. DASH-MI does not add supported contextual value.

On the seven-transcript multi-annotator subset, soft-linear vote prediction improves log score over
the transcript prior by -0.6309 for therapist turns and -0.1113 for client turns. PANEL-MI improves
some disagreement-shape diagnostics but fails its commit-locked log-score gate.

Task A's 0.7389 result is exploratory and post-hoc, not a confirmed replacement. In Task C,
source-safe prototype retrieval reaches 0.4328 and improves Brier score relative to frozen-GRU, but
its macro-F1 interval includes zero. Cross-fitted prediction sets meet the recorded descriptive
coverage/size thresholds for frozen-GRU and adapted-GRU; they are action-label sets, not treatment
recommendations.

## Leakage controls

- Normalized video source, not utterance row, is the main dependency group.
- Inner selection and calibration remain within each outer training partition.
- Neural headline probabilities average five fixed seeds before scoring.
- Inputs are target-only or causal; future turns, eventual length, metadata, and gold histories are
  excluded where the protocol requires it.
- Paired resampling operates on source or transcript clusters, not annotation rows or seeds.
- Row-level probability ledgers and deterministic publication manifests reconstruct reported values.

## Comparability

Source-balanced metrics under source-disjoint folds are not directly interchangeable with random
utterance splits or a single held-out transcript split. No state-of-the-art, clinical-validity, or
cross-corpus claim is made. See the [study report](research/STUDY_REPORT.md) and
[model card](MODEL_CARD.md) for the complete interpretation.
