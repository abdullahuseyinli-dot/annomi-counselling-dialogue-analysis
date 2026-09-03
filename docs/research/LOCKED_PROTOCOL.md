# DASH-MI source-disjoint protocol v1

`configs/research/protocol_v1.json` is the machine-readable authority for this study. This document
explains its interpretation.

## Primary question

Estimate four-class therapist-behaviour performance on source videos absent from model fitting and
selection. The primary endpoint is source-balanced macro-F1 across five fixed outer folds. Each
video URL receives equal total weight regardless of dialogue length.

## Split and selection boundary

- Exact normalized `video_url` is the minimum dependency group.
- All five outer folds are reported; no best fold is selected.
- Tokenizers, vectorizers, class weights, calibration, early stopping, and hyperparameters are fit
  using outer-training data only.
- Inner validation is source-grouped.
- The current utterance and at most ten prior turns from its transcript are available. Later turns,
  eventual transcript length, topic, video title, quality, and gold historical codes are forbidden.
- Bidirectional context, if ever run, is an offline upper bound and not part of the primary result.

## Evidence contract

Every out-of-fold prediction records model, seed, fold, transcript ID, utterance ID, hashed source
ID, true label, predicted label, all class probabilities, seen-text status, resolved configuration,
dataset digest, split digest, and code commit. Aggregate tables must be regenerated from that ledger.

## Multi-annotator study

The 428 utterances with ten annotations form a separate seven-transcript leave-one-transcript-out
study. Vote-distribution Brier/log score, Jensen-Shannon divergence, and entropy association are
primary there. Utterances are not treated as 4,280 independent observations, and claims are bounded
to the seven source dialogues.

## Claim boundary

AnnoMI consists of public demonstrations, not a representative clinical cohort. Results concern
automatic research coding. They do not validate diagnosis, treatment recommendations, therapist
ranking, causal effects, or autonomous clinical use.
