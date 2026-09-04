# DASH-MI execution registration v1

`configs/research/dash_mi_v1.json` was committed before DASH-MI was run or its outer-fold
predictions were inspected. The detailed architecture was designed after reviewing the target-only
and flat-context RoBERTa results, so this evaluation is exploratory and requires replication on
independent counselling data.

DASH-MI means **Disagreement-Aware, Source-Hardened Motivational Interviewing classifier**. It has
two separately encoded, shared-weight RoBERTa streams:

1. A target-only stream receives the current therapist utterance and produces an independently
   supervised target prediction.
2. A causal-history stream receives at most ten preceding, role-marked turns in recent-first order.
   Target-conditioned multi-head attention queries the history with the target representation.
3. A learned gate fuses the attended history into a zero-initialized residual logit correction.
   At initialization the full prediction equals the target-only prediction. History can be removed
   at inference to produce a paired within-model ablation.

The training loss averages the full-model cross-entropy with an auxiliary target-only
cross-entropy. Fifteen percent of history examples are dropped during training to discourage a
brittle dependence on dialogue context. Source-by-class weighting is fitted only on the active
training partition.

The four-recipe nested grid crosses 128 versus 256 history tokens with hard labels versus the
empirical `AnnoMI-full` vote distribution. The latter requires no new annotation: 216 therapist
utterances in seven transcripts already have ten expert annotations. Single-annotated examples
remain one-hot. Test-fold votes are never used for fitting, recipe selection, early stopping, or
the hard-label primary endpoint.

Each outer fold uses three source-grouped inner folds, the fixed selection seed, and the same
early-stopping rule as the RoBERTa controls. The chosen recipe is retrained for the rounded median
inner best epoch under seeds 17, 42, 101, 314, and 2718. Probabilities are averaged before scoring.
All predictions, target-only ablations, context diagnostics, selection traces, hashes, and failures
are retained.

The primary comparison is DASH-MI versus target-only RoBERTa under the locked paired source-level
bootstrap and success gate. A higher point estimate alone is insufficient for promotion.
