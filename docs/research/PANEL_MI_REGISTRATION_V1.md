# PANEL-MI multi-annotator registration v1

This document and `configs/research/panel_mi_v1.json` were frozen before any model was evaluated
on the seven-transcript, ten-annotation subset. Data-shape checks were permitted before this lock;
model scores, recipe rankings, and held-out predictions were not inspected.

## Question and scope

The study asks whether text predicts the *distribution* of AnnoMI coding judgments, and whether
retaining stable anonymous-annotator tendencies improves that prediction. It contains 428
utterances, each coded by the same ten anonymous annotator IDs, across seven transcripts. Therapist
and client codes have different label spaces and are therefore modeled as two linked tasks rather
than falsely merged. The therapist task is primary; the client task is a preregistered secondary
replication.

This is label-distribution learning in the sense of [Geng (2016)](https://doi.org/10.1109/TKDE.2016.2545658).
The annotator-conditioned candidate is motivated by the crowd layer of
[Rodrigues and Pereira (2018)](https://doi.org/10.1609/aaai.v32i1.11506), annotator
representations in [Deng et al. (2023)](https://aclanthology.org/2023.findings-emnlp.832/), and
the item-annotator aggregation design of
[DisCo](https://aclanthology.org/2023.findings-acl.287/). These precedents prevent an unsupported
claim that annotator conditioning itself is novel.

## Leakage boundary

The outer design exhaustively holds out one complete transcript. Recipe selection repeats
leave-one-transcript-out validation inside the remaining six transcripts. PCA and scaling are fit
again on each active training partition. The encoder is pinned, frozen RoBERTa-base and receives
only the role-prefixed current utterance. Held-out text is encoded but never used to fit encoder or
preprocessing parameters. Held-out votes are used only after predictions exist.

Items, not their 4,280 annotation rows, are the prediction units. Transcript-balanced weights give
each dialogue equal influence during fitting and evaluation. All ten votes remain visible as an
empirical target distribution; they are never collapsed for the primary score.

## Registered systems

1. `transcript_balanced_prior` averages one vote distribution per training transcript.
2. `hard_linear` is the aggregation ablation: a linear head trained on deterministic plurality
   labels, with registered label order breaking the few exact ties.
3. `soft_linear` is a convex label-distribution head trained on fractional vote mass.
4. `panel_mi` starts from a soft linear head, then learns centered annotator biases and a low-rank
   item-by-annotator interaction. It jointly optimizes individual-vote likelihood and the aggregate
   vote-distribution likelihood. At inference, its ten annotator-head probabilities are averaged.

PANEL-MI estimates the response distribution of this observed anonymous panel. It does not infer
annotator demographics, clinical truth, or a population-wide consensus.

## Endpoints and inference

The primary endpoint is transcript-balanced vote log score. Brier score, Jensen-Shannon divergence,
plurality macro-F1, predicted-versus-observed entropy association, and entropy absolute error are
secondary. The registered candidate comparison is PANEL-MI versus soft-linear on therapist turns.

Probabilities are averaged over five fixed seeds before inference. The seven held-out transcripts
are the only sampling units: a 10,000-resample paired cluster bootstrap supplies the interval, and
all 128 sign assignments supply an exact one-sided sign-flip test. A successful candidate must
improve at least five transcripts, have a negative log-score delta with an upper interval below
zero, pass the exact test at 0.05, and not worsen Jensen-Shannon divergence. With only seven
clusters, uncertainty and individual transcript effects take precedence over a binary gate.
