# Model card: portfolio therapist-behaviour classifier

The evidence in this card is preserved as development-consumed. It predates the
source-video-grouped research protocol and is not a confirmatory source-disjoint result.

## Model

The headline model fine-tunes `FacebookAI/roberta-base` for four-way therapist-behaviour
classification. Each example combines the current therapist utterance with up to ten
preceding turns, truncated to 384 tokens. Selection uses transcript-grouped training folds
and confirmation across three seeds.

## Intended use

This model is suitable for reproducible NLP research, method comparison, and aggregate
exploratory analysis of the AnnoMI benchmark. It may help researchers study how context
affects motivational-interviewing label prediction.

It is not designed for clinical diagnosis, patient triage, therapist ranking, live
intervention, automated feedback to clients, or decisions affecting access to care.

## Evaluation

On 973 therapist utterances from 31 held-out transcripts, the model reaches 0.8181 accuracy
and 0.7744 macro-F1. The elastic-net baseline reaches 0.7770 and 0.7358 respectively.
Transcript-grouped 95% intervals for the gains exclude zero, and exact McNemar testing gives
p = 0.0006. Temperature scaling improves probabilistic calibration but remains less
calibrated by ECE than the sparse baseline.

## Limitations

- AnnoMI contains demonstrations sourced from public videos; it is not a representative
  sample of real-world clinical care.
- Labels simplify nuanced dialogue acts into four classes.
- The held-out partition covers 31 transcripts, so topic- and speaker-specific uncertainty
  remains material.
- Conversational context can encode sensitive information and source-specific artefacts.
- Results do not establish clinical validity, fairness across demographic groups, or safe
  performance under distribution shift.

## Data and privacy

Raw utterance text, source-video metadata, predictions, logits, and checkpoints are not
distributed. The repository tracks only aggregate metrics and transcript identifiers needed
to reconstruct the split. Users are responsible for reviewing the upstream data terms and
applying appropriate governance to any derivative dataset or model.
