# Experiment protocol

## Primary question

Predict the main behaviour label for each therapist utterance: `reflection`, `question`,
`therapist_input`, or `other`. The comparison is between an elastic-net sparse baseline
and a fine-tuned RoBERTa-base encoder with preceding conversational context.

## Data contract

The pinned AnnoMI-simple file has 9,699 utterances across 133 transcripts. Transcript IDs,
not utterance rows, define the split boundary. The official split contains 102 training
transcripts and 31 held-out transcripts; the two sets are disjoint and exhaustive.

Raw counselling text and source-video metadata are local-only. Version control contains
the source manifest, transcript IDs used by the split, and aggregate results.

## Selection and evaluation

1. Build chronological context within each transcript only.
2. Keep the official held-out transcripts outside model and hyperparameter selection.
3. Tune with transcript-grouped folds on the training partition.
4. Confirm the selected RoBERTa configuration across seeds 17, 42, and 101.
5. Evaluate once on the fixed 973-item held-out set.
6. Resample and permute at transcript level for uncertainty estimates.
7. Use paired item correctness for the exact McNemar comparison.

The selected encoder uses ten preceding turns, a 384-token maximum sequence length,
learning rate `2e-5`, weight decay `0.01`, effective batch size 16 through gradient
accumulation, and three epochs. `results/protocol/roberta_config.json` is authoritative.

## Metrics

Macro-F1 is the primary class-balanced metric. Accuracy and weighted F1 are reported for
comparability. Multiclass Brier score and 15-bin expected calibration error describe
probabilistic quality. Per-class precision, recall, and F1 remain visible to prevent the
aggregate score from hiding weak classes.

## Calibration

Temperature scaling uses a fitted temperature of 1.13249. It improves RoBERTa ECE from
0.1073 to 0.0877 and Brier score from 0.2938 to 0.2860. Because a positive scalar
temperature does not change the argmax, classification metrics remain unchanged.

## Supporting analyses

Summary extraction compares KMeans + MMR with BERTopic + MMR using ROUGE, BERTScore, and
a five-component support rubric. The aggregate table selects KMeans + MMR on overall
usefulness. Transcript-quality classification and next-behaviour forecasting are retained
as separate extensions; neither is used to select the primary classifier.

## Reproduction policy

A new run must record package versions, random seeds, split hash, configuration, hardware,
and all aggregate outputs. Do not replace tracked evidence merely because a rerun differs.
Investigate version, hardware, preprocessing, and stochastic effects, then commit a new
evidence lineage with a documented comparison.
