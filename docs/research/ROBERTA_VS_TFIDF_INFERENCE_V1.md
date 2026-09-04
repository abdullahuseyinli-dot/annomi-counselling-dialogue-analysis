# Paired source inference: RoBERTa utterance versus TF-IDF

The registered 10,000-resample paired source bootstrap supports the target-utterance RoBERTa
candidate under every predeclared success criterion.

| Source-balanced metric | Candidate − baseline | Paired 95% source-bootstrap interval |
|---|---:|---:|
| macro-F1 | **+0.0935** | **[+0.0778, +0.1097]** |
| Brier (lower is better) | **-0.1109** | **[-0.1296, -0.0922]** |
| log loss (lower is better) | **-0.2083** | **[-0.2450, -0.1714]** |

The 119 source-video clusters were sampled with replacement as paired units. Each sampled source
retained all of its utterances, and each source instance received equal total mass. Seeds were not
treated as sampling units. The interval is an equal-tailed percentile interval using bootstrap seed
`20260904`.

All five individual RoBERTa seeds improve source-balanced macro-F1 over TF-IDF, with deltas from
+0.0875 to +0.0906. The probability ensemble improves by +0.0935. Descriptively, individual-source
macro-F1 improves for 90 sources, declines for 22, and ties for 7; the median source delta is
+0.0784. Those source counts are diagnostic, not an additional hypothesis test.

## Registered gate

- Minimum +0.02 source-balanced macro-F1: pass.
- Paired source-bootstrap interval excludes zero: pass.
- At least four positive seeds: pass (five of five).
- Source-balanced Brier degradation no greater than +0.01: pass (it improves by 0.1109).
- No class collapse: pass.

The ECE diagnostic still worsens, as documented in the model result, and is not erased by this gate.
Future calibration work must fit parameters without using the relevant outer-test labels.

The complete bootstrap draw ledger, per-source diagnostic ledger, source hashes, prediction-ledger
hashes, code commit, and configuration digest are stored under
`results/research/comparisons/roberta_utterance_vs_tfidf_v1/`.
