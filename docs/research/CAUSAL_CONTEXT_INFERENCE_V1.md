# Paired source inference: flat causal context versus target-only RoBERTa

The registered 10,000-resample paired source bootstrap does **not** support flat causal context as
an improvement over target-only RoBERTa under the predeclared success gate.

| Source-balanced metric | Causal - target-only | Paired 95% source-bootstrap interval |
|---|---:|---:|
| macro-F1 | +0.0056 | [-0.0040, +0.0154] |
| Brier (lower is better) | +0.0041 | [-0.0053, +0.0132] |
| log loss (lower is better) | +0.0294 | [+0.0096, +0.0501] |

The hard-label F1 point estimate is slightly higher, but its interval includes zero and its upper
endpoint remains below the registered minimum improvement of +0.02. The source-balanced log-loss
interval is entirely positive, supporting a deterioration in probabilistic performance. The Brier
interval is inconclusive.

The 119 source-video clusters were sampled with replacement as paired units. Each sampled source
retained all of its utterances, and each source instance received equal total mass. Seeds were not
treated as sampling units. The interval is an equal-tailed percentile interval using bootstrap seed
`20260904`.

Four of five matched-seed macro-F1 differences are positive: +0.0061 (17), +0.0019 (42), +0.0079
(101), -0.0054 (314), and +0.0040 (2718). Descriptively, individual-source macro-F1 improves for
44 sources, declines for 45, and ties for 30; the median source difference is zero. These counts are
diagnostic rather than a separate hypothesis test.

## Registered gate

- Minimum +0.02 source-balanced macro-F1: fail (+0.0056).
- Paired source-bootstrap macro-F1 interval excludes zero: fail.
- At least four positive matched seeds: pass (four of five).
- Source-balanced Brier degradation no greater than +0.01: pass (+0.0041 point estimate).
- No class collapse: pass.

Overall gate: **fail**. Flat causal concatenation remains a useful control but did not meet the
criterion for replacing target-only RoBERTa. A subsequent model should preserve the target signal
explicitly and make context an ablatable residual contribution.

The complete bootstrap draw ledger, per-source diagnostic ledger, source hashes, prediction-ledger
hashes, code commit, and configuration digest are stored under
`results/research/comparisons/roberta_causal10_vs_utterance_v1/`.
