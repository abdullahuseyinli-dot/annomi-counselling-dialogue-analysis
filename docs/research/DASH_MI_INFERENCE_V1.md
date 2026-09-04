# Paired source inference for DASH-MI

The registered primary comparison does **not** support DASH-MI as an improvement over the
independently trained target-only RoBERTa control. Two registered secondary comparisons likewise
do not support a measurable benefit from the context-residual architecture or a difference from
flat causal-context RoBERTa.

## Primary comparison: DASH-MI versus target-only RoBERTa

| Source-balanced metric | DASH-MI - target-only | Paired 95% source-bootstrap interval |
|---|---:|---:|
| macro-F1 | +0.0009 | [-0.0058, +0.0073] |
| Brier (lower is better) | +0.0032 | [-0.0014, +0.0079] |
| log loss (lower is better) | +0.0296 | [+0.0168, +0.0435] |

The hard-label F1 point estimate is effectively tied and its interval includes zero. Its upper
endpoint is far below the predeclared minimum improvement of +0.02. The log-loss interval is
entirely positive, supporting worse probabilistic performance for DASH-MI. The Brier interval is
inconclusive.

Four of five matched-seed macro-F1 differences are positive: +0.0040 (17), +0.0002 (42), -0.0004
(101), +0.0016 (314), and +0.0019 (2718). Descriptively, DASH-MI improves individual-source
macro-F1 for 33 sources, declines for 32, and ties for 54; the median source difference is zero.

The registered primary gate therefore resolves as follows:

- Minimum +0.02 source-balanced macro-F1: fail (+0.0009).
- Paired source-bootstrap macro-F1 interval excludes zero: fail.
- At least four positive matched seeds: pass (four of five).
- Source-balanced Brier degradation no greater than +0.01: pass (+0.0032 point estimate).
- No class collapse: pass.

Overall primary gate: **fail**. DASH-MI did not meet the criterion for replacing target-only
RoBERTa.

## Secondary mechanism ablation: full DASH-MI versus its target path

| Source-balanced metric | Full - target-path ablation | Paired 95% source-bootstrap interval |
|---|---:|---:|
| macro-F1 | +0.0011 | [-0.0008, +0.0031] |
| Brier (lower is better) | -0.0002 | [-0.0011, +0.0008] |
| log loss (lower is better) | +0.0009 | [-0.0016, +0.0034] |

All three intervals include zero. Only three of five matched seeds have positive F1 differences.
At source level, the full model improves 14 sources, declines on 5, and ties on 100; the median
difference is zero. This is consistent with the architecture diagnostics: the context residual
changes only 26 of 4,882 ensemble hard decisions. The result does not support a measurable benefit
from the learned context branch under this protocol.

This ablation compares two inference paths from the same jointly trained weights. It isolates the
incremental prediction-time residual but does not estimate what would happen if the auxiliary
target loss, context dropout, or two-stream training were removed and the model were retrained.

## Secondary ranking: DASH-MI versus flat causal-context RoBERTa

| Source-balanced metric | DASH-MI - flat causal | Paired 95% source-bootstrap interval |
|---|---:|---:|
| macro-F1 | -0.0047 | [-0.0146, +0.0047] |
| Brier (lower is better) | -0.0009 | [-0.0103, +0.0085] |
| log loss (lower is better) | +0.0003 | [-0.0205, +0.0204] |

Every interval includes zero. Only one of five matched seed differences favors DASH-MI. At source
level, DASH-MI improves 40 sources, declines on 54, and ties on 25; the median difference is zero.
Flat causal RoBERTa remains the numerical F1 leader (0.8163 versus 0.8116), but this secondary
analysis does not distinguish the models statistically.

## Interpretation and evidence

Simple flat causal concatenation gives the highest F1 point estimate, but its F1 interval includes
zero and its log-loss deterioration versus target-only RoBERTa is supported. DASH-MI does not
repair that probabilistic trade-off, and the within-model context effect is unresolved.

The 119 source-video clusters were sampled with replacement as paired units. Each sampled source
retained all of its utterances and received equal total mass. Every comparison used 10,000
resamples, an equal-tailed percentile interval, and bootstrap seed `20260904`. Seeds were matched
stability diagnostics, not sampling units. Complete draw ledgers, per-source ledgers, configuration
digests, input-ledger hashes, code commit, and output hashes are stored under:

- `results/research/comparisons/dash_mi_vs_utterance_v1/`
- `results/research/comparisons/dash_mi_context_ablation_v1/`
- `results/research/comparisons/dash_mi_vs_causal10_v1/`

DASH-MI was adaptively designed after the controls were observed. These intervals quantify this
dataset and protocol; they are not external replication and do not establish clinical validity.
