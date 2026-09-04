# DASH-MI result v1

Status: completed adaptive five-fold, five-seed evaluation under
`dash-mi-source-cv-v1` and the prospectively registered `annomi-dash-mi-source-cv-neural-v1`
configuration.

The DASH-MI seed-probability ensemble obtains source-balanced macro-F1 **0.8116** and ordinary
utterance macro-F1 **0.7926** over all 4,882 out-of-source therapist utterances from 119 source
videos. It does not exceed the completed flat causal-context RoBERTa control (0.8163), and its
**+0.0009** point difference over the independently trained target-only RoBERTa control (0.8108)
is too small to support a superiority claim without the separately registered paired source
bootstrap.

| Metric | TF-IDF target | RoBERTa target | RoBERTa causal-10 | DASH-MI |
|---|---:|---:|---:|---:|
| source-balanced macro-F1 | 0.7172 | 0.8108 | **0.8163** | 0.8116 |
| utterance macro-F1 | 0.6895 | 0.7912 | **0.7981** | 0.7926 |
| source-balanced Brier (lower is better) | 0.3906 | **0.2796** | 0.2838 | 0.2828 |
| source-balanced log loss (lower is better) | 0.7670 | **0.5587** | 0.5880 | 0.5883 |
| equal-frequency ECE (lower is better) | **0.0237** | 0.0658 | 0.0800 | 0.0810 |
| worst-20% source log-loss CVaR (lower is better) | 1.2391 | **1.1006** | 1.2039 | 1.2076 |

The model therefore does not become the benchmark leader. It nearly matches the two simpler
RoBERTa controls on hard-label F1, but target-only RoBERTa retains clearly better probabilistic
metrics and flat causal concatenation retains the highest F1 point estimate. DASH-MI remains an
informative adaptive experiment and ablation platform rather than a promoted replacement.

## Matched context ablation

Removing DASH-MI's context residual at inference while keeping the same jointly trained weights
produces source-balanced macro-F1 0.8106. The full model's **+0.0011** difference is negligible as
a point estimate. Context changes only 26 of 4,882 ensemble hard predictions (0.53%).

| Metric | Target-only ablation | Full DASH-MI | Full - ablation |
|---|---:|---:|---:|
| source-balanced macro-F1 | 0.8106 | 0.8116 | +0.0011 |
| utterance macro-F1 | 0.7915 | 0.7926 | +0.0011 |
| source-balanced Brier (lower is better) | 0.2831 | 0.2828 | -0.0002 |
| source-balanced log loss (lower is better) | **0.5874** | 0.5883 | +0.0009 |
| equal-frequency ECE (lower is better) | 0.0810 | 0.0810 | -0.0000 |
| worst-20% source log-loss CVaR (lower is better) | **1.2018** | 1.2076 | +0.0058 |

Context slightly improves F1 and Brier but slightly worsens log loss and worst-source tail loss.
Only three of five matched seeds improve: -0.0005 (17), -0.0019 (42), +0.0005 (101), +0.0012
(314), and +0.0007 (2718). These are descriptive diagnostics, not independent tests. The paired
source bootstrap must determine whether any difference is distinguishable from sampling noise.

The learned gate remains conservative: its mean is 0.1268, with mean context-residual L2 norm
0.1557. Mean normalized context-attention entropy is 0.9674 and the mean maximum attention weight
is 0.0118. Together with the low hard-decision change rate, these diagnostics indicate that the
model mostly relies on its explicitly supervised target path.

Per-class F1 for the full ensemble is 0.8862 (`other`), 0.8557 (`question`), 0.7713
(`reflection`), and 0.6573 (`therapist_input`). Relative to the matched target-only ablation,
context changes these by -0.0005, -0.0000, +0.0008, and +0.0042, respectively. The rare
`therapist_input` class remains the principal performance bottleneck.

## Nested choices and disagreement supervision

The leakage-safe inner procedure selected:

| Outer fold | Recipe | Epochs |
|---:|---|---:|
| 0 | `dash_hist128_hard` | 4 |
| 1 | `dash_hist128_votes` | 2 |
| 2 | `dash_hist256_votes` | 4 |
| 3 | `dash_hist128_votes` | 4 |
| 4 | `dash_hist256_hard` | 3 |

Vote-distribution supervision wins three of five outer-fold selections, while hard labels win two;
128-token history wins three and 256-token history wins two. This heterogeneity is descriptive.
Only 216 utterances from seven transcripts have ten-annotator vote distributions, so the result
does not establish a general benefit from disagreement-aware supervision. A separate analysis of
probabilistic agreement with those votes is required.

## Evidence and provenance

- `results/research/neural_v1/dash_mi/predictions_by_seed.csv` contains 24,410 out-of-fold
  predictions (4,882 times five seeds), including matched target-only probabilities and context
  diagnostics.
- `predictions_seed_ensemble.csv` contains one seed-averaged prediction per therapist utterance.
- `selection.json` retains all 60 inner training traces and five outer-fold choices.
- The code commit used for training is `5523e8f1c05b60617f7d6e5ecab98cf12279bed5`.
- The pinned pretrained revision is `e2da8e2f811d1448a5b465c236feacd80ffbac7b`.
- The per-seed ledger SHA-256 is
  `f503530b1048206ff9ba15fabaffe944f858dda0e3006e4694e92f629ceb5a86`.
- The ensemble ledger SHA-256 is
  `8e6cf0e96c2c70d3b5fe2f2f5a0480fa4ee1a33430d93b28e014ea2a12cab619`.
- The complete invocation took 19,607 seconds (about 5.45 hours) on the recorded RTX PRO 3000
  Blackwell laptop GPU environment.

DASH-MI was designed after inspecting the completed RoBERTa controls. Its result is adaptive and
requires external replication even if a later secondary analysis is favorable. The benchmark
concerns automatic coding of public AnnoMI demonstrations under this exact protocol; it does not
establish clinical validity, treatment efficacy, therapist quality, or state of the art against
work using different data, modalities, labels, or split units.
