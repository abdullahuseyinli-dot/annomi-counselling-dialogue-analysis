# RoBERTa flat causal-context result v1

Status: completed prospective five-fold, five-seed evaluation under
`dash-mi-source-cv-v1` and `annomi-neural-v1`.

The seed-probability ensemble obtains source-balanced macro-F1 **0.8163** and ordinary
utterance macro-F1 **0.7981** over all 4,882 out-of-source therapist utterances from 119 source
videos. This is the highest completed point estimate in the registered benchmark. It exceeds the
target-only RoBERTa ensemble by **0.0056** source-balanced macro-F1 and the sparse target-utterance
baseline by **0.0991**. The causal-versus-target-only difference is small and is not treated as a
supported gain until the separately registered paired source bootstrap is complete.

| Metric | TF-IDF target | RoBERTa target | RoBERTa causal-10 | Causal - target |
|---|---:|---:|---:|---:|
| source-balanced macro-F1 | 0.7172 | 0.8108 | **0.8163** | **+0.0056** |
| utterance macro-F1 | 0.6895 | 0.7912 | **0.7981** | **+0.0068** |
| source-balanced Brier (lower is better) | 0.3906 | **0.2796** | 0.2838 | +0.0041 |
| source-balanced log loss (lower is better) | 0.7670 | **0.5587** | 0.5880 | +0.0294 |
| equal-frequency ECE (lower is better) | **0.0237** | 0.0658 | 0.0800 | +0.0141 |
| worst-20% source log-loss CVaR (lower is better) | 1.2391 | **1.1006** | 1.2039 | +0.1033 |

Flat causal context has higher point estimates for the two F1 endpoints and worse values for every
recorded probabilistic metric relative to target-only RoBERTa. The paired F1 interval later crossed
zero. This mixed pattern motivated the registered DASH-MI candidate's explicit target path and gated
context residual.

Per-class F1 for the causal ensemble is 0.8898 (`other`), 0.8516 (`question`), 0.7826
(`reflection`), and 0.6683 (`therapist_input`). Relative to target-only RoBERTa, causal context has
higher F1 point estimates for `other`, `reflection`, and `therapist_input` by 0.0050, 0.0156, and
0.0118, respectively, while the `question` point estimate is 0.0051 lower.

Four of five causal seeds have a higher source-balanced macro-F1 than the matching target-only
seed. Causal seed scores are 0.8139 (17), 0.8091 (42), 0.8153 (101), 0.8016 (314), and 0.8087
(2718). The probability ensemble reaches 0.8163.

The nested procedure selected the following fold-specific configurations:

| Outer fold | Recipe | Epochs |
|---:|---|---:|
| 0 | `c_lr1e-5_len384` | 3 |
| 1 | `c_lr2e-5_len256` | 3 |
| 2 | `c_lr1e-5_len256` | 5 |
| 3 | `c_lr2e-5_len384` | 4 |
| 4 | `c_lr1e-5_len256` | 4 |

The repeated-text slice remains difficult: macro-F1 is 0.4075 across 1,116 rows, compared with
0.7388 across 3,766 unseen-text rows. This is descriptive and does not identify a causal effect of
text repetition.

## Evidence and provenance

- `results/research/neural_v1/roberta_flat_causal10/predictions_by_seed.csv` contains 24,410
  out-of-fold predictions (4,882 times five seeds).
- `predictions_seed_ensemble.csv` contains one seed-averaged prediction per therapist utterance.
- `selection.json` retains all 60 inner training traces and five fold choices.
- The code commit used for training is `85cd54e36d87ac200c69251d702af83f20d40fb2`.
- The pinned pretrained revision is `e2da8e2f811d1448a5b465c236feacd80ffbac7b`.
- The per-seed ledger SHA-256 is
  `aa50e9f1cfda677cf5539e9d61ee8e1d703ddbc89cdcd2ea155b7785211953bb`.
- The ensemble ledger SHA-256 is
  `8fb7b6a79982832c50dd19ac6d93aaac1247e7658070f4d12f0b15dde62b4f85`.
- The invocation took 18,051 seconds (about 5.01 hours) on the recorded RTX PRO 3000 Blackwell
  laptop GPU environment.

The result concerns automatic coding of public AnnoMI demonstrations under this exact protocol. It
does not establish clinical validity, treatment efficacy, therapist quality, or state of the art
against papers using different data, modalities, labels, or split units.
