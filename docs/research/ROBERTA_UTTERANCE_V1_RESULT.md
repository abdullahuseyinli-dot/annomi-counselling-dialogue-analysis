# RoBERTa target-utterance result v1

Status: completed prospective five-fold, five-seed evaluation under
`dash-mi-source-cv-v1` and `annomi-neural-v1`.

The seed-probability ensemble obtains source-balanced macro-F1 **0.8108** and ordinary
utterance macro-F1 **0.7912** over all 4,882 out-of-source therapist utterances from 119 source
videos. The registered sparse target-utterance baseline obtains 0.7172 source-balanced macro-F1,
for an unadjusted difference of **+0.0935**. Statistical inference is deliberately deferred to the
paired source-bootstrap stage; this document does not treat the raw difference alone as a
significance claim.

| Metric | TF-IDF target baseline | RoBERTa seed ensemble | Raw difference |
|---|---:|---:|---:|
| source-balanced macro-F1 | 0.7172 | **0.8108** | **+0.0935** |
| utterance macro-F1 | 0.6895 | **0.7912** | **+0.1017** |
| source-balanced Brier (lower is better) | 0.3906 | **0.2796** | **-0.1109** |
| source-balanced log loss (lower is better) | 0.7670 | **0.5587** | **-0.2083** |
| equal-frequency ECE (lower is better) | **0.0237** | 0.0658 | +0.0422 |
| worst-20% source log-loss CVaR (lower is better) | 1.2391 | **1.1006** | **-0.1385** |

RoBERTa's higher F1 and lower Brier score coexist with worse calibration under the recorded ECE
diagnostic.

Per-class F1 for the ensemble is 0.8848 (`other`), 0.8566 (`question`), 0.7670
(`reflection`), and 0.6565 (`therapist_input`). All five individual seeds are positive relative to
the sparse baseline: source-balanced macro-F1 ranges from 0.8048 to 0.8078. Probability averaging
raises the ensemble estimate to 0.8108.

The nested procedure selected the following fold-specific configurations:

| Outer fold | Recipe | Epochs |
|---:|---|---:|
| 0 | `u_lr1e-5_len128` | 4 |
| 1 | `u_lr2e-5_len128` | 3 |
| 2 | `u_lr1e-5_len256` | 4 |
| 3 | `u_lr1e-5_len128` | 5 |
| 4 | `u_lr1e-5_len128` | 4 |

The repeated-text slice again performs worse than the unseen-text slice (macro-F1 0.3964 across
1,116 rows versus 0.7277 across 3,766 rows). This is descriptive and likely reflects short,
generic, label-ambiguous phrases; it is not evidence that duplication harms performance.

## Evidence and provenance

- `results/research/neural_v1/roberta_utterance/predictions_by_seed.csv` contains 24,410
  out-of-fold predictions (4,882 × five seeds).
- `predictions_seed_ensemble.csv` contains one seed-averaged prediction per therapist utterance.
- `selection.json` retains all 60 inner training traces and five fold choices.
- The code commit used for training is `3cf8cce6d6025f46a44d725abc90575047806097`.
- The pinned pretrained revision is `e2da8e2f811d1448a5b465c236feacd80ffbac7b`.
- The per-seed ledger SHA-256 is
  `a3c22c4329ad952d70a8290701345fcdbfef6dfd9593639c17b2af1d7f362a1c`.
- The ensemble ledger SHA-256 is
  `679d1fd2d09303f96f285e0bfaa75c8cff0925d96a1bfffbe237df3384aba7dc`.
- The uninterrupted invocation took 8,913 seconds (about 2.48 hours) on the recorded RTX PRO 3000
  Blackwell laptop GPU environment.

The result concerns automatic coding of public AnnoMI demonstrations under this exact protocol. It
does not establish clinical validity, treatment efficacy, therapist quality, or state of the art
against papers using different split units.
