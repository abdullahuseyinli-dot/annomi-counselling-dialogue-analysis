# Source-disjoint sparse baseline v1

Status: completed prospective baseline under `dash-mi-source-cv-v1`.

The run used all five fixed outer folds, grouped by normalized source-video URL. Model and
regularization selection occurred only inside each outer-training partition with three
source-grouped inner folds. The prediction ledger contains one out-of-fold prediction for every
therapist utterance and is the authority for all reported metrics.

| Model | Source-balanced macro-F1 | Utterance macro-F1 | Source-balanced Brier | ECE (10) |
|---|---:|---:|---:|---:|
| class prior | 0.1038 | 0.1226 | 0.7548 | 0.0654 |
| TF-IDF, target utterance | **0.7172** | **0.6895** | **0.3906** | **0.0237** |
| TF-IDF, flattened causal 10-turn context | 0.4682 | 0.4482 | 0.6629 | 0.0721 |

The target-utterance sparse model is the baseline to beat. Its per-class F1 values are 0.8585
(`other`), 0.7452 (`question`), 0.6453 (`reflection`), and 0.5089 (`therapist_input`). Its worst
20% source-level mean log-loss CVaR is 1.2391.

The flattened-context result is a retained negative result. It does not show that context is
intrinsically unhelpful: concatenating all turns into a single bag of n-grams obscures which words
belong to the target utterance and which belong to prior speakers. Subsequent contextual models
must preserve that boundary. This interpretation was made only after the registered sparse run;
new contextual candidates will be identified as post-baseline development.

Repeated normalized target text is not automatically an easier subset: the model obtains macro-F1
0.4035 on 1,116 rows whose normalized text occurs in the outer-training data and 0.6098 on 3,766
unseen-text rows. This is descriptive, not a causal comparison; repeated short counselling phrases
can be generic and label-ambiguous.

## Evidence

- `results/research/baseline_v1/predictions.csv`: immutable row-level out-of-fold ledger.
- `results/research/baseline_v1/summary.json`: reconstructed metrics and provenance.
- `results/research/baseline_v1/selection.json`: inner-fold recipe scores and selections.
- Code commit recorded by the run: `9fdc69dc4a98e16f4a19dd3426f304025b6a4299`.
- Prediction-ledger SHA-256:
  `3a3bd0b61d0932fc6e05257cee89bce8420a21bdb3127dda78d589448a4f069c`.

These numbers are estimates for public AnnoMI demonstrations under the locked source-disjoint
protocol. They are not clinical-validity, treatment-effect, or cross-benchmark state-of-the-art
claims.
