# Q-TRACE-MI Task A/C result v1

## Verdict

The registered run is complete. Q-TRACE-MI is the best observed non-oracle Task A model, but it
does **not** pass the joint success gate because its Task A interval includes zero and it performs
worse than the matched C-only model on Task C. This negative joint result is retained without
post-test tuning.

The strongest task-specific systems are:

- **Task A:** Q-TRACE-MI, with 0.7000 source-balanced balanced accuracy and 0.6276 low-class AUPRC
  after 10 observed therapist turns.
- **Task C:** C-only neural, with 0.4251 source-balanced macro-F1 and 0.6904 Brier score across
  4,743 strict client-to-therapist handoffs.

No new manual labels were used. Task A predicts an uploader-supplied demonstration-quality label;
it is not a measure of clinical fidelity, treatment effect, or patient outcome.

## Execution

The run used the locked five source-disjoint outer folds, a disjoint fit/validation/calibration
partition inside every outer-training fold, and five fixed seeds. Four modes produced 100 neural
fits in total. Frozen RoBERTa embeddings, training caches, and weights remain ignored artifacts;
out-of-source probability ledgers and all failed gates are retained.

| Item | Recorded value |
|---|---:|
| Transcript/source coverage | 133 transcripts / 119 sources |
| Task A t10 coverage | 115 transcripts / 108 sources / 18 low labels |
| Task C coverage | 4,743 decisions / 119 sources |
| Neural fits | 4 modes x 5 folds x 5 seeds = 100 |
| Paired source-bootstrap draws | 5,000 |
| Full neural-run time | 1,661.1 seconds |
| Hardware | NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU |
| CUDA smoke gate | Pass |

## Task A: early demonstration-quality classification

| Model | t10 balanced accuracy | Low-class AUPRC | Brier |
|---|---:|---:|---:|
| Class prior | 0.5000 | 0.1259 | 0.1444 |
| TF-IDF raw prefix | 0.5556 | 0.4587 | 0.1664 |
| Structure only | 0.6259 | 0.2775 | 0.2520 |
| A-only neural | 0.6500 | 0.4921 | 0.1335 |
| Joint, no transition | 0.6611 | 0.5362 | **0.1304** |
| **Q-TRACE-MI** | **0.7000** | **0.6276** | 0.1453 |
| Gold-code oracle | 0.7435 | 0.6407 | 0.1474 |

Q-TRACE-MI improves the point estimate over A-only by +0.0500, with improvement in four of five
seeds. The paired-source 95% interval is [-0.0734, 0.1731], however, so the registered evidence
does not establish a positive population-level gain. Its t10 low-class AUPRC approaches the
gold-code oracle without using gold behaviour codes at inference.

The Q-TRACE-MI balanced-accuracy trajectory is 0.6018 (t3), 0.6470 (t5), 0.7000 (t10), 0.5265
(t20), and 0.7186 (full). The t20 estimate contains only five low-label transcripts and should not
be interpreted as a reliable learning curve.

## Task C: next therapist-action forecasting

| Model | Source-balanced macro-F1 | Brier | Log loss |
|---|---:|---:|---:|
| Class prior | 0.1442 | 0.7487 | 1.3833 |
| TF-IDF causal context | 0.3258 | 0.7405 | 1.3634 |
| Joint, no transition | 0.3710 | 0.7149 | 1.3139 |
| Q-TRACE-MI | 0.3742 | 0.7116 | 1.3084 |
| Gold-history Markov oracle | 0.4079 | 0.7079 | 1.3097 |
| **C-only neural** | **0.4251** | **0.6904** | **1.2679** |

Against C-only, Q-TRACE-MI changes macro-F1 by -0.0509 (paired-source 95% interval
[-0.0768, -0.0258]) and worsens Brier by +0.0212 ([0.0101, 0.0316]). Every one of the five seed
contrasts is negative. The evidence therefore supports the simpler task-specific model for this
forecasting target; latent quality conditioning and transition regularization do not help here.

At the registered 20% set-miscoverage target, Q-TRACE-MI obtains 0.8844 source-balanced coverage
with mean set size 3.1192 and zero singleton sets. The coverage target is met, but the wide sets
show that the four-way action forecast remains highly ambiguous.

## Registered gate

| Gate component | Observed | Pass |
|---|---:|:---:|
| Task A interval excludes zero positively | [-0.0734, 0.1731] | No |
| Task A positive seeds | 4 / 5 | Yes |
| Task C gain is at least +0.02 | -0.0509 | No |
| Task C interval excludes zero positively | [-0.0768, -0.0258] | No |
| Task C Brier degradation is at most +0.01 | +0.0212 | No |
| Task C positive seeds | 0 / 5 | No |
| No Task C class collapse | All four classes predicted | Yes |
| **Joint gate** |  | **No** |

## Relation to the earlier exports

The earlier Task A nested-CV values used complete-session aggregate features, while this study
tests absolute causal prefixes and source-disjoint folds. The earlier Task C hybrid result used a
single test split and features that included offline state; this study excludes the target text,
future length, true quality, and gold historical codes. The old and new scores therefore measure
different evaluation contracts and must not be presented as direct regressions or improvements.

The research contribution is the source-safe joint formulation, uncertainty-propagating history,
quality-conditioned transition mechanism, calibrated action sets, fixed multi-seed ablations, and
the falsifiable negative result. A stronger claim about MI fidelity still requires independent MITI
ratings or external validation.

## Reproduce and audit

```bash
python -m annomi_research run-ac-baselines
python -m annomi_research smoke-qtrace
python -m annomi_research run-qtrace
python -m annomi_research validate
python tools/build_ac_assets.py
python tools/validate_repository.py
```

The machine-readable protocol and model definition are in
[`protocol_ac_v1.json`](../../configs/research/protocol_ac_v1.json) and
[`qtrace_mi_v1.json`](../../configs/research/qtrace_mi_v1.json). Exact publication tables and their
hash manifest are in [`publication_ac_v1`](../../results/research/publication_ac_v1/).
