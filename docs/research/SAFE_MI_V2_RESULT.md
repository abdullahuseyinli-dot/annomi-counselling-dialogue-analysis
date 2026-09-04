# SAFE-MI staged Task A/C result

## Verdict

The executable AnnoMI stages are complete. SAFE-MI v2 screened ten neural configurations plus
source-safe prototype retrieval, advanced two candidates under a frozen stopping rule, and then
ran five-seed source-disjoint evaluation. The registered exploratory candidates did not beat the
matched Task C baseline. A separately registered post-hoc audit found a stronger Task A point
estimate and a useful calibrated-set result, but neither is confirmatory.

The strongest updated point estimates are:

- **Task A:** one-way detached multitask learning reaches **0.7389** source-balanced balanced
  accuracy at ten therapist turns. This is +0.0389 over the earlier Q-TRACE-MI point estimate, but
  its paired interval against A-only crosses zero and its Brier score is worse.
- **Task C:** the new frozen-GRU baseline reaches **0.4359** source-balanced macro-F1. The best
  SAFE-MI candidate is safe prototype retrieval at 0.4328; it improves Brier score but not F1.
- **Calibrated Task C sets:** outer-cross-fitted sets for adapted-GRU reach **0.8072** coverage with
  mean size **2.4845**, passing the descriptive 0.80 coverage and 2.50 size thresholds.

All values reconstruct from retained out-of-source probability ledgers. No manual labels were
added, and no failed gate was discarded.

## Execution

The run used the same five exhaustive source folds, with disjoint fit, validation, calibration,
and test sources inside every outer fold. It used the fixed seeds 17, 42, 101, 314, and 2718.

| Item | Recorded value |
|---|---:|
| SAFE-MI v2 neural fits | 110 |
| Post-hoc extension fits | 20 |
| Total new neural fits | 130 |
| Task A t10 coverage | 115 transcripts / 108 sources / 18 low labels |
| Task C coverage | 4,743 decisions / 119 sources |
| Paired source-bootstrap draws | 5,000 per reported candidate |
| Main plus extension runtime | 5,785.3 seconds |
| Hardware | NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU |
| CUDA smoke gate | Pass |

The staged screen compared frozen and fold-adapted RoBERTa embeddings, GRU and client-attention
context encoders, logit adjustment, shared and one-way multitask transfer, discounted quality
evidence, transition supervision, a bounded zero-start transition residual, and prototype
retrieval. Fold-local LoRA adapters used fit-source behaviour supervision only. Retrieval excluded
the same source and normalized-text matches. Gold history, target therapist text, future turns,
true quality at Task C inference, and source metadata were unavailable to the models.

## Task A: early demonstration-quality classification

| Model | t10 balanced accuracy | Low-class AUPRC | Brier |
|---|---:|---:|---:|
| A-only neural | 0.6500 | 0.4921 | 0.1335 |
| Joint, no transition | 0.6611 | 0.5362 | **0.1304** |
| Earlier Q-TRACE-MI | 0.7000 | **0.6276** | 0.1453 |
| **One-way detached multitask** | **0.7389** | 0.5990 | 0.1526 |

The one-way model improves balanced accuracy over its registered A-only reference by +0.0889 and
does so in four of five seeds. Its paired-source 95% interval is [-0.0521, 0.2256], however, and
its Brier delta is +0.0191 with interval [-0.0180, 0.0557]. It therefore passes the point-gain and
seed-direction components but fails the calibration constraint. Relative to Q-TRACE-MI, +0.0389
is a descriptive cross-run point difference, not a new confirmatory contrast.

The result suggests that blocking Task A gradients from changing the Task C path can reduce the
negative-transfer problem seen in Q-TRACE-MI. It does not establish that demonstration-quality
metadata measures MI fidelity, clinical quality, or patient outcome.

## Task C: strict next therapist-action forecasting

| Model | Source-balanced macro-F1 | Brier | Log loss | Status |
|---|---:|---:|---:|---|
| Earlier C-only neural | 0.4251 | 0.6904 | 1.2679 | Earlier registered baseline |
| **Frozen-GRU** | **0.4359** | 0.6851 | 1.2558 | Matched v2 baseline |
| Adapted-GRU | 0.4339 | 0.6843 | 1.2523 | Post-hoc descriptive gate pass |
| Safe prototype retrieval | 0.4328 | **0.6798** | **1.2459** | Registered candidate gate fail |
| One-way detached multitask | 0.4229 | 0.6855 | 1.2538 | Post-hoc descriptive gate fail |
| Discounted one-way | 0.4218 | 0.6843 | 1.2536 | Registered candidate gate fail |

Safe prototype retrieval changes F1 by -0.0032 versus frozen-GRU, with paired-source interval
[-0.0285, 0.0241], while improving Brier by -0.0053. Four of five seed-level F1 contrasts are
positive, but the ensemble contrast is not. Discounted one-way transfer changes F1 by -0.0141,
interval [-0.0423, 0.0151]. The post-hoc adapted-GRU result is non-inferior under its descriptive
margin and has slightly better Brier, but it is not superior: F1 delta -0.0021, interval
[-0.0261, 0.0243].

The numerical increase from the earlier C-only run (0.4251) to the frozen-GRU baseline (0.4359) is
+0.0108. Because that comparison was not a registered paired candidate contrast, it should be
reported as a pipeline update rather than a supported population-level improvement.

## Prediction-set audit

The registered split-source sets met coverage but remained too wide. The later outer-cross-fitted
sensitivity analysis used only out-of-source calibration predictions and was not used to select a
new model.

| Model | Calibration | Coverage | Mean set size | Threshold result |
|---|---|---:|---:|---|
| Frozen-GRU | Outer-crossfit | 0.8068 | 2.4729 | Pass |
| Adapted-GRU | Outer-crossfit | 0.8072 | 2.4845 | Pass |
| One-way multitask | Outer-crossfit | 0.8065 | 2.5023 | Size fail by 0.0023 |

These are four-class action sets, not treatment recommendations. Coverage is measured only on the
AnnoMI demonstrations under this split design.

## External confirmation boundary

The MI-TAGS protocol was committed before retrieving the official public samples. It freezes a
MITI-to-AnnoMI mapping, overlap thresholds, deterministic source partition, ordinal Task A
endpoints, and a no-retuning confirmation rule. The official
[MI-TAGS paper](https://aclanthology.org/2024.lrec-main.1017/) reports a 242-session corpus, while
the [public repository](https://github.com/Advanced-Reality-Lab/MI-TAGS) exposes only small sample
files without completing the full-data access process.

The reproducible sample audit found 9 possible AnnoMI overlaps among 12 records and quarantined
them. Only two records landed in the locked test partition before quarantine, below the required 20
test groups. The samples therefore support schema and leakage checks only; no external performance
claim is permitted. Full confirmation remains blocked until the researcher obtains authorized
access to the official full files.

## Research contribution and interpretation

The publishable unit is not an unsupported state-of-the-art claim. It is a falsifiable, auditable
study of asymmetric transfer and uncertainty under strict source separation:

- a staged negative-transfer diagnosis across shared, detached, and time-discounted objectives;
- fold-local parameter-efficient adaptation without test-source or calibration-source leakage;
- retrieval augmentation with explicit same-source and repeated-text exclusion;
- zero-initialized bounded transition residuals and matched ablations;
- source-cluster inference, seed stability, proper scores, and cross-fitted prediction sets;
- a pre-access external-overlap protocol that detected likely benchmark reuse before evaluation.

The substantive finding is that Task A discrimination and Task C probability quality can improve
on different Pareto axes, while more coupling does not reliably improve Task C F1. That negative
result, plus the detected external overlap, is more defensible than selecting the best fold or
metric after the fact. The repository-specific combination is a candidate methodological
contribution; this study does not claim that each component, or the combination, is the first of
its kind.

## Reproduce and audit

```bash
python -m annomi_research smoke-safe-mi
python -m annomi_research run-safe-mi
python -m annomi_research run-safe-mi-extension
python -m annomi_research audit-mi-tags
python tools/build_safe_mi_assets.py
python -m annomi_research validate
pytest -q
python tools/validate_repository.py
```

The governing files are
[`protocol_safe_mi_v2.json`](../../configs/research/protocol_safe_mi_v2.json),
[`protocol_safe_mi_v2_1.json`](../../configs/research/protocol_safe_mi_v2_1.json), and
[`protocol_mi_tags_external_v1.json`](../../configs/research/protocol_mi_tags_external_v1.json).
Exact tables and their hash manifest are in
[`publication_safe_mi_v2`](../../results/research/publication_safe_mi_v2/). The immutable aggregate
summaries are
[`safe_mi_v2/summary.json`](../../results/research/safe_mi_v2/summary.json),
[`safe_mi_v2_1/summary.json`](../../results/research/safe_mi_v2_1/summary.json), and the
[`MI-TAGS sample audit`](../../results/research/mi_tags_external_v1/sample_overlap_audit.json).
