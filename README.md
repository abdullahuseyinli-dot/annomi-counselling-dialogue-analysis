# AnnoMI Counselling Dialogue Analysis

[![Python 3.11-3.12](https://img.shields.io/badge/Python-3.11--3.12-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository evaluates therapist-behaviour classification on
[AnnoMI](https://github.com/uccollab/AnnoMI) using cross-validation grouped by source video. It also
includes early-session forecasting and a separate analysis of multi-annotator label distributions.
Result files and evaluation code are included for reproducibility.

> **Status:** the tables and figures below can be rebuilt from the tracked result files. Full
> MI-TAGS evaluation requires access to the complete corpus. This development version has not been
> archived and has no DOI. See the
> [project status](docs/PROJECT_STATUS.md), [documentation index](docs/README.md), and
> [manuscript evidence map](paper/CLAIM_EVIDENCE_CROSSWALK.md).

Here, *registered* means that the analysis plan was committed before the corresponding local run;
it does not imply public preregistration.
Raw counselling text and model weights are not distributed.

![Overview of leakage-controlled classification and vote-distribution results](assets/research/research_overview.png)

## Main source-disjoint result

The primary task predicts four therapist-behaviour codes on 4,882 utterances from 119 normalized
video sources. Each test prediction comes from a model trained without data from the same video
source. Model and recipe selection are nested within five outer folds, and neural probabilities are
averaged across five fixed seeds.

| Model | Source-balanced macro-F1 ↑ | Brier ↓ | Log loss ↓ | ECE ↓ |
|---|---:|---:|---:|---:|
| TF-IDF elastic net | 0.7172 | 0.3906 | 0.7670 | **0.0237** |
| RoBERTa, target only | 0.8108 | **0.2796** | **0.5587** | 0.0658 |
| RoBERTa, causal history | **0.8163** | 0.2838 | 0.5880 | 0.0800 |
| DASH-MI | 0.8116 | 0.2828 | 0.5883 | 0.0810 |

Paired comparisons show:

- Target-only RoBERTa improves source-balanced macro-F1 over TF-IDF by **0.0935**, with a paired
  source-bootstrap 95% interval of **[0.0778, 0.1097]**. This met the prespecified success criterion.
- Causal-history RoBERTa is the **numerical F1 leader at 0.8163**, but its +0.0056 gain over
  target-only has interval **[-0.0040, 0.0154]** and its log loss is worse. It is not a supported
  replacement for the better-calibrated target-only model.
- DASH-MI reaches 0.8116. Its context residual changes only 0.53% of decisions, and neither its
  +0.0011 context-ablation delta nor its -0.0047 delta versus causal RoBERTa is supported.

Causal-history RoBERTa has the highest macro-F1 at **0.8163**. Target-only RoBERTa is the
better-supported choice: its gain over TF-IDF is statistically supported, and it has lower Brier
score and log loss than the other neural models. See the
[neural result](docs/research/ROBERTA_FLAT_CAUSAL_V1_RESULT.md),
[DASH-MI result](docs/research/DASH_MI_V1_RESULT.md), and
[paired inference](docs/research/DASH_MI_INFERENCE_V1.md).

## Early quality and next-action forecasting

The Task A/C track adds causal early-session quality classification and strict client-to-therapist
next-action forecasting. We evaluated frozen and adapted encoders, GRU and attention context
models, one-way multitask transfer, transition priors, and prototype retrieval.

| Task | Earlier registered result | Updated exploratory point result | Interpretation |
|---|---:|---:|---|
| A: quality after 10 therapist turns | Q-TRACE-MI: 0.7000 balanced accuracy | **One-way multitask: 0.7389** | +0.0389 cross-run difference; paired intervals vs A-only cross zero for accuracy and Brier |
| C: next therapist action | C-only: 0.4251 macro-F1 | **Frozen-GRU: 0.4359** | +0.0108 descriptive cross-run difference, not a paired comparison; no SAFE-MI candidate beats its matched baseline |

The one-way Task A model is 0.0889 higher than A-only overall and is higher in four of five seeds.
Its accuracy interval is [-0.0521, 0.2256], and its Brier point estimate is 0.0191 higher with an
interval of [-0.0180, 0.0557]. For Task C, prototype retrieval reaches 0.4328 macro-F1 versus 0.4359
for frozen-GRU and has a 0.0053 lower Brier point estimate; both paired intervals include zero.
Adapted-GRU prediction sets have 0.8072
cross-fitted coverage and a mean size of 2.4845 in a post-hoc analysis. These results do not establish
superiority.

![SAFE-MI Task A and Task C results](assets/research/safe_mi_results.png)

These tasks use no new manual labels. The quality target comes from source metadata and is not an
independent measure of clinical fidelity. A comparison of the public MI-TAGS samples with AnnoMI
flagged 9 of 12 records as possible overlaps, leaving too little independent data for external
evaluation. See the
[earlier Q-TRACE-MI result](docs/research/QTRACE_MI_V1_RESULT.md) and
[complete SAFE-MI result](docs/research/SAFE_MI_V2_RESULT.md).

## Multi-annotator result

The study uses AnnoMI's existing 428 utterances with ten annotations each
across seven transcripts. The registered leave-one-transcript-out study predicts the complete vote
distribution, models therapist and client codes separately, and treats seven transcripts—not 4,280
annotation rows—as the inferential units.

| Task | Model | Vote log score ↓ | Brier ↓ | JSD ↓ | Plurality macro-F1 ↑ |
|---|---|---:|---:|---:|---:|
| Therapist | Transcript prior | 1.4441 | 0.5953 | 0.3099 | 0.0818 |
| Therapist | Hard linear | **0.8099** | 0.2351 | 0.1381 | 0.7401 |
| Therapist | Soft linear | 0.8132 | **0.2245** | 0.1260 | **0.7430** |
| Therapist | PANEL-MI | 0.8567 | 0.2272 | **0.1169** | 0.7326 |
| Client | Transcript prior | 1.0566 | 0.3781 | 0.1822 | 0.2417 |
| Client | Hard linear | 0.9798 | 0.3235 | 0.1582 | 0.3399 |
| Client | Soft linear | **0.9453** | **0.3011** | **0.1453** | 0.4405 |
| Client | PANEL-MI | 1.0253 | 0.3380 | 0.1518 | **0.4724** |

Soft label-distribution learning beats the transcript prior on therapist log score by **-0.6309**
(95% interval **[-0.7257, -0.5055]**, exact sign-flip p = **0.0078**, 7/7 transcripts) and on
client log score by **-0.1113** (interval **[-0.1773, -0.0402]**, p = **0.0156**, 6/7).

PANEL-MI did not improve the primary log-score metric, although it had the lowest therapist JSD and
entropy-error point estimates. Full results and selection details are in the
[PANEL-MI report](docs/research/PANEL_MI_V1_RESULT.md).

![Registered paired effects with cluster-level intervals](assets/research/registered_effect_intervals.png)

## Evaluation controls

- Normalized source video URL, not utterance or transcript alone, is the dependency group for the
  main benchmark.
- Every outer fold is retained; no favorable fold is selected.
- Tokenization, PCA, scaling, class weights, early stopping, and recipe choice are training-only.
- Inputs are causal: the target utterance and, where registered, preceding turns only.
- Metrics can be recomputed from out-of-fold probability files whose hashes are recorded.
- Bootstrap sampling respects source or transcript clusters. Seeds and annotation rows are never
  treated as independent samples.
- Run records include failed checks, convergence fallbacks, and ablation results.

The dataset creators describe AnnoMI as 133 professionally transcribed, expert-annotated
demonstration dialogues and explicitly distinguish it from real therapy sessions
([Wu et al., 2023](https://doi.org/10.3390/fi15030110)). Accordingly, this repository makes no
clinical-validity, therapist-ranking, causal-effect, or state-of-the-art claim.

## Reproduce

The default check uses CPU-only neural dependencies and does not retrain models or download raw
dialogue data:

```bash
uv sync --locked --extra dev --extra neural-cpu
uv run python -m pytest --cov=annomi_research --cov-report=term-missing
uv run python tools/validate_repository.py
```

Data-backed validation, classical reproduction, CUDA/BF16 neural reproduction, and the optional
MI-TAGS audit have different access and hardware requirements. See
[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the commands and requirements. Existing result
files are not overwritten; changed runs use a new versioned directory. Dataset files are pinned to
a commit and checked by SHA-256 before training.

## Repository map

```text
configs/research/                 Locked machine-readable protocols and comparisons
src/annomi_research/              Audits, splits, models, inference, and validators
results/research/                 Reconstructable out-of-fold evidence and summaries
results/research/publication_v1/  Derived exact tables plus a hash manifest
results/research/publication_ac_v1/  Derived Task A/C tables plus a hash manifest
results/research/publication_safe_mi_v2/  SAFE-MI tables, intervals, and audit manifest
docs/research/                    Registrations, execution logs, and result reports
docs/                             Benchmark, data, model, artifact, and release documentation
paper/                            Manuscript outline, bibliography, and claim-evidence crosswalk
assets/research/                  Script-generated publication figures
tests/                            Fast contracts for data, models, metrics, and evidence
tools/                            Pinned download, validation, and asset builders
.github/workflows/                Cross-platform CI, data audit, and history secret scan
```

Earlier exploratory results are available in `results/main/`, `results/extensions/`, and
`results/summarisation/`. They use a different holdout design and are not directly comparable with
the source-grouped results above.

## License and attribution

Original code and documentation are MIT licensed. AnnoMI and pretrained encoders are separate
third-party works and are not redistributed here. Review
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before reuse.
