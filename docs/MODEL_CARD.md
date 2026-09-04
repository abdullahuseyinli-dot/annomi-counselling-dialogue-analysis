# Model card: AnnoMI research model suite

## Summary

This repository evaluates text models for automatic research coding of AnnoMI therapist utterances
and, in a separate study, prediction of empirical therapist/client annotation distributions. The
main benchmark uses five-fold nested cross-validation grouped by normalized source video URL. The
multi-annotator benchmark uses exhaustive leave-one-transcript-out evaluation over seven dialogues.

No checkpoint is designated for clinical deployment. The recommended *research baseline* is the
target-only RoBERTa model because its improvement over TF-IDF is supported and it has the best Brier
and log-loss values among the neural classifiers. Causal-history RoBERTa has the highest observed
macro-F1, but the difference from target-only is small, uncertain, and accompanied by worse log
loss.

## Systems

| System | Input | Training target | Role |
|---|---|---|---|
| TF-IDF elastic net | Current therapist utterance | Four-class hard code | Sparse baseline |
| RoBERTa target | Current therapist utterance | Four-class hard code | Supported neural baseline |
| RoBERTa causal | Target plus up to ten preceding turns | Four-class hard code | Numerical F1 leader |
| DASH-MI | Separate target/history encoders with gated residual context | Hard or vote-mixed code | Registered context candidate |
| Soft linear | Frozen target-only RoBERTa embedding | Full ten-vote distribution | Distribution baseline |
| PANEL-MI | Frozen embedding plus low-rank anonymous-annotator heads | Individual votes and aggregate distribution | Registered disagreement candidate |
| SAFE-MI frozen/adapted GRU | Causal utterance embeddings and roles | Next therapist action | Updated Task C baselines |
| SAFE-MI prototype | Fold-local retrieved action prototypes with source/text exclusions | Next therapist action | Exploratory retrieval candidate |
| SAFE-MI one-way | Detached Task C representation feeding the quality head | Quality plus next action | Post-hoc Task A candidate |

The neural encoder is `FacebookAI/roberta-base` at revision
`e2da8e2f811d1448a5b465c236feacd80ffbac7b`. Model-specific configurations, seeds, optimization,
selection, truncation, and ablations are machine-readable under `configs/research/`.

## Evaluation

### Therapist code classification

The evaluation covers 4,882 therapist utterances from 119 source video groups. Aggregate neural
probabilities average five fixed seeds after every item has received one out-of-source prediction.

| Model | Source-balanced macro-F1 ↑ | Brier ↓ | Log loss ↓ | ECE ↓ |
|---|---:|---:|---:|---:|
| TF-IDF | 0.7172 | 0.3906 | 0.7670 | **0.0237** |
| RoBERTa target | 0.8108 | **0.2796** | **0.5587** | 0.0658 |
| RoBERTa causal | **0.8163** | 0.2838 | 0.5880 | 0.0800 |
| DASH-MI | 0.8116 | 0.2828 | 0.5883 | 0.0810 |

Target-only RoBERTa improves macro-F1 over TF-IDF by 0.0935, 95% paired source-bootstrap interval
[0.0778, 0.1097]. Causal context adds 0.0056 over target-only, interval [-0.0040, 0.0154], and
worsens log loss. DASH-MI is not better than causal RoBERTa and its registered context ablation is
null. Numerical rankings must not be restated as supported pairwise differences.

The weakest class remains `therapist_input`: causal RoBERTa F1 is 0.6683, compared with 0.8898 for
`other`, 0.8516 for `question`, and 0.7826 for `reflection`. This class gap should remain visible in
any downstream report.

### Annotation-distribution prediction

The subset contains 216 therapist and 212 client utterances, ten existing votes per utterance, and
seven transcript clusters. The primary distribution metric is transcript-balanced vote log score.

- Soft-linear strongly improves over the transcript prior on both tasks: therapist delta -0.6309
  (exact p = 0.0078) and client delta -0.1113 (p = 0.0156).
- PANEL-MI fails its registered therapist log-score gate: +0.0435 versus soft-linear, interval
  [-0.0353, 0.1074]. It does achieve the best therapist JSD (0.1169) and entropy MAE (0.2910), an
  exploratory Pareto trade-off rather than a primary win.
- With seven clusters, intervals and transcript-specific effects are more informative than small
  numerical rank changes.

### Early quality and next-action prediction

Task A uses 115 eligible transcripts from 108 sources at the primary ten-therapist-turn checkpoint.
The post-hoc one-way model reaches 0.7389 source-balanced balanced accuracy, versus 0.7000 for the
earlier Q-TRACE-MI candidate. Its paired interval against A-only crosses zero and its Brier score
degrades beyond the frozen limit, so it is a numerical result rather than a supported replacement.

Task C uses 4,743 strict client-to-therapist handoffs from 119 sources. Frozen-GRU reaches 0.4359
source-balanced macro-F1. Safe prototype retrieval reaches 0.4328 and improves Brier from 0.6851 to
0.6798, but its F1 interval versus frozen-GRU crosses zero. Adapted-GRU reaches 0.4339 and passes a
post-hoc descriptive non-inferiority/calibrated-set gate; it does not establish superiority.

The uploader-designated Task A label is not a MITI rating. Task C predicts an observed action, not
an appropriate or optimal intervention. The official MI-TAGS public samples are insufficient for
external evaluation after the locked possible-overlap quarantine; no transportability claim is
made.

## Intended use

Appropriate uses are reproducible NLP benchmarking, method comparison, annotation-disagreement
research, and aggregate exploratory analysis of public AnnoMI demonstrations. The outputs may help
researchers understand which inputs or objectives predict the corpus's coding scheme.

The systems are not intended for:

- clinical diagnosis, risk scoring, treatment selection, or patient triage;
- live feedback to clients or direct control of a conversational agent;
- therapist ranking, employment evaluation, reimbursement, or access-to-care decisions;
- claims about treatment effectiveness, causal therapist effects, or real-world clinical quality;
- replacement of trained human coders.

## Limitations and risks

- AnnoMI consists of public demonstration videos, not a representative sample of clinical care.
- Source-disjoint validation reduces direct production/speaker leakage but does not establish
  transportability to private sessions, other languages, new coding manuals, or demographic groups.
- The labels compress nuanced dialogue behavior. Disagreement can reflect ambiguity, coding
  conventions, or annotator tendencies rather than removable noise.
- The multi-annotator result describes one anonymous ten-person panel across only seven dialogues.
- Context can expose sensitive conversational details and modestly harms probabilistic quality in
  the present study.
- The SAFE-MI architecture search is explicitly exploratory because earlier AnnoMI fold outcomes
  informed its design; the v2.1 audit is explicitly post-hoc.
- The MI-TAGS sample audit indicates possible source reuse with AnnoMI. External confirmation
  requires authorized access to the full corpus and the already locked quarantine/split rules.
- No demographic attributes support a fairness audit. Absence of measured disparity is not evidence
  of fairness.
- Model confidence is not clinical certainty. Calibration is evaluated only within this corpus and
  split design.

## Data, privacy, and artifacts

Raw utterance text, source metadata, model weights, checkpoints, and embedding caches are ignored by
Git. The repository does retain privacy-reduced out-of-fold ledgers containing transcript/utterance
IDs, hashed source IDs, labels or vote distributions, predictions, probabilities, folds, and seeds;
these are necessary to reconstruct every result. Anonymous annotator IDs are not present in released
prediction ledgers.

Users must review the upstream dataset terms, model licenses, and local governance requirements.
The code does not make AnnoMI or RoBERTa part of this repository's MIT license.

## Reproducibility and provenance

The protocol, data digests, exact encoder revision, code commit, seed list, selected recipes,
runtime environment, prediction hashes, failed attempts, and paired inference are retained. Evidence
files are create-only. Run `python -m annomi_research validate`, `pytest -q`, and
`python tools/validate_repository.py` to verify the current package.

The older 31-transcript portfolio holdout remains preserved as development-consumed evidence. It is
not the source-disjoint result described in this card.
