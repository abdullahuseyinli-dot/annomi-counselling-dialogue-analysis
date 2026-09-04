# Literature positioning matrix

This is a focused positioning aid for the repository's completed studies, not a systematic review.
The bibliography keys refer to [the paper bibliography](../paper/references.bib). Statements about
this repository are bounded by the [claim–evidence crosswalk](../paper/CLAIM_EVIDENCE_CROSSWALK.md).

| Area and reference | What the prior work contributes | Use in this repository | Limits |
|---|---|---|---|
| AnnoMI dataset: [Wu et al. (2022)](https://doi.org/10.1109/ICASSP43922.2022.9746035), `wu2022annomi` | Introduces the public expert-annotated counselling-dialogue dataset. | Establishes the upstream corpus and label semantics. | Dataset creation and annotation are upstream contributions, not contributions of this repository. |
| Extended AnnoMI analysis: [Wu et al. (2023)](https://www.mdpi.com/1999-5903/15/3/110), `wu2023creation` | Documents collection, extended annotations, corpus analysis, and baseline prediction tasks. | Provides the primary dataset description and the basis for auditing the simple and full releases. | This study evaluates a pinned copy under a different source-grouped contract; it does not redefine the upstream data. |
| Direct AnnoMI action forecasting: [Wu et al. (2022)](https://www.isca-archive.org/interspeech_2022/wu22c_interspeech.html), `wu2022forecasting` | Studies therapist-action forecasting with context, augmentation, observed labels, and dialogue quality. | Motivates strict Task C forecasting and the need to identify which information is available before the target turn. | Scores are not treated as directly comparable because feature and split contracts differ. |
| Pretrained text encoder: [Liu et al. (2019)](https://arxiv.org/abs/1907.11692), `liu2019roberta` | Introduces the RoBERTa pretraining recipe. | Supplies the pinned encoder used for target-only fine-tuning and frozen representations. | The encoder is established prior work; the contribution is its controlled evaluation here. |
| Label-distribution learning: [Geng (2016)](https://doi.org/10.1109/TKDE.2016.2545658), `geng2016labeldistribution` | Formalizes learning a distribution over labels rather than a single hard label. | Motivates training directly on the ten-vote empirical distribution. | Predicting this panel's distribution is not recovery of clinical truth or a population consensus. |
| Crowd layer: [Rodrigues and Pereira (2018)](https://ojs.aaai.org/index.php/AAAI/article/view/11506), `rodrigues2018crowds` | Learns annotator-specific transformations jointly with a shared predictor. | Provides a precedent for explicit annotator-conditioned heads. | Annotator conditioning is not claimed as new. |
| Annotator representations: [Deng et al. (2023)](https://aclanthology.org/2023.findings-emnlp.832/), `deng2023annotator` | Models annotator and annotation representations to retain disagreement. | Motivates retaining anonymous annotator tendencies rather than collapsing every item to plurality. | The seven-transcript experiment cannot characterize broader annotator populations. |
| Joint item/annotator distributions: [Weerasooriya et al. (2023)](https://aclanthology.org/2023.findings-acl.287/), `weerasooriya2023disco` | Aggregates predictions over annotator–item pairs to preserve label diversity. | Closest conceptual precedent for the population prediction formed from the ten annotator heads. | PANEL-MI is a task-specific low-rank implementation, not the first item–annotator distribution model. |
| Parameter-efficient adaptation: [Hu et al. (2022)](https://openreview.net/forum?id=nZeVKeeFYf9), `hu2022lora` | Adapts frozen pretrained weights through trainable low-rank updates. | Motivates fold-local adaptation in the exploratory SAFE-MI screen. | Because earlier AnnoMI results informed model development, this analysis is exploratory. |
| Prototype representations: [Snell et al. (2017)](https://papers.nips.cc/paper_files/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html), `snell2017prototypical` | Classifies through distances to learned class prototypes in a metric space. | Provides a conceptual precedent for prototype-based evidence; the repository adds source and normalized-text exclusions to its retrieval variant. | The implementation is not presented as a new few-shot-learning algorithm. |
| Dialogue-transition regularization: [Rudolph et al. (2026)](https://aclanthology.org/2026.findings-acl.1271/), `rudolph2026transition` | Regularizes next-dialogue-act predictions toward corpus transition statistics. | Motivates the Task C transition term and a matched test of whether such structure transfers to AnnoMI's four-class target. | The local negative result does not contradict gains reported for a different taxonomy and corpus. |
| Adaptive prediction sets: [Huang et al. (2024)](https://proceedings.mlr.press/v235/huang24aa.html), `huang2024conformal` | Develops label-ranking scores for compact classification prediction sets. | Motivates reporting set coverage and efficiency rather than only top-one action accuracy. | Recorded coverage is empirical under source-held-out AnnoMI evaluation, not a clinical guarantee. |
| Conformal risk control: [Angelopoulos et al. (2024)](https://research.google/pubs/conformal-risk-control/), `angelopoulos2024conformalrisk` | Extends conformal calibration to control expected monotone losses. | Motivates treating a source's mean miscoverage as the calibration loss in the action-set audit. | Cluster count, exchangeability, and post-hoc status must remain visible; no universal guarantee is asserted. |
| MI-TAGS: [Cohen et al. (2024)](https://aclanthology.org/2024.lrec-main.1017/), `cohen2024mitags` | Describes 242 demonstrations with therapist behaviour/global scores and client-language annotations. | Supplies the candidate external corpus and motivates mapping AnnoMI actions to MITI codes. | The public sample supports schema and overlap auditing only; 9 of 12 records were quarantined and no external performance was computed. |

## Interpretation

The current evidence supports a paper about the interaction of source grouping, coding
disagreement, task coupling, and uncertainty on one public demonstration corpus. Its most reliable
positive findings are the source-held-out target-only encoder gain and vote-distribution prediction
over a transcript prior. The context, annotator-conditioned, transition-coupled, and asymmetric
transfer experiments contribute informative mixed or negative findings under matched protocols.
The overlap audit shows why a nominally external corpus cannot be assumed independent.

The literature reviewed here does not establish that this is the first study to combine every
component. Before submission, the authors should repeat the search for work published after the
dates above and narrow any novelty sentence to a directly evidenced, repository-specific design
choice. Claims of state-of-the-art performance, clinical safety, publication acceptance, or
confirmed external generalization are outside the available evidence.
