# Leakage-controlled behaviour and disagreement modeling in AnnoMI

## Abstract

Automatic coding of motivational-interviewing dialogue is commonly evaluated at utterance level,
although utterances from the same source share speakers, production conditions, and conversational
content. This study audits and replaces a development-consumed AnnoMI portfolio split with nested
cross-validation grouped by normalized source video URL. On 4,882 therapist utterances from 119
sources, a target-only RoBERTa classifier improves source-balanced macro-F1 from 0.7172 for an
elastic-net TF-IDF baseline to 0.8108 (paired source-bootstrap difference 0.0935, 95% interval
[0.0778, 0.1097]). Causal conversational context reaches the numerical maximum of 0.8163, but its
difference from target-only is uncertain and log loss is worse. A registered gated context model,
DASH-MI, does not improve performance.

A separate registered study uses 428 utterances from seven transcripts with ten existing
annotations each. Frozen text representations predict full coding-vote distributions under
leave-one-transcript-out evaluation. Soft label-distribution learning improves transcript-balanced
vote log score over a transcript-balanced prior for both therapist (difference -0.6309, interval
[-0.7257, -0.5055], exact p = 0.0078) and client turns (-0.1113, interval
[-0.1773, -0.0402], p = 0.0156). A low-rank annotator-conditioned model, PANEL-MI, improves
therapist Jensen-Shannon divergence and entropy error but fails its registered log-score gate. The
results support text-based research coding under this source-held-out protocol, while providing no
evidence of clinical validity or transportability to real care.

A third registered track estimates demonstration quality from causal prefixes and forecasts the
next therapist action before its text is observed. Q-TRACE-MI reaches 0.7000 balanced accuracy after
ten therapist turns, a +0.0500 point gain over A-only whose paired-source interval crosses zero. For
next-action forecasting, a C-only neural model reaches 0.4251 source-balanced macro-F1 and
Q-TRACE-MI is reliably worse by -0.0509. The joint candidate therefore fails its registered gate.

## 1. Motivation

[AnnoMI](https://doi.org/10.3390/fi15030110) provides 133 professionally transcribed,
expert-annotated motivational-interviewing demonstration dialogues. Its public availability makes
reproducible work possible in a domain where clinical data access is unusually constrained. It also
creates two methodological risks: demonstrations from the same source video can appear as multiple
transcripts, and hard-label aggregation can hide genuine coding ambiguity.

The study addresses six questions:

1. How well do sparse and transformer classifiers generalize to entirely unseen video sources?
2. Does causal conversational history add reliable value beyond the current therapist utterance?
3. Does a target-conditioned context architecture improve on simple context concatenation?
4. Can the full distribution of existing annotations be predicted, and do anonymous-annotator
   tendencies improve that prediction?
5. Can uploader-designated demonstration quality be estimated from an absolute causal prefix?
6. Can uncertainty-propagating behavioural histories and latent quality improve strict
   client-to-therapist next-action forecasting?

The fourth question follows label-distribution learning
([Geng, 2016](https://doi.org/10.1109/TKDE.2016.2545658)) and prior work that learns from individual
annotators, including the [crowd layer](https://doi.org/10.1609/aaai.v32i1.11506),
[annotator representations](https://aclanthology.org/2023.findings-emnlp.832/), and
[DisCo](https://aclanthology.org/2023.findings-acl.287/). PANEL-MI is therefore presented as a
task-specific architecture, not as the first annotator-conditioned model.

## 2. Data and validity audit

The pinned simple release contains 9,699 utterances across 133 transcripts. Normalized source URLs
identify 119 source groups. The audit found that grouping only by transcript permits source overlap,
and that prior cached portfolio outputs did not retain row-level probabilities sufficient to
reconstruct every inferential claim. Those outputs remain preserved under an annotated
development-consumed lineage.

The new primary population comprises 4,882 therapist utterances. Labels are `reflection`,
`question`, `therapist_input`, and `other`. Inputs exclude future turns, eventual transcript length,
video title, topic, MI-quality metadata, and gold historical codes. Raw counselling text is local
only.

The multi-annotator subset contains 216 therapist and 212 client utterances from transcript IDs 7,
27, 55, 56, 66, 109, and 130. The same ten anonymous annotator IDs label each item. Therapist and
client label spaces are distinct and are never merged. Four therapist and ten client items have
plurality ties, resolved only for the hard-label ablation by registered label order.

The Task A/C track contains 588 quality checkpoints across 133 transcripts and 4,743 strict
client-to-therapist handoffs. Its primary Task A checkpoint contains 115 transcripts from 108
sources after ten observed therapist turns. Task C excludes the target therapist text. Both tasks
exclude future turns, eventual length, titles, topics, gold historical codes, and true quality as a
forecasting input.

## 3. Methods

### 3.1 Source-disjoint classification

Five fixed outer folds are exhaustive and grouped by source URL. Hyperparameters are selected with
three source-grouped inner folds. Each neural finalist is refit for seeds 17, 42, 101, 314, and 2718;
probabilities, rather than metrics, are averaged. Training weights give each source equal total
influence and rebalance hard classes within the active training partition.

The compared systems are a class prior, utterance and causal-history TF-IDF elastic nets, target-only
[RoBERTa](https://arxiv.org/abs/1907.11692), flat causal-history RoBERTa, and DASH-MI. DASH-MI keeps a
target path isolated from history, attends from the target to separately encoded preceding turns,
and adds a zero-initialized gated residual. Its vote-aware recipes use empirical therapist votes only
where the ten-annotation subset is available. A matched inference-time context ablation estimates
the contribution of its context branch.

The primary endpoint is source-balanced macro-F1. Brier score, log loss, equal-frequency ECE,
per-class measures, worst-source-tail log loss, and seen/unseen repeated-text slices are retained.
Paired 10,000-resample cluster bootstraps operate on the 119 source units.

### 3.2 Multi-annotator label distributions

Each outer fold holds out one complete transcript; six inner leave-one-transcript-out folds select
recipes using vote log score. The pinned, frozen RoBERTa encoder mean-pools the role-prefixed target
utterance. PCA to at most 32 dimensions and standardization are refit inside every active training
partition.

Four systems are evaluated:

- a prior that averages one empirical distribution per training transcript;
- a multinomial linear model trained on deterministic plurality labels;
- a multinomial linear model trained on fractional vote mass;
- PANEL-MI, a shared linear distribution head plus centered annotator biases and a low-rank
  item-by-annotator interaction tensor.

PANEL-MI jointly minimizes individual-vote cross-entropy and aggregate-distribution cross-entropy.
Its population prediction is the mean of the ten observed annotator-head distributions. Centering
separates the shared logits from panel deviations, and shrinkage regularizes the low-rank factors.

The primary endpoint is transcript-balanced vote log score. Brier, Jensen-Shannon divergence,
plurality macro-F1, normalized-entropy association, and entropy error are secondary. Five-seed
PANEL-MI probabilities are averaged before a 10,000-resample transcript bootstrap. With seven
clusters, an exhaustive 128-assignment one-sided sign-flip test accompanies the interval.

### 3.3 Joint early-quality and next-action modeling

Q-TRACE-MI mean-pools frozen, commit-pinned RoBERTa embeddings for each utterance. Soft auxiliary
therapist/client behaviour distributions enter a causal GRU. Two action-policy heads represent
latent high- and low-demonstration states; their mixture weight is an online quality posterior
updated by bounded text evidence and the predicted action likelihood ratio. Fit-source-only,
quality-conditioned action-transition tensors use a shrunk Dirichlet prior and learned reliability
gates. Neither gold historical codes nor true quality enter inference.

Four capacity-matched modes isolate A-only, C-only, joint learning, and transition regularization.
Each is evaluated across five outer folds and five seeds, with separate source-disjoint inner fit,
selection, and calibration sets. Temperature scaling and adaptive action sets are calibrated by
source. The joint gate requires supported improvements over both single-task models, stable seed
direction, bounded Brier degradation, and no class collapse.

## 4. Results

The exact tables are in `results/research/publication_v1/` and
`results/research/publication_ac_v1/`; the figures are generated by
`tools/build_research_assets.py` and `tools/build_ac_assets.py`.

### 4.1 Classification

Target-only RoBERTa improves source-balanced macro-F1 by 0.0935 over TF-IDF, and the paired interval
excludes zero. Causal-history RoBERTa reaches 0.8163 versus 0.8108, but the difference interval
crosses zero. Its source-balanced log loss is 0.5880 versus 0.5587, a supported deterioration.

DASH-MI reaches 0.8116 and is not distinguished from either target-only or causal RoBERTa. Removing
its context branch changes macro-F1 by only 0.0011, interval crossing zero. Context alters 26 of
4,882 ensemble decisions. This rejects the registered hypothesis that a constrained context
residual would deliver a material improvement under source shift.

### 4.2 Vote distributions

Soft-linear strongly improves on the transcript prior for therapist and client vote log score. The
therapist improvement occurs in all seven transcripts; the client improvement occurs in six. Soft
versus hard-label differences are not statistically resolved: therapist log-score delta +0.0033,
client -0.0345, both intervals crossing zero.

PANEL-MI worsens therapist log score by 0.0435 relative to soft-linear and improves only two
transcripts. It therefore fails every directional component of the registered primary gate except
non-degradation in JSD. Its therapist JSD improves from 0.1260 to 0.1169, entropy mean absolute error
from 0.3725 to 0.2910, and entropy Spearman correlation from 0.3368 to 0.3611. The divergence between
proper scores indicates a sharper-versus-shape trade-off rather than a uniformly better model.

### 4.3 Early quality and next action

At ten therapist turns, Q-TRACE-MI is the best non-oracle Task A model: balanced accuracy 0.7000 and
low-class AUPRC 0.6276, compared with 0.6500 and 0.4921 for A-only. Its +0.0500 balanced-accuracy
delta has paired-source interval [-0.0734, 0.1731], so the positive effect is not established.

For Task C, C-only neural is best at 0.4251 source-balanced macro-F1 and 0.6904 Brier. Q-TRACE-MI
reaches 0.3742 and 0.7116, respectively. The candidate-minus-C-only macro-F1 interval is
[-0.0768, -0.0258], and all five seed contrasts are negative. Adaptive sets obtain 0.8844 coverage
at a target of 0.80, but average 3.12 of four labels and produce no singletons. The registered joint
gate fails.

## 5. Discussion

The largest robust result is not a novel architecture: it is the combination of a pretrained text
encoder and a leakage-resistant evaluation boundary. More elaborate context mechanisms fail to add
supported value. The current utterance contains most of the recoverable four-class signal, and
context can worsen probability quality even when top-1 F1 rises slightly.

The vote study shows that disagreement is predictable beyond transcript-level label prevalence.
That matters scientifically even though soft targets do not decisively beat hard targets: the model
can estimate a graded coding distribution on unseen dialogues. PANEL-MI's entropy/JSD advantage
suggests a future calibration or mixture model, but tuning such a model on these completed outer
predictions would be post-selection. It should be registered as a new study and evaluated on new
dialogues or an external corpus, not optimized against the seven reported clusters.

The Task A/C result similarly argues against adding complexity by default. The joint model extracts
a promising early quality signal, but the available low-quality sample is too small to resolve its
gain. For action forecasting, quality-conditioned transitions create negative transfer relative to
the C-only representation. Future work should register task-specific encoders or collect external
quality ratings rather than tune this mechanism on the completed outer predictions.

## 6. Limitations

The 119 source groups improve dependence control but are not a random clinical sample. The seven
multi-annotator transcripts support especially narrow inference. Demonstration speech, annotation
instructions, transcription choices, and source production may all differ from care settings. No
demographic metadata permits subgroup fairness analysis. Task A's high/low target is supplied by
source metadata rather than independent MITI rating. The study evaluates corpus labels, not
therapeutic quality, patient outcomes, causal mechanisms, or safety in use.

The paired intervals describe variation across observed source clusters under the registered
resampling scheme. They do not license universal probability statements. Multiple secondary
metrics are descriptive and are not treated as independent discoveries.

## 7. Reproducibility

Both dataset variants are commit-pinned and checksum-locked. Configurations were committed before
their corresponding performance runs. GPU and smoke gates precede full neural evaluation. A CPU
neural failure, an L-BFGS convergence failure, the deterministic numerical fallback, and both failed
candidate gates are retained. The separate Q-TRACE-MI gate failure is also retained. Prediction
ledgers, selection traces, package versions, hardware, seeds, code commits, and output hashes permit
exact metric reconstruction without releasing raw dialogue text.
