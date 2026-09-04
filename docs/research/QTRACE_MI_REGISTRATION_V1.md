# Q-TRACE-MI Task A/C registration v1

`configs/research/protocol_ac_v1.json` and `configs/research/qtrace_mi_v1.json` define this study.
The neural architecture, ablations, endpoints, calibration procedure, and success gate were fixed
before Q-TRACE-MI was evaluated. Earlier Task A/C results and an initial baseline run informed model
development and are reported only as exploratory context.

## Questions

Task A estimates the uploader-designated demonstration-quality label after 3, 5, 10, and 20
observed therapist turns and at the session endpoint. The primary checkpoint is 10 therapist turns.
The label was derived from source-video metadata, so this task is not independent clinical-fidelity
measurement or outcome prediction.

Task C forecasts the behaviour recorded for the next therapist turn, conditional on an observed
client-to-therapist handoff. Exactly 4,743 decisions satisfy this definition. The target therapist
utterance is unavailable to the model. A forecast describes the demonstrator's observed action; it
does not establish that an action is clinically appropriate, safe, optimal, or outcome-improving.

## Data and leakage boundary

The existing five exhaustive outer folds group normalized video URLs. Within each outer-training
partition, another deterministic five-fold source split reserves one fold for model selection and
one disjoint fold for calibration. Training, validation, calibration, and test sources cannot
overlap. Every outer test source appears exactly once.

Permitted inputs are frozen pretrained text embeddings, speaker identity, causal order, and soft
behaviour distributions predicted by the model. The following are forbidden from confirmatory
inputs: video title, topic, eventual length before the endpoint, end-normalized position, future
turns, true session quality for Task C, the target therapist text, and gold historical behaviour
codes. Gold-code aggregates and a gold-history Markov model are retained only as named oracle
baselines.

## Registered model

Q-TRACE-MI expands to *Quality-conditioned Transition-Regularized Causal Evidence for
Motivational Interviewing*.

1. A commit-pinned frozen RoBERTa-base encoder produces one mean-pooled vector per utterance.
2. Auxiliary heads predict therapist and client codes as probability distributions. These soft
   distributions, rather than gold codes, enter a unidirectional session GRU.
3. Two policy heads model the next therapist action under latent high- and low-demonstration
   states. Their mixture weight is the model's accumulated online quality posterior; the true
   quality label is used only in training losses.
4. Quality evidence combines a bounded text residual with the soft observed-action likelihood ratio
   under the two policies. This makes the Task A trajectory auditable turn by turn.
5. Quality-specific action-transition tensors are estimated from fit sources only, with a
   hierarchically shrunk Dirichlet prior. At inference, predicted previous-action and current-client
   distributions are marginalized through this tensor. A learned gate controls how strongly each
   policy uses it.
6. Temperature scaling and deterministic adaptive prediction sets are calibrated on held-out
   sources. Conformal risk control treats each source's mean miscoverage as one calibration loss.

The transition term is motivated by
[Rudolph et al. (2026)](https://aclanthology.org/2026.findings-acl.1271/), but transition
regularization alone is not claimed as novel. The set-valued endpoint responds to the ambiguity
identified in the direct AnnoMI forecasting study
([Wu et al., 2022](https://www.isca-archive.org/interspeech_2022/wu22c_interspeech.html)) and uses
the conformal prediction principle exemplified by
[Huang et al. (2024)](https://proceedings.mlr.press/v235/huang24aa.html). The proposed contribution
is their source-safe joint use with uncertainty-propagating histories and online Task A evidence.

## Comparisons and endpoints

Classical Task A controls are a source prior, observed-structure-only model, raw-prefix TF-IDF, and
gold-code oracle. Task C controls are a source prior, causal-context TF-IDF, and gold-history Markov
oracle. Neural ablations are A-only, C-only, joint without transition regularization, and full
Q-TRACE-MI. All neural finalists use the same encoder features, capacity, partitions, and seeds.

Task A's primary metric is source-balanced balanced accuracy at 10 therapist turns; low-class
AUPRC, macro-F1, Brier score, log loss, and other checkpoints are secondary. Task C's primary metric
is source-balanced macro-F1; Brier score, log loss, per-class performance, prediction-set coverage,
mean set size, and singleton rate are secondary. Paired intervals resample complete sources. Seeds
are stability checks, not independent sampling units.

The joint success gate requires a positive Task A paired-source interval over A-only, at least
+0.02 Task C macro-F1 over C-only with a positive paired-source interval, no more than +0.01 Task C
Brier degradation, concordant improvement in at least four of five seeds for both tasks, and no
Task C class collapse. A failed gate remains a reportable negative result.

## External validation boundary

No additional annotation is required for this benchmark. Stronger claims about MI fidelity require
independent MITI ratings. MI-TAGS contains 242 demonstrations with MITI 4.2 behaviour codes and
global scores and is a candidate external resource after access and source-overlap auditing
([Cohen et al., 2024](https://aclanthology.org/2024.lrec-main.1017/)). Until such validation exists,
the words clinical fidelity, effectiveness, and patient outcome are outside the claim scope.
