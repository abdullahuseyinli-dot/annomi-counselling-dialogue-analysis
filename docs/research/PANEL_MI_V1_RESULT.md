# PANEL-MI seven-transcript result v1

The complete registered run finished on 2026-09-04. It used code commit
`937cd4f4ebd150e8e3d8e23c0ea70f8ea28d41a7`, configuration hash
`aaa9a31df44094a0e7cecae2d3f908c7ea38ee7035abf05d8b35e7b86115076b`, all
seven outer transcript folds, and all five registered PANEL-MI seeds. The row-level evidence
reconstructs exactly from `results/research/multiannotator_v1/panel_mi/`.

## Main result

Text carries a strong, source-held-out signal about the distribution of therapist coding votes.
Soft-linear label-distribution learning lowers therapist vote log score from **1.4441** for the
transcript-balanced prior to **0.8132**. The paired transcript delta is **-0.6309**, with a 95%
cluster-bootstrap interval of **[-0.7257, -0.5055]** and an exact one-sided sign-flip
**p = 0.0078**; all seven held-out transcripts improve.

The linked client task replicates the signal at smaller magnitude. Soft-linear lowers vote log score
from **1.0566** to **0.9453**: delta **-0.1113**, 95% interval **[-0.1773, -0.0402]**,
exact **p = 0.0156**, with six of seven transcripts improving.

| Task | Model | Vote log score ↓ | Vote Brier ↓ | JSD ↓ | Plurality macro-F1 ↑ | Entropy MAE ↓ |
|---|---|---:|---:|---:|---:|---:|
| Therapist | Transcript-balanced prior | 1.4441 | 0.5953 | 0.3099 | 0.0818 | 0.7688 |
| Therapist | Hard-linear | **0.8099** | 0.2351 | 0.1381 | 0.7401 | 0.4380 |
| Therapist | Soft-linear | 0.8132 | **0.2245** | 0.1260 | **0.7430** | 0.3725 |
| Therapist | PANEL-MI | 0.8567 | 0.2272 | **0.1169** | 0.7326 | **0.2910** |
| Client | Transcript-balanced prior | 1.0566 | 0.3781 | 0.1822 | 0.2417 | 0.5623 |
| Client | Hard-linear | 0.9798 | 0.3235 | 0.1582 | 0.3399 | 0.4950 |
| Client | Soft-linear | **0.9453** | **0.3011** | **0.1453** | 0.4405 | 0.4284 |
| Client | PANEL-MI | 1.0253 | 0.3380 | 0.1518 | **0.4724** | **0.3757** |

Bold values are within-task numerical optima, not separate significance claims.

## Registered candidate decision

PANEL-MI does **not** pass the registered primary success gate. Against soft-linear on therapist
turns, its log-score delta is **+0.0435** (positive is worse), with a 95% interval of
**[-0.0353, +0.1074]**, exact one-sided **p = 0.8672**, and improvement on only two of seven
transcripts. It also increases Brier by 0.0027.

The negative primary result hides a useful Pareto trade-off: PANEL-MI obtains the best therapist JSD
(0.1169 versus 0.1260) and entropy MAE (0.2910 versus 0.3725). Its item-level predicted-versus-vote
entropy Spearman correlation is 0.3611 versus 0.3368 for soft-linear. This suggests that the panel
heads capture aspects of disagreement shape while sacrificing strict predictive likelihood. It is a
hypothesis for external follow-up, not grounds to override the preregistered endpoint.

Soft labels do not significantly dominate hard plurality labels in this small study. Therapist
soft-minus-hard log score is +0.0033 (exact p = 0.5625); client is -0.0345 (exact p = 0.1328), with
both cluster intervals crossing zero. Soft-linear is nevertheless the most balanced registered
choice because it wins client log/Brier/JSD and therapist Brier/plurality F1 without discarding vote
mass.

## Selection and numerical checks

- Therapist hard-linear selected `C=0.1` in all seven outer folds; client selected `C=0.01` in all
  seven. Soft-linear selected `C=0.1` or `C=1.0` for therapist and `C=0.1` throughout client.
- Six of seven therapist folds selected rank-4 PANEL-MI with shrinkage 0.01. Client selections were
  mixed across ranks 2/4 and shrinkage 0.01/0.1. Final epochs ranged from 40 to 66.
- L-BFGS completed all therapist fits. Two of 175 client soft-linear fits used the preregistered
  `newton-cholesky` convergence fallback; all other reported linear fits used L-BFGS.
- Every item has one out-of-transcript prediction per deterministic model and one per each of five
  PANEL-MI seeds. Seed probabilities are averaged before inference.

## Boundary

No new labels were created: this study uses the ten existing annotations per item. It estimates the
response distribution of this anonymous ten-person panel on seven public demonstration dialogues.
The 428 utterances are the analysis units and the seven transcripts are the inferential clusters;
the 4,280 annotation rows are never treated as independent observations. Results do not establish
clinical validity, population-wide annotator behavior, algorithmic novelty, or state of the art.
