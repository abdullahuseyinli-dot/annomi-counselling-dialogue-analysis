# DASH-MI inference registration v1

These analyses were registered after completing and immutably committing the DASH-MI prediction
ledger, but before computing any DASH-MI bootstrap draws or confidence intervals.

The sole primary comparison is DASH-MI versus the independently trained target-only RoBERTa
ensemble. It uses the locked candidate success gate: at least +0.02 source-balanced macro-F1, a
paired 95% source-bootstrap interval excluding zero, at least four positive matched seeds, Brier
degradation no greater than +0.01, and no class collapse.

Two secondary comparisons do not use the candidate promotion gate:

1. Full DASH-MI versus its target-only inference ablation tests whether the learned context
   residual contributes within the same jointly trained model. The evaluator reads the registered
   `prob_target_only_*` and `target_only_prediction` columns without altering the source ledger.
2. DASH-MI versus flat causal-context RoBERTa estimates the uncertainty around their benchmark
   ranking. It is descriptive because both candidates were developed in the same adaptive cycle.

All three comparisons sample the same 119 source-video clusters with replacement 10,000 times,
retain all utterances within each sampled source, give each source instance equal mass, and use
bootstrap seed `20260904`. Seeds are reported as matched stability diagnostics and are not treated
as independent sampling units. The secondary analyses must not be promoted to confirmatory claims
based on favorable intervals.
