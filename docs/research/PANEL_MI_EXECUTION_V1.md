# PANEL-MI execution log v1

## Failed full attempt: 2026-09-04

The first full invocation used code commit `d417150e9a16c12e3d4472865cd8cdf537379a4c` and the
registered configuration hash `4456c464c505008b42d1d4c06e2e6589b2946856768b09d17c39bd1e7c8e50f5`.
It stopped during nested selection for client outer transcript 56 when scikit-learn reported an
L-BFGS `ConvergenceWarning` with status 2 (abnormal line-search termination). The strict runner
promoted that warning to an exception.

No result directory, prediction ledger, aggregate metric, or summary was written. Recipe scores
were not printed or inspected. The ignored frozen-embedding cache is the only reusable computation
from the attempt.

Before restarting, the configuration was amended to use an explicit `1e-6` tolerance and retry the
identical multinomial logistic objective with scikit-learn's `newton-cholesky` optimizer only when
L-BFGS emits a convergence warning. Every selected and final fit records which optimizer completed.
This is a numerical engineering fallback, not a change to the model, folds, targets, hyperparameter
grid, endpoint, or success gate.
