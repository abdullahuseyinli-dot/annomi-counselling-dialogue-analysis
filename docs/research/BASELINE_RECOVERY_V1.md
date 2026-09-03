# Sparse baseline execution recovery v1

The first prospective baseline command at protocol commit `51bbd18` repeatedly reached the SAGA
iteration ceiling during inner-fold tuning. It was manually interrupted before any outer-fold
predictions or aggregate results were written. The failure is retained in
`results/research/failures/baseline_v1_attempt1.json`.

The recovery does not change the task, features, outer folds, selection metric, number of recipes,
or candidate model family. It uses `SGDClassifier(loss="log_loss", penalty="elasticnet")`, which
optimizes the same sparse linear logistic objective with bounded deterministic recipes. The recovery
was registered before observing outer-fold results.
