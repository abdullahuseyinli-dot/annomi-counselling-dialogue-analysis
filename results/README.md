# Result evidence

This directory contains compact, aggregate exports from the completed experiment lineage.

- `main/`: the locked RoBERTa-versus-elastic-net comparison, calibration, per-class
  metrics, uncertainty, and aggregate error slices.
- `summarisation/`: method-level and behaviour-level summary evaluation.
- `extensions/`: transcript-quality classification, context/topic robustness, and
  next-behaviour forecasting.
- `protocol/`: the held-out transcript split and selected RoBERTa configuration.
- `provenance.json`: hashes for every imported result file and the source notebook.

These files contain no counselling utterances, model weights, logits, or row-level
predictions. Do not overwrite them with a new run. Export new evidence separately, review
it, and update the lineage only through an explicit commit.
