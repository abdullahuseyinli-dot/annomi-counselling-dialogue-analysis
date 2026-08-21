# Result lineage

The public evidence is a curated subset of the completed run artifacts. File-level hashes
are recorded in `results/provenance.json`; the executed source pipeline is identified only by its
SHA-256 digest, so no machine-specific path or personal identifier is required.

The headline lineage is:

1. `results/protocol/official_split.json` fixes transcript membership.
2. `results/protocol/roberta_config.json` fixes the selected encoder configuration.
3. `results/main/model_comparison.csv` fixes held-out metrics.
4. `results/main/metric_deltas.csv` records direct arithmetic differences.
5. `results/main/grouped_significance.csv` records transcript-grouped uncertainty.
6. `results/main/mcnemar_summary.csv` records paired item correctness.
7. `results/main/calibration.csv` and `calibration_summary.json` record probability quality.

A stale narrative sentence had described BERTopic + MMR as the overall summarisation winner.
The exported aggregate table shows the opposite: KMeans + MMR scores 2.790 overall
usefulness versus 2.445. The repository text follows the recorded table and retains both
methods' component scores so the trade-off remains auditable.
