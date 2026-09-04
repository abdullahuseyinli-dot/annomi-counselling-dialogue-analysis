# Artifact inventory

## Tracked evidence

| Location | Contents | Role |
|---|---|---|
| `configs/research/` | Machine-readable protocols, recipes, seeds, gates, and comparisons | Governs each research lineage |
| `results/research/gate0/` | Privacy-safe data audit, legacy inventory, and fixed source folds | Data and split gate |
| `results/research/gate1/` | Environment and bounded smoke records | Engineering gate, not performance evidence |
| `results/research/baseline_v1/` | Nested sparse out-of-source predictions and summaries | Main classical baseline |
| `results/research/neural_v1/` | Per-seed and ensemble predictions, selections, and summaries | Main neural evidence |
| `results/research/multiannotator_v1/` | Vote-distribution predictions, inference, and selection trace | Multi-annotator evidence |
| `results/research/ac_v1/` | Earlier Task A/C baselines and Q-TRACE-MI evidence | Task A/C study with a protocol fixed before evaluation |
| `results/research/safe_mi_v2/` and `safe_mi_v2_1/` | Staged SAFE-MI and post-hoc extension ledgers | Exploratory Task A/C lineage |
| `results/research/publication_*` | Derived exact tables plus hash manifests | Tables prepared for reports and papers |
| `assets/research/` | Builder-generated figures | Visualizations derived from tracked results |

The legacy notebook and earlier aggregate exports are retained for reproducibility. The principal
findings are based on the source-disjoint experiments in `results/research/`.

## Local-only artifacts

Raw data, source text, embeddings, caches, model checkpoints, and transient training state belong in
ignored paths such as `data/raw/` and `artifacts/`. No raw dialogues or trained weights are part of
the repository release candidate.

## Integrity and regeneration

`results/provenance.json` records hashes for the curated legacy exports. Each publication directory
contains its own manifest binding source summaries, derived tables, figures, and builder code.
Research validators recompute metrics from row-level probability ledgers instead of trusting only
aggregate JSON.

Existing evidence files are not overwritten. A byte-identical rerun may confirm an existing
artifact; a different payload must be written to a new versioned directory. Failed gates,
quarantines, fallbacks, and prior dry runs remain part of the study record.

Before distributing an archive, use the [release evidence gate](RELEASE_EVIDENCE_GATE.md) to verify
that local-only files, build products, and machine-specific paths have not entered the package.
