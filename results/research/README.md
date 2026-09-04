# Source-disjoint research evidence

This tree is independent of the portfolio aggregate exports in `results/main`, `results/extensions`,
and `results/summarisation`. Those earlier exports are preserved as development-consumed evidence.

Research outputs are written create-only. A command may confirm an existing byte-identical output,
but it refuses to replace evidence with different content. Large checkpoints and transient training
state remain under ignored `artifacts/`; compact prediction ledgers and result summaries are retained
here so every reported number can be reconstructed.

## Current contents

- `gate0/`: privacy-safe data audit, legacy inventory, and fixed source folds.
- `gate1/`: CUDA, neural, DASH-MI, and PANEL-MI engineering gates.
- `baseline_v1/`: nested source-grouped prior and TF-IDF predictions.
- `neural_v1/`: five-seed target-only RoBERTa, causal RoBERTa, and DASH-MI evidence.
- `comparisons/`: paired 10,000-resample source-bootstrap comparisons.
- `multiannotator_v1/panel_mi/`: seven-transcript, ten-vote label-distribution study.
- `publication_v1/`: exact compact tables and a source/asset hash manifest.
- `ac_v1/`: source-disjoint Task A/C baselines, Q-TRACE ledgers, calibration, and paired inference.
- `publication_ac_v1/`: exact Task A/C tables and a source/asset hash manifest.

Run `python -m annomi_research validate` to reconstruct the executable evidence. Run
`python tools/build_research_assets.py` to reproduce the publication tables and figures; the
create-only writer refuses drift under an existing filename.

Run `python tools/build_ac_assets.py` to reproduce the separate Task A/C tables and figures.

The main numerical classification score is 0.8163 source-balanced macro-F1 for causal RoBERTa. The
supported neural gain is target-only RoBERTa over TF-IDF (+0.0935, interval [0.0778, 0.1097]). In
the separate multi-annotator study, soft-linear vote prediction beats the transcript prior on both
therapist and client log score. DASH-MI and PANEL-MI fail their registered primary gates and remain
visible as negative results. In the Task A/C track, Q-TRACE-MI leads non-oracle early-quality
classification at 0.7000 t10 balanced accuracy, while C-only neural leads next-action forecasting
at 0.4251 source-balanced macro-F1. Q-TRACE-MI fails its joint gate because its Task A interval
includes zero and its Task C result is worse than C-only.
