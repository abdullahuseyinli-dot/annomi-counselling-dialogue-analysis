# Source-disjoint research evidence

This tree is independent of the portfolio aggregate exports in `results/main`, `results/extensions`,
and `results/summarisation`. Those earlier exports are preserved as development-consumed evidence.

Research outputs are written create-only. A command may confirm an existing byte-identical output,
but it refuses to replace evidence with different content. Large checkpoints and transient training
state remain under ignored `artifacts/`; compact prediction ledgers and result summaries are retained
here so every reported number can be reconstructed.
