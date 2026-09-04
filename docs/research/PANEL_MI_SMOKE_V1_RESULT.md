# PANEL-MI smoke gate v1

The registered engineering smoke gate passed on 2026-09-04 at code commit
`090f50540d9cf602b5cb49797fe9582c26a8f6d0`.

- The pinned frozen RoBERTa-base encoder ran on an NVIDIA RTX PRO 3000 Blackwell Generation
  Laptop GPU and returned finite 768-dimensional embeddings.
- Training-only PCA returned the registered 32 dimensions.
- A two-epoch PANEL-MI head fit completed on 171 development items and returned finite
  probabilities for 12 development-validation items.
- The maximum probability-sum error was `1.1920928955078125e-07`.
- Peak encoder memory was 636,997,632 bytes and the gate took 4.46 seconds.

Transcript 7 was designated as the outer test partition and its votes were not used. Transcript 27
served only as the development validation partition. This is an engineering gate, not a performance
result; it reports no evaluation metric and cannot support a model-quality claim.

The complete machine-readable record is
`results/research/gate1/panel_mi_smoke_v1.json`.

## Post-amendment rerun

After the recorded L-BFGS engineering failure and convergence-fallback amendment, the same smoke
gate passed again at commit `0b0e598980bbe979ef3f5d03f34e69d7653d8494` with configuration hash
`aaa9a31df44094a0e7cecae2d3f908c7ea38ee7035abf05d8b35e7b86115076b`. The dimensions,
partition sizes, peak memory, and maximum probability-sum error were unchanged; elapsed time was
1.97 seconds. The second immutable record is
`results/research/gate1/panel_mi_smoke_v2.json`.
