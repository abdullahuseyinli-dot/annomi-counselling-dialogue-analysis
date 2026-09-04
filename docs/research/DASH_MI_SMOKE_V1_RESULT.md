# DASH-MI CUDA smoke result v1

Status: **pass**. This is an engineering gate, not a performance result.

The largest registered DASH-MI input recipe completed a real one-epoch forward/backward training
run on 256 source-grouped inner-training rows and predicted 128 inner-validation rows. The run used
16 optimizer steps, reached 3.64 GiB peak allocated GPU memory, and produced finite full-model and
target-only probabilities. The maximum row-sum error across both outputs was
`1.1920928955078125e-07`.

The smoke test did not touch outer-test data. It exercised the 192-token target stream, 256-token
history stream, target-conditioned attention, disagreement-distribution loss, context dropout,
auxiliary target loss, residual output, optimizer groups, BF16 autocast, and both inference paths.

PyTorch again warned that its memory-efficient attention backward kernel is nondeterministic under
the registered `warn_only` policy. This limitation is retained rather than concealed. The final
evaluation uses all five fixed seeds and reports each seed separately.

The machine-readable evidence is `results/research/gate1/dash_mi_smoke_v1.json`. It records code
commit `67f3cefaecf71108de63758a98678ebec0caf465`, the DASH-MI configuration hash, split hash,
runtime versions, GPU model, memory, probability checks, and the explicit statement that no
performance result was produced.
