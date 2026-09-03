# Neural execution plan v1

This stage is governed by `configs/research/neural_v1.json` and was registered before inspecting
any new neural validation or outer-fold result.

The first comparison separates encoder capacity from dialogue context:

1. `roberta_utterance` sees only the target therapist utterance.
2. `roberta_flat_causal10` sees the target first, followed by up to ten preceding turns in reverse
   chronological order. Right truncation therefore cannot remove the target or favor distant over
   recent history.

Each of four fixed recipes is assessed by three-fold source-grouped validation inside every outer
training partition. The selected recipe is retrained on the full outer-training partition for the
rounded median inner best epoch. Five fixed optimization seeds are then run; the primary estimate
uses seed-averaged probabilities, while every seed remains visible.

Training loss weights give each source equal total mass and then correct class imbalance using only
the active training partition. No outer-test labels affect tokenization, weighting, early stopping,
recipe selection, epoch choice, or calibration. The five outer folds remain the only headline
evaluation.

Before the complete run, a bounded smoke test must demonstrate a real CUDA forward/backward pass,
finite probabilities, valid label order, and acceptable peak memory. Smoke output is an engineering
gate and is not a performance result.

Model checkpoints and Hugging Face cache files live under ignored `artifacts/`. Completed evidence
contains row-level predictions, source-grouped selections, hashes, resolved software/device
provenance, and the exact pretrained revision. Evidence files are create-only.
