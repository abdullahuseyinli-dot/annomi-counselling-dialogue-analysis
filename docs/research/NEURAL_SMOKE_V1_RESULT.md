# Neural CUDA smoke result v1

Both registered input paths passed their bounded engineering smoke tests on commit `28e6495`.
Neither command accessed outer-fold 0 test examples or produced a performance estimate.

| Input path | Maximum length | Training rows | Steps | Peak allocated GPU memory |
|---|---:|---:|---:|---:|
| target utterance | 256 | 256 | 8 | 3.71 GiB |
| target-first causal context | 384 | 256 | 16 | 3.31 GiB |

Both runs completed a forward pass, backward pass, optimizer update, and evaluation forward pass.
All probabilities were finite; the largest row-sum error was `1.1920928955078125e-07`.

PyTorch emitted a warning that its memory-efficient attention backward kernel is not bitwise
deterministic. This is consistent with the prospectively registered `warn_only` deterministic
policy and remains a limitation. The protocol therefore reports all five fixed optimization seeds
and uses their probability ensemble rather than treating one seed as definitive. Exact reruns also
require the recorded software, driver, GPU family, and model revision.
