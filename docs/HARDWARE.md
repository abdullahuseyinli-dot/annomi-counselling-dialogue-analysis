# Hardware and execution environment

Hardware values in this document are copied from retained run evidence. They describe the machine
used for the completed neural campaign; they are not a benchmark minimum or a promise of runtime on
other systems.

## Recorded neural environment

`results/research/gate1/gpu_environment_v1.json` records:

| Item | Recorded value |
|---|---|
| Operating-system report | Windows 10 (`10.0.26200`) |
| Python | 3.11.9 |
| GPU | NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU |
| Reported GPU memory | 12,227 MiB |
| Driver | 596.72 |
| CUDA runtime | 13.0 |
| Compute capability | 12.0 |
| PyTorch | 2.14.0+cu130 |
| Transformers | 5.16.1 |
| BF16 gate | Supported and finite |

The environment gate produced no performance result. Model-specific smoke records under
`results/research/gate1/` capture finite probabilities, optimizer steps, and measured peak
allocation before full runs.

## Workload classes

- Repository validation, unit tests, metric reconstruction, and sparse baselines do not require a
  GPU.
- Frozen-embedding downstream heads can run on CPU after embeddings exist, but extracting the
  pinned transformer representations is a separate neural dependency.
- Full RoBERTa, DASH-MI, Q-TRACE-MI, and SAFE-MI reproduction follows the recorded CUDA/BF16 gate.

A preserved failure record shows that the default package index previously resolved a CPU-only
PyTorch wheel despite a visible GPU. The maintained lock separates the `neural-cpu` test extra from
the CUDA 13.0 `neural` extra. Reproducers should verify the resolved build rather than infer CUDA
support from the machine alone.

Total disk, RAM, energy, and portable runtime requirements were not systematically benchmarked and
are therefore not specified. Per-run elapsed times remain in the original summaries as provenance,
not service-level estimates.
