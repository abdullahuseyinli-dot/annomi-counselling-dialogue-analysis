# Full experiment source

`pipeline_source.ipynb` preserves the complete Python cell sequence while omitting
execution outputs and long-form narrative. It is intended for audit and full reruns, not
as the primary repository walkthrough.

Before running it:

1. Create a fresh Python 3.11 environment.
2. Install `requirements-experiment.txt`.
3. Run `python tools/download_dataset.py`.
4. Set `ANNOMI_PROJECT_DIR` to the repository root if the notebook is launched elsewhere.
5. Use a new artifact directory and retain logs, configuration, and seeds.

The pipeline is computationally expensive. A CUDA-capable GPU is recommended for encoder
training. The notebook never installs packages at runtime.

The registered Task A/C track is executable from the package CLI:

```bash
python -m annomi_research run-ac-baselines
python -m annomi_research smoke-qtrace
python -m annomi_research run-qtrace
```

Its authority and claim boundary are recorded in
`docs/research/QTRACE_MI_REGISTRATION_V1.md`. Q-TRACE uses a frozen encoder and a causal session
model; raw text, embedding caches, and weights remain outside Git. The completed metrics, failed
joint gate, and interpretation are recorded in `docs/research/QTRACE_MI_V1_RESULT.md`.
