# Full experiment source

`pipeline_source.ipynb` preserves the complete Python cell sequence while omitting
execution outputs and long-form narrative. It is intended for audit and full reruns, not
as the primary repository walkthrough.

Before running it:

1. Create a fresh Python 3.11 environment.
2. Run `uv sync --locked --extra analysis --extra dev --extra neural`.
3. Acquire both pinned variants with `uv run python tools/download_dataset.py --variant simple`
   and `uv run python tools/download_dataset.py --variant full`.
4. Set `ANNOMI_PROJECT_DIR` to the repository root if the notebook is launched elsewhere.
5. Use a new artifact directory and retain logs, configuration, and seeds.

The pipeline is computationally expensive. A CUDA-capable GPU is recommended for encoder
training. The notebook never installs packages at runtime.

`requirements-experiment.txt` and `requirements-lock.txt` document the earlier portfolio
environment. They are retained for provenance and are not the dependency authority for the
current research package; `pyproject.toml` and `uv.lock` are authoritative.

The Task A/C tracks are executable from the package CLI:

```bash
uv run annomi-research run-ac-baselines
uv run annomi-research smoke-qtrace
uv run annomi-research run-qtrace
uv run annomi-research smoke-safe-mi
uv run annomi-research run-safe-mi
uv run annomi-research run-safe-mi-extension
```

The governing protocols and limits on interpretation are recorded in
`docs/research/QTRACE_MI_REGISTRATION_V1.md` and the SAFE-MI protocol files under
`configs/research/`. Q-TRACE uses a frozen encoder and a causal session model; raw text, embedding
caches, and weights remain outside Git. The completed metrics, failed gates, and interpretation are
recorded in `docs/research/QTRACE_MI_V1_RESULT.md` and `docs/research/SAFE_MI_V2_RESULT.md`.
