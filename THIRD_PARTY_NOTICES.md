# Third-party notices

This repository's original code and documentation are released under the MIT License.
That license does not replace the terms attached to external datasets, pretrained model
weights, libraries, or source media.

## AnnoMI

Raw counselling text is not redistributed. `tools/download_dataset.py` retrieves the
simplified dataset from the official repository at a pinned commit and verifies its
digest. The official AnnoMI repository did not expose a separate dataset licence when this
notice was prepared. Public availability is not, by itself, a licence grant. Users should
review the current upstream terms, source-media permissions, and cited papers before using
or redistributing the data. The repository's MIT licence covers this project's original
software and documentation, not the upstream dialogue data or third-party annotations.

- Repository: https://github.com/uccollab/AnnoMI
- Initial paper: https://doi.org/10.1109/ICASSP43922.2022.9746035
- Extended paper: https://doi.org/10.3390/fi15030110

## Pretrained models

No pretrained or fine-tuned model weights are included.

- `FacebookAI/roberta-base`: MIT license
  (https://huggingface.co/FacebookAI/roberta-base)
- `sentence-transformers/all-MiniLM-L6-v2`: Apache-2.0 license
  (https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- `microsoft/deberta-v3-base`: MIT license
  (https://huggingface.co/microsoft/deberta-v3-base)

Python package licenses remain with their respective projects and distributions.
