# Maintenance Quick Start

This fork maintains compatibility and token-selection code around the upstream instructional-fingerprinting implementation. The small checks below use no model weights, network, GPU, API credentials, or training dependencies.

## Offline checks

Clone the fork, then run from its root with Python 3.9 or later:

```bash
git clone https://github.com/Ardeur-HK/Model-Fingerprint.git
cd Model-Fingerprint
python3 -m unittest discover -s tests -v
```

Expected: three passing tests. They cover input-only trainable tokens, materializing dataset columns for tokenization, and preserving shared/special tokens. The same suite runs in [maintenance CI](../.github/workflows/maintenance-checks.yml).

These are regression checks for the helper used by `run_clm.py`. They do not perform model inference, reproduce the paper's metrics, or establish that full training succeeds.

## Model reproduction

The [upstream instructions retained in the README](../README.md#dataset) describe dataset preparation, fingerprinting, downstream fine-tuning, and evaluation. The reported upstream environment is CUDA 11.3, PyTorch 2.0, and Python 3.9. The inherited [requirements.txt](../requirements.txt) is unpinned; a fresh, fully validated training environment is not supplied by the offline checks above.

For a new reproduction run, record the model and tokenizer revisions, upstream/fork commit, installed environment, dataset source and hash, configuration, seed, and original logs. Store personal results separately from the unchanged upstream `harmlessness_eval/` artifacts. Model and dataset access and redistribution conditions remain applicable.

The historical `inference.py` compatibility patch loads a serialized adapter with `weights_only=False`. It is intended for trusted, verified artifacts and is not a security improvement. Review the source and provenance of adapters and any `trust_remote_code` model before executing them. This Quick Start does not load either.

## Change in training behavior

The current token-selection repair includes tokens found only in the dataset's `input` column. Earlier code called `set.union` without retaining its result. A new training run using this repair may differ from a historical run; retain the old outputs and identify the exact code version instead of overwriting or relabeling prior experiments.
