# Contributions and upstream attribution

This is a maintenance/reproduction fork of [Instructional Fingerprinting](https://github.com/cnut1648/Model-Fingerprint). The upstream snapshot is commit [`4ae5e8a124c37f25a3711c407e85a45fda6ecb08`](https://github.com/cnut1648/Model-Fingerprint/commit/4ae5e8a124c37f25a3711c407e85a45fda6ecb08). The paper authors, fingerprinting method, adapter architecture, training/evaluation pipelines, hosted models, and bundled experimental results remain attributed to the original project.

## Local engineering by Puhao Zheng (`Ardeur-HK`)

Historical commit [`18f0cf02bcfaebed5276eeec94d7aa2e0a31afeb`](https://github.com/Ardeur-HK/Model-Fingerprint/commit/18f0cf02bcfaebed5276eeec94d7aa2e0a31afeb) changed three files:

- `run_clm.py`: materialized two dataset columns as lists before tokenization.
- `inference.py`: explicitly selected `weights_only=False` for the existing serialized adapter format.
- `configs/adapter.yaml`: changed the GPT-J-6B total batch size from 48 to 24.

The current maintenance change also repairs input-token selection in [run_clm.py](../run_clm.py) through [a dependency-free helper](../utils/fingerprint_tokens.py). Both instruction and input tokens now contribute to the trainable-token set. [Regression tests](../tests/test_fingerprint_tokens.py) and [CI](../.github/workflows/maintenance-checks.yml) cover that behavior without loading a model. AI tools assisted this maintenance and documentation work.

These are compatibility and correctness changes around a third-party method. They do not establish a new fingerprinting algorithm or a measured detection/performance improvement. Adapter initialization and other upstream training semantics are unchanged by this maintenance patch and require separate model-level validation.

## Result provenance

The inherited `harmlessness_eval/` files, paper figures, A100 timings, and hosted models referenced in the original README are upstream results. They have not been converted into personal reproduction claims. A separate reproduction experience does not make those inherited files new experimental evidence.

No existing experimental output, configuration, or Git history is rewritten by this maintenance change. The original [MIT license and copyright](../LICENSE), author list, and paper citation remain in place. Model weights and datasets retain their own terms.
