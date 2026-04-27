# Remediation Summary: cross_encoder/passage_ranking/pytorch-ms-marco-MiniLM-L12-v2 Inference Test

## Test

`tests/runner/test_models.py::test_all_models_torch[cross_encoder/passage_ranking/pytorch-ms-marco-MiniLM-L12-v2-single_device-inference]`

## Result

**SILICON_PASS** — The test passed without any code changes needed.

## Details

The reported failure `PCC comparison failed. Calculated: pcc=0.0. Required: pcc=0.95` could not
be reproduced on the configured branch (`arch-c-36-tt-xla-dev/nsmith/2026-04-22_16-58/hf-bringup-39`).

Running the test on the configured branch produced a PASS result with benchmark completing
successfully and PCC well above the 0.95 threshold.

No changes were required to any of the three subprojects (tt-xla, tt-mlir, tt-metal).

## Submodule Hashes

- tt-metal: `3fa4d753550dba1d4aacc9af45b111ae540f63fc`
- tt-mlir:  `553c0632b353f8ac457aba0d01a460a5e0f5b5ee`
- tt-xla:   `bc9390c8d118ece9d516712ca66c4aae794a7e11`
