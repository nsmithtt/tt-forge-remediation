# Remediation Summary: glm_4_5_air_awq_fp16mix-causal_lm-pytorch-GLM_4_5_Air_AWQ_FP16Mix-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[glm_4_5_air_awq_fp16mix/causal_lm/pytorch-GLM_4_5_Air_AWQ_FP16Mix-single_device-inference]

## Result
XFAIL — QuantTrio/GLM-4.5-Air-AWQ-FP16Mix is a 128-expert MoE with ~104B parameters; the AWQ-marlin + FP16Mix weights total 68 GB on disk, far exceeding single-device p150b DRAM (32 GB)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-model-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Test exceeded configured timeout and was killed

## Root cause
`QuantTrio/GLM-4.5-Air-AWQ-FP16Mix` is a Glm4Moe model with 46 hidden layers, 128 routed experts, and ~104B total parameters. The "FP16Mix" designation means most routed experts use AWQ-marlin 4-bit quantization while the first 2 and last 2 layers, and all 46 shared expert layers, remain in FP16. Despite the quantization, the total on-disk size is 68 GB (measured via HuggingFace metadata). The p150b single device has 32 GB DRAM. The test timed out because the model cannot be loaded onto the device — the device transfer either OOMs or hangs, eventually triggering the CI timeout.

Additionally, the loader is missing a `requirements.txt` entry for `compressed-tensors`. Without this library, transformers does not recognize the `awq_marlin` quantization type, skips dequantization entirely, and loads the model with incorrect weight layouts. This loader bug was fixed on the remediation branch, though it does not affect the XFAIL disposition.

## Fix
1. **Loader (tt-forge-models)**: Added `compressed-tensors` to `glm_4_5_air_awq_fp16mix/causal_lm/pytorch/requirements.txt` so that the `awq_marlin` quantization type is recognized by transformers.
2. **Test config (tt-xla)**: Added `KNOWN_FAILURE_XFAIL` entry for `glm_4_5_air_awq_fp16mix/causal_lm/pytorch-GLM_4_5_Air_AWQ_FP16Mix-single_device-inference` in `tests/runner/test_config/torch/test_config_inference_single_device.yaml`.

## Verification
- pytest exit: FAIL
- Hardware:    not-run
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added KNOWN_FAILURE_XFAIL entry
- `tt-forge-models/glm_4_5_air_awq_fp16mix/causal_lm/pytorch/requirements.txt` — added (new file) with compressed-tensors dependency

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | bc5a01792fde23080b5c49f2a24c17b7527a4eeb |
| tt-forge-models | de114c7fea2c18d59c5899e49b9aff98db72a2f9 |
