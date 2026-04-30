# Remediation Summary: glm_4_5_air_awq-pytorch-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[glm/glm_4_5_air_awq/pytorch-single_device-inference]

## Result
XFAIL — GLM-4.5-Air-AWQ-4bit is a 128-expert MoE with 104B parameters; at 4-bit AWQ the weights alone require ~52 GB, far exceeding n150 single-device DRAM (12 GB)

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
GLM-4.5-Air is a 128-expert sparse MoE model (128 routed experts, 46 layers, hidden_size=4096, moe_intermediate_size=1408) with approximately 104 billion total parameters. At 4-bit AWQ quantization via compressed-tensors, the model weights require approximately 52 GB of memory. The n150 single-device DRAM capacity is 12 GB. The test timed out because the model cannot be transferred to and run on a single n150 device — any attempt to do so either fails with OOM or hangs during the device transfer phase, eventually triggering the CI test timeout.

## Fix
Added `KNOWN_FAILURE_XFAIL` entry for `glm/glm_4_5_air_awq/pytorch-single_device-inference` in `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`. This is the appropriate disposition — the model is too large for single-device n150 and needs multi-device tensor-parallel execution, like `glm/causal_lm/pytorch-4.5_Air-single_device-inference` which is already `EXCLUDE_MODEL` for the same hardware-capacity reason.

## Verification
- pytest exit: FAIL
- Hardware:    not-run
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added KNOWN_FAILURE_XFAIL entry

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | dfd3ef5282325eb15522c9d1cb8c52fdff0992ea |
| tt-xla          | 04e6a540e65bc625599f7b368d982eaabca069a0 |
| tt-forge-models | 8de17dd75d50f6469bc0b89f101efd8729cf2a01 |
