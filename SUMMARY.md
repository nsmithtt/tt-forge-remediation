# Remediation Summary: l3_3_70b_euryale_heretic_gguf-causal_lm-pytorch-L3_3_70B_Euryale_v2_3_Heretic_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[l3_3_70b_euryale_heretic_gguf/causal_lm/pytorch-L3_3_70B_Euryale_v2_3_Heretic_GGUF-single_device-inference]

## Result
XFAIL — L3.3-70B Euryale Heretic GGUF (70B params) dequantizes to ~140 GB BF16, far exceeding n150 (12 GB) and p150b (32 GB) single-device DRAM

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-70b-exceeds-single-device-dram

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
L3.3-70B Euryale v2.3 Heretic is a 70B-parameter LLaMA 3.3 model distributed as a Q4_K_M GGUF (~37 GB on disk). When loaded by transformers, it is dequantized to BF16, producing a weight tensor footprint of approximately 140 GB. The n150 device has only ~12 GB of DRAM. The test times out during model loading as the host exhausts RAM trying to materialize the full BF16 tensor before device transfer. This is a hardware capacity ceiling, not a compiler bug.

## Fix
Updated `tests/runner/test_config/torch/test_config_inference_single_device.yaml` in tt-xla to mark the test `KNOWN_FAILURE_XFAIL` with a hardware capacity reason. No compiler stack changes were needed.

## Verification
- pytest exit: TIMEOUT (not run on silicon — hardware class XFAIL)
- Hardware: n150
- Duration: N/A
- Tier A attempts: N/A

## Files changed
- tt-xla: tests/runner/test_config/torch/test_config_inference_single_device.yaml

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 5b3bfba2f497ddd7ac642449bff13b4cf3623786 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
