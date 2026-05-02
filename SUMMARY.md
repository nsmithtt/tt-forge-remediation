# Remediation Summary: qwen_3_5_122b_a10b_nvfp4-causal_lm-pytorch-122B_A10B_NVFP4-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[qwen_3_5_122b_a10b_nvfp4/causal_lm/pytorch-122B_A10B_NVFP4-single_device-inference]

## Result
XFAIL — Model weight files total 76.9 GB (NVFP4 packed + BF16 scales), exceeding single-device DRAM capacity on all available hardware

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-class-model-exceeds-dram-capacity

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
The `AxionML/Qwen3.5-122B-A10B-NVFP4` model is a 122B-parameter hybrid SSM+attention VLM with 256 MoE experts (48 layers). The model index reports `total_size: 76,888,604,032 bytes` (~71.6 GB) for the NVFP4-packed safetensors weight files. This exceeds the DRAM of all available single-device hardware (p150b: 32 GB, n150: 12 GB). The test timeout results from the loading attempt running into OOM or the sheer time required to download 71.6 GB of weight files.

A secondary loader bug was also found and fixed: `ignore_mismatched_sizes=True` was missing from the `load_model` call. Without `nvidia-modelopt` installed, NVFP4 packed weight shapes differ from the model definition and transformers raises a shape mismatch error unless this flag is set. The fix mirrors the pattern used in the TinyLlama 1.1B Chat NVFP4 loader.

## Fix
- Added `ignore_mismatched_sizes=True` to `load_model` in `tt_forge_models/qwen_3_5_122b_a10b_nvfp4/causal_lm/pytorch/loader.py` (loader bug fix)
- Added `KNOWN_FAILURE_XFAIL` status to `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` for the test entry (hardware-class XFAIL)

## Verification
- pytest exit: not-run
- Hardware:    not-run
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/qwen_3_5_122b_a10b_nvfp4/causal_lm/pytorch/loader.py` — added `ignore_mismatched_sizes=True`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added `KNOWN_FAILURE_XFAIL`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | cdc1be33d46c7e6b17ec21094973401301797a4a |
| tt-forge-models | f4061e666367149fd988c5c27da92dbb9065a104 |
