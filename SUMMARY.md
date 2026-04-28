# Remediation Summary: longcat_video_avatar_comfyui_gguf-pytorch-Single_Q8_0-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[longcat_video_avatar_comfyui_gguf/pytorch-Single_Q8_0-single_device-inference]

## Result
XFAIL — model has 16.8B params; bfloat16 dequantization at load time requires ~33.6 GB, exceeding all single-device DRAM (n150: 12 GB, p150b: 24 GB)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-16b-model-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
2026-04-19 18:18:23.441 | critical |          Always | TT_THROW: TIMEOUT: device timeout in fetch queue wait, potential hang detected (assert.hpp:104)

## Root cause
The LongCat-Video-Avatar model is a 16.8B-parameter WAN-based video transformer.
The loader passes `torch_dtype=bfloat16` to `WanTransformer3DModel.from_single_file`,
which dequantizes the GGUF weights from Q8_0 to bfloat16 at load time. The resulting
model occupies ~33.6 GB (16.8B × 2 bytes), which exceeds all single TT device DRAM:
n150 (12 GB) and p150b/Blackhole (24 GB). The original CI failure (TIMEOUT: device
timeout in fetch queue wait) is consistent with the device running out of DRAM during
tensor allocation, causing a hang rather than a clean OOM error.

Secondary loader bug (not the binding constraint): diffusers' `infer_diffusers_model_type`
checks for the exact key `"head.modulation"` in the GGUF checkpoint, but the GGUF stores
more specific sub-keys (`"head.modulation.1.bias"`, `"head.modulation.1.weight"`). The
detection falls through to the `"v1"` fallback, causing `cls.load_config` to attempt
fetching config from `stable-diffusion-v1-5/stable-diffusion-v1-5/transformer/config.json`,
which returns a 404 in current environments. This secondary bug prevents reproduction
in the current environment but is not the binding constraint.

## Fix
Added `KNOWN_FAILURE_XFAIL` entry for
`longcat_video_avatar_comfyui_gguf/pytorch-Single_Q8_0-single_device-inference` in
`tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`.

## Verification
- pytest exit: not-run (XFAIL — would OOM before silicon execution)
- Hardware:    not-run
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 954a515cdfbd49170d7f4f7f3794cf317f18bf8b |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
