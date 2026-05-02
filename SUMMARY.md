# Remediation Summary: model_007-pytorch-Model_007-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[model_007/pytorch-Model_007-single_device-inference]

## Result
XFAIL — pankajmathur/model_007 is a LLaMA-2-70B fine-tune (~69B params, ~138 GB BF16) which exceeds single-device DRAM on all supported hardware

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-class-model-exceeds-single-device-dram

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
`pankajmathur/model_007` is a fine-tune of LLaMA-2-70B (80 layers, hidden_size=8192,
intermediate_size=28672, num_key_value_heads=8). Parameter count is ~69B, requiring
~138 GB in BF16. The CI timeout is triggered because loading 15 safetensors shards
(129 GB cached) takes several minutes before device transfer is even attempted.
Even if loading completed, the model would OOM immediately on any single Tenstorrent
device (n150: 12 GB DRAM, p150b: 32 GB DRAM). This is a hardware capacity ceiling,
not a compiler bug.

## Fix
Added `model_007/pytorch-Model_007-single_device-inference: status: KNOWN_FAILURE_XFAIL`
to `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`
with a reason explaining the hardware capacity limitation. This causes pytest to mark
the test as xfail rather than timing it out, preventing CI noise.

## Verification
- pytest exit: XFAIL (xfailed, not failed)
- Hardware:    wormhole
- Duration:    178.92s
- Tier A attempts: N/A

## Files changed
- tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 5efca574e3a28225a99b6b930905d5567d17e0ab |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
