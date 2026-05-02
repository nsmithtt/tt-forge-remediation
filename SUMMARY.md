# Remediation Summary: optimind-sft-causal-lm-pytorch-20b-single-device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[optimind_sft/causal_lm/pytorch-20B-single_device-inference]

## Result
XFAIL — hardware capacity ceiling: model weights (~32.9 GB) fill p150b DRAM leaving insufficient room for inference activation buffers

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-class-dram-capacity-weights-fill-device

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Out of Memory: Not enough space to allocate 1061683200 B DRAM buffer across 8 banks, where each bank needs to store 132710400 B, but bank size is 4273390016 B (allocated: 4111191104 B, free: 162198912 B, largest free block: 66355200 B)

## Root cause
The microsoft/OptiMind-SFT model's weights load to ~32.9 GB of the p150b's ~34.2 GB total DRAM (8 banks × 4.27 GB). During inference execution, the runtime attempts to allocate a 1.06 GB activation buffer but only ~1.3 GB total remains free across all 8 banks (~162 MB per bank). The model fills the device to capacity and leaves no headroom for inference activations. This is a hardware-class capacity ceiling — the model cannot run inference on a single p150b device.

## Fix
Marked `optimind_sft/causal_lm/pytorch-20B-single_device-inference` as `KNOWN_FAILURE_XFAIL` in `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`. Confirmed locally that pytest reports `1 xfailed` (195s wall-clock).

## Verification
- pytest exit: XFAIL (1 xfailed)
- Hardware:    blackhole-p150b
- Duration:    195.00s (0:03:15)
- Tier A attempts: N/A

## Files changed
- tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 46467ce060057d52103fbd2840e02f23da195d61 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
