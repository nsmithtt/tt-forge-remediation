# Remediation Summary: ig1_qwen3_vl-pytorch-30b_a3b_instruct_nvfp4-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[ig1_qwen3_vl/pytorch-30b_a3b_instruct_nvfp4-single_device-inference]

## Result
XFAIL — ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4 has 30.7B parameters (60.8 GB BF16), which exceeds all single-device TT DRAM (n150: 12 GB, p150b: 24 GB)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-30b-moe-vlm-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

## Root cause
`ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4` is a 30.7B-parameter MoE Vision-Language
Model (Qwen3-VL-MoE, 48 layers, 128 experts, hidden=2048). Its weights use
`nvfp4-pack-quantized` compressed-tensors format (~15 GB at 4 bits/weight). TT
hardware has no native FP4 inference support, so execution requires dequantising
weights to BF16 at runtime, raising the effective model footprint to 60.8 GB.
This exceeds all single-device TT DRAM: n150 (12 GB) and p150b (24 GB). INTERNAL
Error code 13 is the OOM indicator from the TT runtime when it cannot allocate
device buffers.

A secondary loader bug was also present: `compressed-tensors` was not listed in
`requirements.txt`, so the quantisation config could not be instantiated and the
loader failed with ImportError before reaching silicon. This is fixed in the
tt_forge_models commit.

## Fix
1. **tt_forge_models** (`remediation/ig1_qwen3_vl-pytorch-30b_a3b_instruct_nvfp4-single_device-inference`):
   - `ig1_qwen3_vl/pytorch/requirements.txt` — added `compressed-tensors` dependency.

2. **tt-xla** (`remediation/ig1_qwen3_vl-pytorch-30b_a3b_instruct_nvfp4-single_device-inference`):
   - `tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added `KNOWN_FAILURE_XFAIL` entry with the hardware-capacity reason.

## Verification
- pytest exit: FAIL (not run on silicon — hardware-class XFAIL)
- Hardware:    p150b
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `ig1_qwen3_vl/pytorch/requirements.txt` (tt_forge_models — new file)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 23945585effe53da97c3547c23182c1df9c9ab06 |
| tt-forge-models | edc5a28fee12299f817a9531f345fff643b5a791 |
