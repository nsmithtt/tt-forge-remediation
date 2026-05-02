# Remediation Summary: nemotron_3_super_120b_a12b_mlx_3_6bit-causal_lm-pytorch-3_Super_120B_A12B_MLX_3_6bit-tensor_parallel-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[nemotron_3_super_120b_a12b_mlx_3_6bit/causal_lm/pytorch-3_Super_120B_A12B_MLX_3_6bit-tensor_parallel-inference]

## Result
XFAIL — model weights (~54 GB at 3.6-bit quantization) exceed n300-llmbox DRAM capacity (24 GB)

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
`MoringLabs/Nemotron-3-Super-120B-A12B-MLX-3.6bit` is a 120B parameter NemotronH model (88 layers: Mamba SSM + MoE + Attention). At 3.6-bit MLX quantization, the weight storage is approximately 120B × 3.6 bits / 8 = ~54 GB. This exceeds the DRAM capacity of every tested single-device and 2-device configuration: n150 (12 GB), n300-llmbox (2 × 12 GB = 24 GB), and p150b (32 GB). The test timed out because loading and allocating a 54 GB model causes an OOM that manifests as a hang or extremely slow allocation failure.

This is the same hardware-class ceiling as the previously confirmed XFAIL for the 4.5-bit (67.5 GB) and 9-bit (135.83 GB) MLX variants of the same model family.

## Fix
Added `KNOWN_FAILURE_XFAIL` entry to `tests/runner/test_config/torch/test_config_inference_tensor_parallel.yaml` in the tt-xla repo (remediation branch `remediation/nemotron_3_super_120b_a12b_mlx_3_6bit-causal_lm-pytorch-3_Super_120B_A12B_MLX_3_6bit-tensor_parallel-inference`, commit cf29bc025).

## Verification
- pytest exit: FAIL (timeout — hardware not available to confirm XFAIL disposition)
- Hardware:    not-run
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- tt-xla/tests/runner/test_config/torch/test_config_inference_tensor_parallel.yaml

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | cf29bc0258299d1f1fe868150d7e23cdfc74e77e |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
