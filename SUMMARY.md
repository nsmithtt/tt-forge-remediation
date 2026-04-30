# Remediation Summary: inferencerlabs_nvidia_nemotron_3_super_120b_a12b_mlx_9bit-causal_lm-pytorch-NVIDIA-Nemotron-3-Super-120B-A12B-MLX-9bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[inferencerlabs_nvidia_nemotron_3_super_120b_a12b_mlx_9bit/causal_lm/pytorch-NVIDIA-Nemotron-3-Super-120B-A12B-MLX-9bit-single_device-inference]

## Result
XFAIL — NemotronH 120B hybrid-Mamba/MoE/Attention model at MLX 9-bit quantization weighs 135.83 GB, far exceeding single-device DRAM capacity (max 32 GB on p150b Blackhole)

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
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

## Root cause
The `inferencerlabs/NVIDIA-Nemotron-3-Super-120B-A12B-MLX-9bit` repository ships
14 safetensors shards totalling **135.83 GB** of 9-bit MLX-quantized weights for the
NemotronH architecture (88 layers: 40 Mamba SSM + 40 MoE + 8 Attention; 512
routed experts, 22 active per token).  The maximum single-device DRAM is 32 GB
(p150b Blackhole), which is less than 1/4 of the model size.  The PJRT layer
surfaces the resulting out-of-memory condition as `INTERNAL: Error code: 13`.

The same error and disposition applies to the sibling 4.5-bit variant
(`inferencerlabs/NVIDIA-Nemotron-3-Super-120B-A12B-MLX-4.5bit`, ~67.5 GB), and
the GGUF variant (`AesSedai/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF` at Q4_K_M,
~67.5 GB) was already XFAIL'd in a prior report for the same reason.

## Fix
Added `KNOWN_FAILURE_XFAIL` entry to
`tests/runner/test_config/torch/test_config_inference_single_device.yaml` in
`tt-xla` on branch
`remediation/inferencerlabs_nvidia_nemotron_3_super_120b_a12b_mlx_9bit-causal_lm-pytorch-NVIDIA-Nemotron-3-Super-120B-A12B-MLX-9bit-single_device-inference`
(commit `3bbaaf6ae67e145966b82536400a6f54e6c2f2e0`).

## Verification
- pytest exit: FAIL (not run on silicon — model too large to download/run; hardware ceiling confirmed by file-size enumeration)
- Hardware:    not-run
- Duration:    n/a
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 3bbaaf6ae67e145966b82536400a6f54e6c2f2e0 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
