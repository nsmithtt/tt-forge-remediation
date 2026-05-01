# Remediation Summary: mlx_community_qwen3_5_122b_a10b_5bit-causal_lm-pytorch-122B_A10B_5bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mlx_community_qwen3_5_122b_a10b_5bit/causal_lm/pytorch-122B_A10B_5bit-single_device-inference]

## Result
XFAIL — 122B param MoE VLM; 5-bit MLX safetensors ~84.5 GB exceeds p150b 32 GB DRAM

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
`mlx-community/Qwen3.5-122B-A10B-5bit` is a `Qwen3_5MoeForConditionalGeneration` VLM
with 122B total parameters (256 experts, 8 active, 48 layers). The HuggingFace repo
contains 17 safetensor shards with MLX 5-bit affine quantization: U32 tensors holding
~19B packed 5-bit weights and BF16 tensors for scales/biases/norms, totalling ~84.5 GB
of model data. This exceeds the p150b's 32 GB DRAM by 2.6×; when dequantized to BF16
the model is ~244 GB. The loader uses `AutoModelForCausalLM` which maps `qwen3_5_moe`
→ `Qwen3_5MoeForCausalLM`, causing a weight-key prefix mismatch (VLM checkpoint uses
`language_model.model.layers.*` while the CausalLM model expects `model.layers.*`),
which inflates all parameters to random BF16 → ~244 GB. The CI kills the test after
its configured timeout while attempting to download or load 84.5 GB of weight files
on a host with limited disk and RAM.

## Fix
Added `KNOWN_FAILURE_XFAIL` entry to
`tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`
with a reason explaining the hardware capacity ceiling.

## Verification
- pytest exit: not-run (model cannot be downloaded on this host: ~84.5 GB > 17 GB free disk)
- Hardware:    not-run
- Duration:    n/a
- Tier A attempts: N/A

## Files changed
- tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 7539acecc08974998ce159134bfc75f6e56325eb |
| tt-forge-models | e2aaa4b6bc39ac094a277fe435c1c8b7bc39fd54 |
