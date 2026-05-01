# Remediation Summary: huihui_qwen3_5_27b_claude_4_6_opus_abliterated_4bit-causal_lm-pytorch-27B_Claude_4.6_Opus_Abliterated_4bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[huihui_qwen3_5_27b_claude_4_6_opus_abliterated_4bit/causal_lm/pytorch-27B_Claude_4.6_Opus_Abliterated_4bit-single_device-inference]

## Result
XFAIL — Qwen3.5-27B VLM with MLX 4-bit affine quantization: ~53 GB BF16 exceeds p150b 32 GB DRAM

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-qwen35-27b-mlx-affine-bf16-overflow

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
`mlx-community/Huihui-Qwen3.5-27B-Claude-4.6-Opus-abliterated-4bit` is a `Qwen3_5ForConditionalGeneration`
VLM whose safetensors use MLX 4-bit affine quantization: weights are stored as uint32-packed 4-bit integers
with shape `[out, in//8]` plus separate BF16 `scales` and `biases` tensors. The checkpoint key prefix is
`language_model.model.layers.*` (VLM layout), but `AutoModelForCausalLM` maps `Qwen3_5Config` to
`Qwen3_5ForCausalLM` (text-only), whose expected keys are `model.layers.*`. This complete key mismatch
causes every weight in the model to be randomly initialized as BF16.

The resulting randomly-initialized model has ~27 billion BF16 parameters (~53 GB), far exceeding the 32 GB
DRAM on the p150b device. The test hangs while reading 15 GB of safetensors files and initializing ~53 GB
of random weights, eventually being killed by the CI timeout watchdog.

Even if the key mismatch were fixed (loading as `Qwen3_5ForConditionalGeneration`), the weights are stored
in MLX affine uint32-packed format which transformers cannot dequantize. Loading as BF16 with
`ignore_mismatched_sizes=True` still results in ~53 GB random weights, and TT silicon has no native INT4
execution path to use the quantized format directly.

## Fix
Added `KNOWN_FAILURE_XFAIL` entry to:
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

The reason field documents both the MLX quantization incompatibility and the BF16 size overflow.

## Verification
- pytest exit: TIMEOUT (test killed by shell timeout during reproduction; confirms hang)
- Hardware:    blackhole-p150b
- Duration:    >120s (killed by timeout before completion)
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added KNOWN_FAILURE_XFAIL entry

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | d934b0800fca0106ce8834cb4f909aa5d8452ed2 |
| tt-forge-models | 0f7b734348fee3e3fa9c17e8e65fe7bc6c35b80c |
