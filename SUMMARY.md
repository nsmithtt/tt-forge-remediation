# Remediation Summary: deepseek_r1_distill_llama_70b_8bit-causal_lm-pytorch-DeepSeek_R1_Distill_Llama_70B_8bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek_r1_distill_llama_70b_8bit/causal_lm/pytorch-DeepSeek_R1_Distill_Llama_70B_8bit-single_device-inference]

## Result
XFAIL — 70B model dequantized to BF16 is ~141 GB, exceeding single p150b DRAM (96 GB)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-70b-bf16-exceeds-p150b-dram

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
The model `mlx-community/DeepSeek-R1-Distill-Llama-70B-8bit` uses MLX affine 8-bit
quantization: weights are stored as `torch.uint32` (packed int8, 4 values per uint32),
with per-group `scales` and `biases` tensors. The checkpoint total size is ~75 GB
(int8 weights + float16 scales/biases).

To run on TT hardware, the weights must be in BF16. Dequantizing the 70.55B parameters
from int8 to BF16 yields ~141 GB of weight data, which exceeds the p150b DRAM ceiling
of 96 GB. The test timed out in CI either because the 75 GB download exceeded the
watchdog window, or because the dequantized model could not be allocated on device.

The loader has an additional bug: `AutoModelForCausalLM.from_pretrained` with
`ignore_mismatched_sizes=True` loads the uint32-packed weight tensors (shape mismatch
vs. the float model) without dequantization, meaning the model runs with garbage
weights even if it could somehow fit. However, fixing the loader dequantization does
not change the fundamental hardware-capacity verdict.

## Fix
Added `KNOWN_FAILURE_XFAIL` entry in
`tests/runner/test_config/torch/test_config_inference_single_device.yaml` in tt-xla.

## Verification
- pytest exit: not-run (model not cached; 75 GB download needed; BF16 model 141 GB > 96 GB DRAM)
- Hardware:    blackhole-p150b
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added KNOWN_FAILURE_XFAIL entry

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a029d9885a8ce08ccc20682d2ca874f9e854af99 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
