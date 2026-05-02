# Remediation Summary: magistral_small_mlx-causal_lm-pytorch-Small_2509_MLX_5bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[magistral_small_mlx/causal_lm/pytorch-Small_2509_MLX_5bit-single_device-inference]

## Result
XFAIL — 24B BF16 model (~48 GB) exceeds single-device p150b DRAM (32 GB); hardware-class OOM

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
mlx-5bit-mistral3-loader-rewrite

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
RuntimeError: TT_FATAL @ tt_metal/impl/allocator/bank_manager.cpp:439: false
info:
Out of Memory: Not enough space to allocate 335544320 B DRAM buffer across 8 banks,
where each bank needs to store 41943040 B, but bank size is 4273390016 B
(allocated: 4186371136 B, free: 87018880 B, largest free block: 37079488 B)

Original reported failure: sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
(This is a trailing Python warning, not the actual root cause.)

## Root cause
Two distinct issues:

**Loader bugs (fixed):**
1. `lmstudio-community/Magistral-Small-2509-MLX-5bit` has `model_type=mistral3`
   (`Mistral3Config`), which is not in `AutoModelForCausalLM`'s registry. The original
   loader called `AutoModelForCausalLM.from_pretrained()` which raises
   `ValueError: Unrecognized configuration class Mistral3Config`. Fix: use
   `Mistral3ForConditionalGeneration` directly with `quantization_config=None` and
   implement MLX affine 5-bit dequantization manually (uint32-packed, LSB-first,
   per-group bf16 scales+biases, group_size=64).

2. MLX stores Conv2d weights as `[out, H, W, in]` (channel-last) but PyTorch expects
   `[out, in, H, W]`. The vision tower's `patch_conv.weight` failed with size mismatch
   `[1024, 14, 14, 3]` vs model's `[1024, 3, 14, 14]`. Fix: permute 4D float weights
   with `.permute(0, 3, 1, 2).contiguous()` during dequantization.

**Hardware-class ceiling:**
After the loader was fixed, the model compiled and began executing on TT silicon but
OOM'd during DRAM allocation. The 24B parameter model at BF16 precision requires ~48 GB
of DRAM, which exceeds the p150b's ~32 GB (8 banks × ~4.27 GB). This is a genuine
hardware capacity ceiling, not a compiler bug.

## Fix
- **tt_forge_models** (`remediation/magistral_small_mlx-causal_lm-pytorch-Small_2509_MLX_5bit-single_device-inference`):
  - Commit `13f2512734`: Rewrote `loader.py` — replaced `AutoModelForCausalLM` with
    `Mistral3ForConditionalGeneration`, added MLX 5-bit affine dequantization
    (`_unpack_mlx_5bit`, `_dequantize_shard`), key remapping (`_remap_key`), and
    `list_repo_files()` for shard discovery (stale index file references 10 shards
    but repo has 4). Also created `requirements.txt` with `safetensors>=0.4.0` and
    `huggingface_hub>=0.23.0`.
  - Commit `e9e1ac5913`: Added Conv2d channel-last permutation in `_dequantize_shard`
    for 4D (and 5D) floating-point weight tensors.
  - File: `third_party/tt_forge_models/magistral_small_mlx/causal_lm/pytorch/loader.py`
  - File: `third_party/tt_forge_models/magistral_small_mlx/causal_lm/pytorch/requirements.txt`

- **tt-xla** (`remediation/magistral_small_mlx-causal_lm-pytorch-Small_2509_MLX_5bit-single_device-inference`):
  - Commit `ade86cef3`: Added `KNOWN_FAILURE_XFAIL` entry to
    `tests/runner/test_config/torch/test_config_inference_single_device.yaml` with OOM
    reason and hardware context.

## Verification
- pytest exit: FAIL (OOM — expected hardware-class ceiling)
- Hardware:    p150b (wormhole)
- Duration:    2718.51s (0:45:18)
- Tier A attempts: N/A

## Files changed
- `third_party/tt_forge_models/magistral_small_mlx/causal_lm/pytorch/loader.py` (rewritten)
- `third_party/tt_forge_models/magistral_small_mlx/causal_lm/pytorch/requirements.txt` (created)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (KNOWN_FAILURE_XFAIL entry added)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 3e4e73b33386d3a87e33bad713194da6bcffe197 |
| tt-forge-models | e9e1ac5913824846b869ee10460010d0d06835ee |
