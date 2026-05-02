# Remediation Summary: kaitchup_qwen_3_5_9b_autoround_nvfp4_linearattn_bf16-causal_lm-pytorch-9B_AutoRound_NVFP4_linearattn_BF16-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[kaitchup_qwen_3_5_9b_autoround_nvfp4_linearattn_bf16/causal_lm/pytorch-9B_AutoRound_NVFP4_linearattn_BF16-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
compressed-tensors-nvfp4-no-weight-key

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.7824903182638582. Required: pcc=0.99.

## Root cause
Three loader bugs, all in `tt_forge_models`:

1. **`load_shard_spec` AttributeError**: The Qwen3.5 hybrid architecture has two layer types — `full_attention` layers (with `self.self_attn`) and `linear_attention` GDA layers (with `self.linear_attn`). The original `load_shard_spec` unconditionally accessed `layer.self_attn`, crashing on GDA layers.

2. **`Qwen3_5DynamicCache` TypeError** (preemptive): `Qwen3_5DynamicCache` is not a subclass of `transformers.cache_utils.Cache`. The comparison evaluator calls `torch.equal()` on the cache object and raises `TypeError`. Fixed by disabling caching in inputs.

3. **NVFP4 weights never loaded — PCC=0.78** (root cause of the PCC failure): The kaitchup model stores weights exclusively in NVFP4 packed format. The safetensors shards contain `weight_packed` (uint8, two FP4 nibbles per byte), `weight_scale` (float8_e4m3fn, per-block), and `weight_global_scale` (float32) — but NO bare `weight` tensors. Without `compressed-tensors` awareness, `AutoModelForCausalLM.from_pretrained` silently leaves all quantized MLP and full-attention layers at their random initialization. The GDA (linear_attention) layers are unquantized and load correctly, which explains why PCC was ~0.78 rather than random noise.

## Fix
All fixes in `tt_forge_models`, on branch `remediation/kaitchup_qwen_3_5_9b_autoround_nvfp4_linearattn_bf16-causal_lm-pytorch-9B_AutoRound_NVFP4_linearattn_BF16-single_device-inference`:

- **`kaitchup_qwen_3_5_9b_autoround_nvfp4_linearattn_bf16/causal_lm/pytorch/loader.py`**:
  - `load_shard_spec`: Added `hasattr(layer, "self_attn")` guard with `elif hasattr(layer, "linear_attn")` branch that maps `in_proj_qkv.weight`, `in_proj_z.weight` → `("model","batch")` and `out_proj.weight` → `("batch","model")`.
  - `load_inputs`: Added `inputs["use_cache"] = False` to suppress `Qwen3_5DynamicCache` creation.
  - `load_model`: Added call to `_dequantize_nvfp4_weights()` after `from_pretrained`.
  - Added `_dequantize_nvfp4_weights()` static method: downloads safetensors via `snapshot_download`, reads `model.safetensors.index.json`, iterates all `weight_packed` keys per shard, calls `unpack_fp4_from_uint8(packed, m, n*2)` then `dequantize(x_q, scale, global_scale, dtype)` from `compressed_tensors`, and copies the BF16 result in-place into each `nn.Linear.weight`. Key prefix remapping: safetensors uses `model.language_model.layers.X.*` while Python model navigates from `model.model` (the `Qwen3_5TextModel`), so `model.language_model.` is stripped before traversal.
  - Added `import json` and `import os` at top.
- **`kaitchup_qwen_3_5_9b_autoround_nvfp4_linearattn_bf16/requirements.txt`**: Created with `compressed-tensors`.

Two commits:
- `14a07d5413` — Fix load_shard_spec hasattr guard and use_cache=False for Qwen3.5 hybrid
- `58914ecfc5` — Add NVFP4 dequantization and compressed-tensors requirement

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    3460.26s (0:57:40)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/kaitchup_qwen_3_5_9b_autoround_nvfp4_linearattn_bf16/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/kaitchup_qwen_3_5_9b_autoround_nvfp4_linearattn_bf16/requirements.txt` (new)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 36e045217cbda1d386e7e9fe1238491f547734a6 |
| tt-forge-models | 58914ecfc5955c424ed75a2ea80352f72f14411c |
