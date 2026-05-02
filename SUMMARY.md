# Remediation Summary: mlx_community_ministral_3_8b_instruct_2512_4bit-pytorch-Ministral-3-8B-Instruct-2512-4bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mlx_community_ministral_3_8b_instruct_2512_4bit/pytorch-Ministral-3-8B-Instruct-2512-4bit-single_device-inference]

## Result
FAIL — ttmlir-bf16-precision-pixtral-vision-encoder: PCC 0.5277 after loader and masked_scatter fixes

## Stack layer
loader, tt-xla

## Tier
B

## Bug fingerprint
ttmlir-bf16-precision-pixtral-vision-encoder

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure:
```
2026-04-23 23:56:27.799 | critical |          Always | TT_THROW: Fabric Router Sync: Timeout after 10000 ms. Device 2: Expected status 0xa2b2c2d2, got 0xa1b1c1d1 (assert.hpp:104)
```

After loader and Tier A compiler fixes, residual failure:
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.5277219405652328. Required: pcc=0.99.
```

## Root cause

Four bugs were found in sequence:

**Bug 1 (loader): load_shard_spec wrong model hierarchy.**
The loader referenced `model.language_model.layers` and `model.vision_tower.vision_model.encoder.layers`, but `Mistral3ForConditionalGeneration` nests under `model.model.*`. Vision layers also use `layer.attention.q_proj` (PixtralAttention) and `layer.feed_forward.{gate,up,down}_proj`, not `layer.self_attn.q_proj`/`layer.mlp.fc1`.

**Bug 2 (loader): split_sizes int64 BF16 precision.**
`Mistral3Model.get_image_features()` computed split_sizes on the TT device as `image_sizes // downsample_ratio`. TT promotes int64 to bfloat16 internally; bfloat16(2310) == 2320, causing `RuntimeError: split_with_sizes expects sum 2310, got [2320]`.

**Bug 3 (loader): generate_block_attention_mask in-place XLA mutation.**
`pixtral.modeling_pixtral.generate_block_attention_mask` does `causal_mask[start:end, start:end] = 0` in a for-loop on a TT tensor. Iterating TT tensors causes a Dynamo graph break; the in-place mutation then fails in `extract_compiled_graph → torch_xla.sync()` → INTERNAL Error code: 13.

**Bug 4 (tt-xla, Tier A, fixed): masked_scatter cumsum OOM.**
`aten.masked_scatter` decomposition in `decompositions.py` flattens mask to `[B×S×H] = 11.86M` and calls cumsum. TT-MLIR tiles the reduction axis to 1024, creating a 48.6 GB intermediate that triggers INTERNAL Error code: 13 / Fabric Router Sync Timeout in the original report.

**Bug 5 (tt-mlir, Tier B, unfixed): Pixtral ViT BF16 precision loss.**
After all four fixes, the model compiles and runs in 315 s but PCC = 0.5277 vs required 0.99. Root cause is TT-MLIR BF16 precision loss across the 24-layer Pixtral ViT with 9240-token SDPA (1176×1540 image). The vision encoder output is corrupted before reaching the text decoder.

## Fix

**Loader fixes** (tt_forge_models, `mlx_community_ministral_3_8b_instruct_2512_4bit/pytorch/loader.py`):

1. `load_shard_spec`: corrected model hierarchy to `model.model.language_model.layers` and `model.model.vision_tower.transformer.layers`, and corrected attribute names (`layer.attention.*`, `layer.feed_forward.*`).

2. `load_model → _patched_get_image_features`: replaces `Mistral3Model.get_image_features` on the instance to compute split_sizes on CPU in int64: `torch.as_tensor(image_sizes, dtype=torch.int64).cpu() // downsample_ratio`.

3. `load_model → _fixed_generate_block_attention_mask`: replaces `pixtral.modeling_pixtral.generate_block_attention_mask` globally with a functional version that returns `torch.zeros(...)` for single-image (all-zero mask = no masking = correct) and builds multi-image masks on CPU then moves to device.

**Tier A compiler fix** (tt-xla, `python_package/tt_torch/torch_overrides.py`):

`TorchFunctionOverride.__torch_function__` now intercepts `aten.masked_scatter` when `data.dim() == 3 and mask.shape == data.shape`. Replaces the decomposition (which creates a 48.6 GB cumsum intermediate) with a token-level float32 cumsum + one-hot matmul gather. Float32 preserves integer exactness up to 2^24, avoiding the TT int64→bfloat16 corruption for indices > 256.

**Proposed fix for residual PCC bug (Tier B):**
TT-MLIR's BF16 reduction handling in SDPA for the Pixtral ViT (24 layers, K-dim ~1152, seq_len ~9240) produces accumulated error that propagates through all 2896 text decoder positions. A fix would require either f32 accumulation in the SDPA attention score softmax, or a mixed-precision policy in the TTNN SDPA kernel for large K-dim configurations. This is cross-cutting across all Pixtral/Mistral3-based VLMs.

## Tier B justification

`cross-cutting` — fixing Pixtral ViT BF16 precision requires changes to TT-MLIR's SDPA lowering or TTNN kernel accumulation policy, affecting all Pixtral-based VLMs (Mistral3, Pixtral-12B family) and potentially all large-seq SDPA models.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    315.99s (0:05:15)
- Tier A attempts: 1

## Files changed
- `tt-xla/python_package/tt_torch/torch_overrides.py` — masked_scatter 3D guard with float32 one-hot matmul gather
- `tt-xla/third_party/tt_forge_models/mlx_community_ministral_3_8b_instruct_2512_4bit/pytorch/loader.py` — three loader fixes (load_shard_spec paths, split_sizes CPU int64, generate_block_attention_mask)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 21131ad299469c4d0e6ed84eeaf87760f99c634d |
| tt-forge-models | bc35ad8beed20126e6d82f88cbd946acc41bf831 |
