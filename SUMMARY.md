# Remediation Summary: fara-pytorch-7B-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[fara/pytorch-7B-single_device-inference]

## Result
FAIL — Conv3d patch_embed L1 overflow: Qwen2.5-VL Conv3d(C_in=3, C_out=1280, kernel=[2,14,14]) statically allocates 1,745,920 B of circular buffers, exceeding max L1 of 1,572,864 B; Tier B compiler bug

## Stack layer
loader, tt-metal

## Tier
B

## Bug fingerprint
conv3d-patch-embed-l1-overflow

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

Underlying hardware error (logged before the Python exception):
```
TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=10,y=9)] grow to 1745920 B which is beyond max L1 size of 1572864 B
```

## Root cause
Two loader bugs were present and fixed before the compiler-level root cause became visible:

1. **transformers 5.x `use_cache` kwarg** — `from_pretrained` in transformers 5.2.0 passes all remaining kwargs directly to the model constructor (`cls(config, **model_kwargs)`). `Qwen2_5_VLForConditionalGeneration.__init__()` only accepts `config`, so `use_cache=False` raised `TypeError`. Fix: set `model.config.text_config.use_cache = False` after `from_pretrained`.

2. **Qwen2.5-VL `.tolist()` on TT device tensors** — Five methods in `modeling_qwen2_5_vl` call `.tolist()` or perform control-flow on device tensors (`rot_pos_emb`, `get_window_index`, `get_rope_index`, `get_image_features`, `Qwen2_5_VLVisionAttention.forward`). TT device does not support eager tensor reads; these calls surface as `INTERNAL: Error code: 13`. Patched all five methods to move metadata tensors to CPU before the `.tolist()` calls.

After both loader fixes, the test fails in `tt-metal` with the Conv3d L1 overflow. The `Qwen2_5_VLPatchEmbed` uses `Conv3d(in_channels=3, out_channels=1280, kernel_size=[2,14,14])`. The tt-mlir `Conv3dOpConversionPattern` pads `C_in=3` to 32 (the tile width), producing `K=2×14×14×32=12544` elements and `K_tiles=392`. The `conv3d_program_factory` allocates `cb_vol2col_tiled` and `cb_weight_tiled` based on `K_tiles`, totalling ~1.57–1.75 MB, which exceeds the 1.5 MB L1 per core.

The Tier A fix (capping `c_in_block` in `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`) was previously applied for Qwen3-VL (commit `151e2000ce`) but exposed a secondary tt-metal bug: `CB page size 192 should be greater than the config tensor page size 392` (`conv3d-cb-page-size-vs-tensor-mismatch`). Until that secondary bug is resolved, no Tier A fix is feasible for any Qwen2.5-VL or Qwen3-VL model.

## Fix
Loader fixes committed to `tt-forge-models` on branch `remediation/fara-pytorch-7B-single_device-inference`:

- `fara/pytorch/loader.py`: Added `_patch_qwen2_5_vl_for_tt_device()` function that patches five `modeling_qwen2_5_vl` methods to move grid_thw/input_ids to CPU before `.tolist()` calls. Removed `use_cache=False` from `model_kwargs`; set `model.config.text_config.use_cache = False` post-load.

Proposed fix for the compiler-stack root cause (Conv3d L1 overflow):
- In `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` `Conv3dOpConversionPattern::matchAndRewrite`: cap `c_in_block` to `MAX_CB_TILES * TILE_WIDTH / (D_k * H_k * W_k)` to keep vol2col and weight CBs within L1. But this fix must be coupled with a fix to the CB page size calculation in `tt-metal/ttnn/cpp/ttnn/operations/conv/conv3d/device/conv3d_op_program_factory.cpp` to avoid `conv3d-cb-page-size-vs-tensor-mismatch`.

## Tier B justification
Indicator: **cross-repo** (requires coordinated changes in `tt-mlir` Conv3d lowering and `tt-metal` Conv3d kernel CB page size calculation). The Tier A fix for the L1 overflow has already been attempted on Qwen3-VL and exposed a secondary bug in tt-metal. The secondary bug (`conv3d-cb-page-size-vs-tensor-mismatch`) must be fixed in tt-metal first.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    86.35s
- Tier A attempts: N/A

## Files changed
- `tt-forge-models/fara/pytorch/loader.py` — transformers 5.x use_cache fix + Qwen2.5-VL .tolist() patches

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 1a8f51e89e17faf2d99df525373c00dcaa33cc4d |
| tt-forge-models | eede5bd83b007520c7f4ef1125e347b60ba741c5 |
