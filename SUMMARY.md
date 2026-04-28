# Remediation Summary: gliese_qwen3_5/image_to_text/pytorch-0_8b_abliterated_caption-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gliese_qwen3_5/image_to_text/pytorch-0_8b_abliterated_caption-single_device-inference]

## Result
FAIL — dynamic-shape boolean indexing in `get_placeholder_mask` blocked silicon pass after Conv3d L1 fix

## Stack layer
loader, tt-mlir

## Tier
A

## Bug fingerprint
dynamic-shape-boolean-index-embedding-scatter

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

Original failure in `fast_pos_embed_interpolate` (tolist on TT device tensor). After fixing two loader bugs (tolist patch + pixel limits + device_map removal) and the Conv3d L1 overflow in tt-mlir, the test ran 452s before failing at `get_placeholder_mask` with the same error code, now from dynamic-shape boolean indexing.

## Root cause

Three bugs were found in sequence:

**Bug 1 (fixed): Loader — tolist() on TT device tensors**

`Qwen3_5VisionModel.fast_pos_embed_interpolate`, `rot_pos_emb`, and `Qwen3_5Model.get_rope_index` call `.tolist()` on `grid_thw`/`input_ids` tensors for Python control flow. After the test harness moves these to TT device, `.tolist()` triggers a device sync that fails with `Error code: 13`. Fix: patch all four methods (`fast_pos_embed_interpolate`, `rot_pos_emb`, `get_rope_index`, `get_image_features`) to move `grid_thw`/`input_ids` to CPU before the `.tolist()` calls. Return `position_ids` and `rope_deltas` back to the original device after `get_rope_index`.

Also: the loader used `device_map="auto"` in `from_pretrained()`, which loads model parameters in dispatch mode. This conflicts with how TT XLA handles dynamo-compiled functions — the `.cpu()` calls inside compiled regions trigger `torch_xla.sync()` which fails on a device whose graph was left in a bad state by the dispatch mode. Removing `device_map="auto"` resolves this.

Also: processor had no pixel limits, causing the demo JPEG (1376×2048) to produce excessive patches. Set `min_pixels=56*56, max_pixels=13*28*1280`.

**Bug 2 (fixed): Tier A — Conv3d L1 overflow in `tt-mlir`**

`Qwen3_5VisionPatchEmbed` uses `nn.Conv3d(in_channels=3, out_channels=768, kernel=[2,16,16])`. The `Conv3dOpConversionPattern` in `TTIRToTTNN.cpp` was constructing a `Conv3dConfigAttr` with no blocking parameters, causing `conv3d_program_factory.cpp` to allocate `cb_vol2col_tiled + cb_weight_tiled = 2.1 MB` of circular buffers per core, exceeding the L1 limit (~1.5 MB). The fix computes a safe `c_in_block` (16 for 2×16×16 kernel) and explicitly sets `c_out_block=TILE_WIDTH=32` plus the full device worker grid.

**Bug 3 (not fixed): Dynamic-shape boolean indexing in `get_placeholder_mask`**

After the vision encoder ran successfully, `Qwen3_5Model.get_placeholder_mask` executes:
```python
special_image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
torch_compilable_check(inputs_embeds[special_image_mask].numel() == image_features.numel(), ...)
```
`inputs_embeds[special_image_mask]` is a boolean-masked gather with dynamic output shape (depends on number of image tokens). TT device compilation requires static shapes; this triggers `Error code: 13` via `torch_xla.sync()` in `extract_graph_helper`. Per skill rules, after the first compiler-stack fix (Conv3d), the second compiler-stack bug is filed as FAIL.

## Fix

**Bug 1 fix** (loader):

`tt-xla/third_party/tt_forge_models/gliese_qwen3_5/image_to_text/pytorch/loader.py`:
- Added `_patch_qwen3_5_for_tt_device()` patching 4 methods to move `grid_thw`/`input_ids` metadata to CPU before `.tolist()` calls
- Removed `device_map="auto"` from `from_pretrained()` call
- Added `min_pixels=56*56, max_pixels=13*28*1280` pixel limits to processor

**Bug 2 fix** (tt-mlir, Tier A):

`tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` — `Conv3dOpConversionPattern::matchAndRewrite`:
Compute safe `c_in_block = max(TILE_WIDTH, MAX_CB_TILES * TILE_WIDTH / (D_k * H_k * W_k))` where `MAX_CB_TILES=256` (512 KB). For kernel=[2,16,16]: `c_in_block=16`. Set `c_out_block=TILE_WIDTH=32` and `compute_with_storage_grid_size` to the full device worker grid.

**Bug 3 proposed fix**:

In `tt-xla` or `tt-mlir`: add dynamic-shape indexing support for `tensor[bool_mask]` patterns, or add a graph-break/CPU-fallback guard for data-dependent tensor sizes in `get_placeholder_mask`. A loader-layer workaround (replacing the boolean-index with a static `torch.where` scatter) is feasible but falls outside permitted fixes per skill rules.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
N/A — Tier A was completed. Second bug is the Tier B stop.

## Verification
- pytest exit: FAIL
- Hardware: n150
- Duration: 452.00s (0:07:32)
- Tier A attempts: 1

## Files changed
- `tt-xla/third_party/tt_forge_models/gliese_qwen3_5/image_to_text/pytorch/loader.py` — tolist patch, remove device_map=auto, pixel limits
- `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` — Conv3d L1-safe c_in_block computation

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 151e2000ce185c9cd344c4934963b03c838f91a2 |
| tt-xla          | e2fa9a4fbbc5f6c5c81fe125d7573fd670ce1d16 |
| tt-forge-models | 23f02e02e4e374209f9da84f71f22ca846497279 |
