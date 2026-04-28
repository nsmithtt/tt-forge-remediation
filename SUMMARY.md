# Remediation Summary: glyph/conditional_generation/pytorch-glyph-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[glyph/conditional_generation/pytorch-glyph-single_device-inference]

## Result
XFAIL — GLM4V high-resolution SDPA requires ~75 GB DRAM (24K image patches × 24K sequence full-attention), exceeding single-device capacity on n150

## Stack layer
loader, tt-mlir, hardware-class

## Tier
A

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
Original failure (first run):
  TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=10,y=9)]
  grow to 1747968 B which is beyond max L1 size of 1572864 B

After Conv3d Tier A fix (second and third runs):
  TT_FATAL: Out of Memory: Not enough space to allocate 75874959360 B DRAM buffer
  across 8 banks, where each bank needs to store 9484369920 B, but bank size is
  4273390016 B (allocated: 596017536 B, free: 3677372480 B)
  → surfaces as: RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

## Root cause
Three issues found in total:

**1. transformers 5.x fast-processor breaking change (loader):**
`Glm4vImageProcessor` now loads as fast processor by default. The original reported
failure message was this FutureWarning. Fixed with `use_fast=False` in
`AutoProcessor.from_pretrained`.

**2. TT device dispatch on metadata tensors (loader):**
`Glm4vVisionModel.rot_pos_emb` iterates `grid_thw` to call `torch.arange(h)`, where
`h` is a 0-d TT-device tensor, triggering `INTERNAL: Error code: 13`. Similar
`.tolist()` control-flow issues exist in `get_rope_index` and `get_image_features`.
Fixed with class-level `.cpu().tolist()` patches.

**3. Conv3d L1 CB overflow (tt-mlir, Tier A — FIXED):**
`Glm4vVisionPatchEmbed` uses `nn.Conv3d(in_channels=3, out_channels=1536, kernel=[2,14,14])`.
The MLIR lowering used `C_in_block = TILE_WIDTH = 32`, giving `K_t = 32×2×14×14/32 = 392`
tiles for `cb_vol2col_tiled` and `cb_weight_tiled`. Total CB allocation ≈ 1.57 MB,
exceeding the 1.5 MB L1 limit by ~175 KB. Fixed in `TTIRToTTNN.cpp` by computing
`C_in_block = 16` when `kernelVol = 2×14×14 = 392 > L1_K_TILES = 384`, halving K_t to 196
and CBs to ~784 KB.

**4. DRAM OOM — hardware capacity ceiling (XFAIL):**
The test image (candy.JPG, 4032×3024 pixels) is processed by GLM4V's image processor
into `grid_thw = [1, 134, 180]` = 24,120 image patches regardless of whether the fast or
slow processor is used. These 24,120 image tokens yield a total sequence length of ~30K
tokens. TTNN's full-attention SDPA allocates the full attention matrix:
`24160 × 24160 × 32 heads × 2 bytes ≈ 37–75 GB` (confirmed 75,874,959,360 B allocated).
The n150 device has ~34 GB total DRAM. This is a genuine hardware capacity ceiling, not
a compiler bug.

## Fix
**Loader fixes (applied, committed to tt_forge_models remediation branch):**

`glyph/conditional_generation/pytorch/loader.py`:
- Added `use_fast=False` to `AutoProcessor.from_pretrained` in `_load_processor`.
- Added `_patch_for_tt_device()` classmethod with four class-level patches:
  - `Glm4vVisionModel.rot_pos_emb`: reimplemented with `grid_thw.cpu().tolist()` to
    extract Python ints before `torch.arange()` calls.
  - `Glm4vVisionModel.forward`: passes `grid_thw.cpu()` to avoid `cu_seqlens.tolist()`
    on TT-device tensor.
  - `Glm4vModel.get_image_features`: moves `image_grid_thw` to CPU for split control flow.
  - `Glm4vModel.get_rope_index`: moves `input_ids`, `image_grid_thw`, `video_grid_thw`,
    `attention_mask` to CPU; returns position tensors back to original device.

**Compiler fix (applied, Tier A, committed to tt-mlir remediation branch):**

`tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` (Conv3dOpConversionPattern):
- Added `kernelVol = kD × kH × kW` computation.
- When `kernelVol > L1_K_TILES (384)`, set `cInBlock = MIN_C_IN_BLOCK (16)` instead of
  `TILE_WIDTH (32)` to keep L1 CB usage within the 1.5 MB limit.
- Passes explicit `Conv3dConfigAttr` with the computed `c_in_block` to TTNN.

**XFAIL config (applied, committed to tt-xla remediation branch):**

`tests/runner/test_config/torch/test_config_inference_single_device.yaml`:
- Added `glyph/conditional_generation/pytorch-glyph-single_device-inference` with
  `status: KNOWN_FAILURE_XFAIL` explaining the 75 GB DRAM ceiling.

## Verification
- pytest exit: FAIL (DRAM OOM after Conv3d fix; test config updated to XFAIL)
- Hardware:    n150
- Duration:    244.86s (0:04:04) — third run with both loader and Conv3d fixes applied
- Tier A attempts: 1 (Conv3d c_in_block fix; successfully resolved L1 overflow; DRAM OOM is hardware-class)

## Files changed
- `tt-xla/third_party/tt_forge_models/glyph/conditional_generation/pytorch/loader.py`
- `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 762e48d62918f9b50bbe165f2971010f3b4d1c49 |
| tt-xla          | fe0726af6856bb9daa52f0699f4d93a46a5d8172 |
| tt-forge-models | fc87aaf36b97e9942f301d6c76cca3e105a4bc19 |
