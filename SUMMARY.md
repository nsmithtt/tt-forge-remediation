# Remediation Summary: glyph/conditional_generation/pytorch-glyph-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[glyph/conditional_generation/pytorch-glyph-single_device-inference]

## Result
FAIL — Conv3D circular buffer allocation (1,747,968 B) exceeds L1 max (1,572,864 B); same `conv3d-patch-embed-l1-overflow` bug as Qwen3VL

## Stack layer
tt-metal

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
  TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=10,y=9)]
  grow to 1747968 B which is beyond max L1 size of 1572864 B

Backtrace points to: ttnn::Conv3dDeviceOperation → tt::runtime::ttnn::operations::conv::run

## Root cause
Two bugs were found and fixed in the loader layer:

1. **transformers 5.2.0 breaking change**: `Glm4vImageProcessor` now loads as fast
   processor by default; the original error message ("loaded as a fast processor by
   default") was actually a FutureWarning, not the root failure. Added `use_fast=False`
   to `AutoProcessor.from_pretrained`.

2. **TT device dispatch on metadata tensors**: `Glm4vVisionModel.rot_pos_emb` iterates
   `grid_thw` to call `torch.arange(h)`, where `h` is a 0-d TT-device tensor, triggering
   `INTERNAL: Error code: 13`. Similar `.tolist()` control-flow issues exist in
   `get_rope_index` (iterates `input_ids`) and `get_image_features` (splits on
   `image_grid_thw`). Class-level patches with `.cpu().tolist()` fix all four entry points.

After the loader fixes compile and the visual encoder's graph breaks are handled by dynamo,
the test reaches `Conv3dDeviceOperation` (GLM4V's patch embedding layer:
`nn.Conv3d(in_channels=3, out_channels=1152, kernel=[2,14,14], stride=[2,14,14])`).
`tt-metal` statically allocates CB memory for `cb_vol2col_tiled` and `cb_weight_tiled`
based on `K_t = C_in_padded × T × H_k × W_k / TILE_SIZE = 32×2×14×14/32 = 392 tiles`.
This yields ~784 KB + ~784 KB = 1.57 MB plus input/output CBs, totalling 1,747,968 B
against a 1,572,864 B L1 limit. The 175 KB overflow is independent of batch size or
number of patches — it is determined solely by the Conv3D kernel parameters.

Root cause is in the Conv3D kernel's CB allocation formula, which pads `C_in=3` to
`TILE_WIDTH=32`, dramatically over-inflating the vol2col and weight circular buffers.
This is the same `conv3d-patch-embed-l1-overflow` bug documented for Qwen3VL
(kernel=[2,16,16], 2.247 MB overflow) — the GLM4V variant is smaller (2×14×14 vs
2×16×16, ~11% over limit vs ~43% for Qwen3VL) but the root cause is identical.

## Fix
**Loader fixes (applied, in tt_forge_models):**

`glyph/conditional_generation/pytorch/loader.py`:
- Added `use_fast=False` to `AutoProcessor.from_pretrained` in `_load_processor`.
- Added `_patch_for_tt_device()` classmethod with four class-level patches:
  - `Glm4vVisionModel.rot_pos_emb`: reimplemented with `grid_thw.cpu().tolist()` to
    extract Python ints before `torch.arange()` calls.
  - `Glm4vVisionModel.forward`: wraps original, passes `grid_thw.cpu()` to avoid
    `cu_seqlens.tolist()` firing on TT-device tensor.
  - `Glm4vModel.get_image_features`: moves `image_grid_thw` to CPU.
  - `Glm4vModel.get_rope_index`: moves `input_ids`, `image_grid_thw`, `video_grid_thw`,
    `attention_mask` to CPU; returns position tensors back to original device.

**Proposed compiler fix (NOT applied — Tier B):**

In `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` (Conv3D lowering) or
`tt-metal/.../conv3d_program_factory.cpp`, reduce `C_in` padding from `TILE_WIDTH=32`
to the actual `C_in=3` when computing `K_t`. This requires coordinated changes across:
1. TTIRToTTNN.cpp (lowering pass that pads C_in)
2. conv3d.cpp default config (shard parameters)
3. prepare_conv3d_weights.cpp (weight layout)

This is the same fix documented for the Qwen3VL `conv3d-patch-embed-l1-overflow` bug.

## Tier B justification
Indicator: `more-than-3-files` / `cross-repo`

The fix requires coordinated changes across at least 3 files in tt-mlir and tt-metal
(Conv3D lowering in TTIRToTTNN.cpp, program factory CB allocation formula, weight
preparation). The same bug was previously diagnosed for Qwen3VL and classified Tier B
for the same reason.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    254.61s (0:04:14)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/glyph/conditional_generation/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 53b2f3cdd32d94eb287ab08d48fe5abd3e60e282 |
| tt-forge-models | fc87aaf36b97e9942f301d6c76cca3e105a4bc19 |
