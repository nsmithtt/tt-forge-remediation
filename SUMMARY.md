# Remediation Summary: dolphin_v2-image_to_text-pytorch-dolphin_v2-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[dolphin_v2/image_to_text/pytorch-dolphin_v2-single_device-inference]

## Result
FAIL — Conv3d patch embedding exceeds L1 circular buffer allocation on tt-metal (same Tier B bug as Qwen3-VL)

## Stack layer
tt-metal

## Tier
B

## Bug fingerprint
conv3d-l1-cb-overflow

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The image processor of type `Qwen2VLImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.

After fixing the loader (use_fast=False + tolist() patches), the test fails with:

    RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

    Statically allocated circular buffers on core range [(x=0,y=0) - (x=10,y=9)]
    grow to 1745920 B which is beyond max L1 size of 1572864 B

at `ttnn.experimental.Conv3dDeviceOperation` during graph compilation.

## Root cause
Two-layer failure:

**Layer 1 (loader, fixed):** transformers 5.x loads `Qwen2VLImageProcessor` as a fast
processor by default — a breaking change. Fixed by adding `use_fast=False` to
`AutoProcessor.from_pretrained`. Additionally, `Qwen2_5_VisionTransformerPretrainedModel.rot_pos_emb`,
`get_window_index`, `Qwen2_5_VLModel.get_image_features`, and `get_rope_index` all call
`.tolist()` on tensors that the test runner has placed on TT device. These are patched to
move the relevant metadata tensors to CPU before the `.tolist()` calls.

**Layer 2 (tt-metal, blocking):** `Qwen2_5_VisionPatchEmbed` uses
`nn.Conv3d(in_channels=3, out_channels=1280, kernel=[2,14,14], stride=[2,14,14])`.
The tt-mlir Conv3d lowering pads `C_in=3` to `TILE_WIDTH=32`, making the circular
buffer allocation for `cb_vol2col_tiled` and `cb_weight_tiled` exceed the L1 limit:
1,745,920 B > 1,572,864 B (max L1). This is identical to the Qwen3-VL
`conv3d-l1-cb-overflow` bug, just with [2,14,14] instead of [2,16,16] kernels.

When the model is compiled by torch.compile, `patch_embed` (Conv3d) is the first
sub-graph dispatched to the TT device. The graph sync that triggers compilation
(either directly from `.tolist()` or from a graph break on `.cpu()`) causes
Conv3d to allocate circular buffers, which exceed L1 and throw Error code: 13.

## Fix
**Proposed fix (tt-metal):** Same as the Qwen3-VL fix: adjust
`ttnn/cpp/ttnn/operations/experimental/conv3d/conv3d_op_program_factory.cpp`
(and related files) to tile the convolution along the K dimension so circular
buffers fit within L1, rather than allocating all weight tiles at once.
Specifically, `conv3d_program_factory.cpp` allocates `cb_vol2col_tiled` and
`cb_weight_tiled` as `C_in_padded × D × H × W × 2048 B` each; tiling K would
reduce peak L1 usage to fit within the 1,572,864 B budget. This requires
coordinated changes across `TTIRToTTNN.cpp`, `conv3d.cpp` default config, and
`prepare_conv3d_weights.cpp` in tt-mlir/tt-metal.

**Loader fixes committed (tt_forge_models):**
- `use_fast=False` in `AutoProcessor.from_pretrained`
- `rot_pos_emb`, `get_window_index` patched to move `grid_thw` to CPU
- `get_image_features` patched to move `image_grid_thw` to CPU before `.tolist()`
- `get_rope_index` patched to move `input_ids`, `image_grid_thw`, `video_grid_thw`,
  `attention_mask` to CPU; returns `position_ids` and `rope_deltas` back to original device

## Tier B justification
**Indicator:** cross-cutting (more-than-3-files)

The Conv3d L1 overflow fix requires coordinated changes across `TTIRToTTNN.cpp`
(padding strategy), `conv3d.cpp` (default tiling config), and `prepare_conv3d_weights.cpp`
(weight layout). This is the same Tier B bug as `conv3d-l1-cb-overflow` reported for
Qwen3-VL models.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    85.58s
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/dolphin_v2/image_to_text/pytorch/loader.py`
  — added `use_fast=False`, `_patch_qwen2_5_vl_tolist()` with four method patches

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 16f80cac9d9c69f9f30e7e97b7f4a14269b3573c |
| tt-forge-models | 468f7d9fcfe11455c28a2b1c6a817c0b18762a84 |
