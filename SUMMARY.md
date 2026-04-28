# Remediation Summary: bu_30b-pytorch-30b_a3b_preview-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[bu_30b/pytorch-30b_a3b_preview-single_device-inference]

## Result
FAIL — Conv3d L1 overflow: statically-allocated circular buffers grow to 2,247,168 B, exceeding L1 max of 1,572,864 B; Tier B compiler-stack bug

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

Device log (after loader tolist() patches applied):
TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=10,y=9)] grow to
2247168 B which is beyond max L1 size of 1572864 B
— tt::runtime::ttnn::operations::conv::run(Conv3dOp) → ttnn::experimental::conv3d →
  ttnn::prim::conv3d → Conv3dDeviceOperation

## Root cause
`bu-30b-a3b-preview` is a `Qwen3VLMoeForConditionalGeneration` model whose vision encoder
(`Qwen3VLMoeVisionPatchEmbed`) uses `nn.Conv3d(3, 1152, kernel_size=[2,16,16],
stride=[2,16,16])`. The `ttnn.experimental.Conv3dDeviceOperation` statically allocates
circular buffers determined by the kernel parameters. For `C_in=3, C_out=1152,
kernel=[2,16,16]` the tt-mlir lowering in `TTIRToTTNN.cpp` pads `C_in=3` to
`TILE_WIDTH=32`, resulting in `matmul_K_t = (2×16×16×32)/32 = 512 tiles`. The
`conv3d_program_factory.cpp` then allocates `cb_vol2col_tiled` (1×512×2048 B = 1 MB) and
`cb_weight_tiled` (512×1×2048 B = 1 MB), totalling 2.1 MB > 1.5 MB L1 on every core in the
11×10 core grid.  This overflow is independent of batch size or input spatial dimensions.

A loader-level bug was also present (independently fixed): the `qwen3_vl_moe` model calls
`.tolist()` on TT device tensors in four methods (`fast_pos_embed_interpolate`, `rot_pos_emb`,
`get_rope_index`, `get_image_features`), causing an earlier "Bad StatusOr access: INTERNAL:
Error code: 13" before the compiler ever runs.  That loader bug is fixed in this report.

## Fix
The loader-layer tolist() bug was fixed in
`tt_forge_models/bu_30b/pytorch/loader.py` on branch
`remediation/bu_30b-pytorch-30b_a3b_preview-single_device-inference`:
- Added `_apply_tolist_patches()` that monkey-patches all four `qwen3_vl_moe` methods to move
  metadata tensors (`grid_thw`, `input_ids`, `image_grid_thw`) to CPU before `.tolist()`, and
  returns `position_ids`/`rope_deltas` back to the original device after `get_rope_index`.
- Added pixel limits (`min_pixels=56*56`, `max_pixels=13*28*1280`) on the image processor to
  prevent excessive patch counts from the 1376×2048 demo image.

The Conv3d L1 overflow is unfixed (Tier B). The proposed fix lives in the compiler stack:
- `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`: change the Conv3d lowering to avoid
  padding `C_in=3` to `TILE_WIDTH=32` for the vol2col matrix, or choose a smaller core-grid
  that keeps CB allocation within L1.
- `tt-metal/tt_metal/impl/...conv3d_program_factory.cpp`: reduce per-core CB allocation when
  the padded kernel volume would exceed L1, e.g. by tiling across core groups.
- These changes require coordinated alignment across TTIRToTTNN, the conv3d program factory,
  and `prepare_conv3d_weights.cpp` (at least 3 files, cross-repo).

## Tier B justification
`more-than-3-files` / `cross-repo`: fixing the Conv3d L1 overflow requires coordinated
changes across `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`,
`tt-metal/tt_metal/impl/.../conv3d_program_factory.cpp`, and
`tt-metal/tt_metal/impl/.../prepare_conv3d_weights.cpp` — touching at least 3 files across 2
repositories, with no single-function scoped change available.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    235.65s (after loader fixes; fails at Conv3d kernel dispatch)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/bu_30b/pytorch/loader.py` — tolist() patches + pixel limits

## Submodule hashes
| Submodule       | Commit                                   |
|-----------------|------------------------------------------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 14f0cdc9bfc5bc2de900f400e62e87740cdb72c3 |
| tt-forge-models | efb36869ef870a7bd85a99de74400bdad98a9fa5 |
