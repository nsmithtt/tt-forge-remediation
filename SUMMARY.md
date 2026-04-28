# Remediation Summary: evocua/pytorch-8b_20260105-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[evocua/pytorch-8b_20260105-single_device-inference]

## Result
FAIL — `ttnn.experimental.Conv3dDeviceOperation` statically allocates circular buffers that exceed L1 (2,247,168 B > L1 max 1,572,864 B). This is the same compiler/runtime bug in tt-metal's Conv3d kernel found in the bartowski Qwen3-VL-2B report.

## Stack layer
tt-metal

## Tier
B

## Bug fingerprint
conv3d-l1-circular-buffer-overflow

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

Underlying device log (critical, before the Python error propagates):
```
TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=10,y=9)] grow to 2247168 B
which is beyond max L1 size of 1572864 B
```

The original surface error was `grid_thw.tolist()` in `fast_pos_embed_interpolate`, which was a forced-sync trigger causing the pending Conv3d patch-embedding graph to compile and fail.

## Root cause
**Two issues found: one loader bug (fixed) and one compiler-stack bug (unfixed).**

### Loader bug: `.tolist()` on TT device tensors (FIXED)
`Qwen3VLVisionModel.fast_pos_embed_interpolate` and `rot_pos_emb` call
`grid_thw.tolist()` for Python control flow. `Qwen3VLModel.get_rope_index`
calls `input_ids.tolist()`, `image_grid_thw.tolist()`, and
`video_grid_thw.tolist()`. `Qwen3VLModel.get_image_features` calls
`(image_grid_thw.prod(-1) // ...).tolist()`. TT device does not support
eager tensor readback — any `.tolist()` on a TT tensor triggers a device
sync that fails with Error code: 13.

Additionally, the processor was initialized without `min_pixels`/`max_pixels`
limits, which is non-standard for Qwen VL loaders in this test suite.

**Layer: loader (tt-forge-models)**

### Compiler-stack bug: Conv3d L1 overflow (UNFIXED)
`Qwen3VLVisionPatchEmbed` uses `nn.Conv3d(in_channels=3, out_channels=embed_dim,
kernel_size=[2,16,16], stride=[2,16,16])` to compute patch embeddings.
`ttnn.experimental.Conv3dDeviceOperation` statically allocates 2,247,168 B
of circular buffers on the full 11×10 core grid, exceeding the 1,572,864 B
L1 limit. This failure occurs regardless of input size (the visual encoder
architecture is shared between Qwen3-VL-2B and 8B, producing identical CB
allocation). This is the same bug fingerprinted in the bartowski Qwen3-VL-2B
remediation report.

**Layer: tt-metal (runtime)**

## Fix
### Applied (loader, tt-forge-models):

**Patch `.tolist()` callers:** Monkey-patch four `Qwen3VL` methods to move
metadata tensors to CPU before `.tolist()` calls while returning computed
tensors (`position_ids`, `rope_deltas`) back to the original device:
- `Qwen3VLVisionModel.fast_pos_embed_interpolate(grid_thw)` → `grid_thw.cpu()`
- `Qwen3VLVisionModel.rot_pos_emb(grid_thw)` → `grid_thw.cpu()`
- `Qwen3VLModel.get_rope_index(input_ids, image_grid_thw, ...)` → all args `.cpu()`, returns back to orig device
- `Qwen3VLModel.get_image_features(pixel_values, image_grid_thw)` → `image_grid_thw.cpu()`

**Add pixel limits:** Set `self.processor.image_processor.min_pixels = 56*56` and
`self.processor.image_processor.max_pixels = 13*28*1280` after processor load,
matching the established pattern for all Qwen VL loaders.

### Proposed (compiler-stack):
In `tt-metal`'s `Conv3dDeviceOperation`, implement either:
1. A check that the static circular buffer allocation fits within L1, and fall
   back to a tiled/streamed execution strategy when it does not.
2. A streaming or split-batch strategy for Conv3d kernels where the
   weight + I/O tile footprint exceeds L1 capacity.

The kernel parameters causing the overflow: `in_channels=3, out_channels=1152,
kernel=[2,16,16], stride=[2,16,16]` (Qwen3-VL vision patch embedding).

## Tier B justification (FAIL with Tier=B only — omit otherwise)
Tier B indicator: `new-infrastructure`

Fixing the Conv3d CB overflow requires implementing either a tiled/streaming
execution strategy or an L1-budget-aware fallback in the Conv3d kernel — new
infrastructure in tt-metal beyond a scoped formula tweak.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    60.27s (after loader fixes)
- Tier A attempts: N/A

## Files changed
- `evocua/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 2dc64c4f500020bfb9388ea9aa001e7b72b41bcf |
| tt-forge-models | 479c3e6f72e5a6af7491e379d8f03d5e0e5cab1a |
