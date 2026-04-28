# Remediation Summary: firered_ocr_gguf/image_to_text/pytorch-Q4_K_M-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[firered_ocr_gguf/image_to_text/pytorch-Q4_K_M-single_device-inference]

## Result
FAIL — `ttnn.experimental.Conv3dDeviceOperation` statically allocates circular buffers that exceed L1 (2,247,168 B > L1 max 1,572,864 B) regardless of input batch size. This is a compiler/runtime bug in tt-metal's Conv3d kernel.

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
...
tt::runtime::ttnn::operations::conv::run(tt::target::ttnn::Conv3dOp const*, ...)
```

The original surface error was `grid_thw.tolist()` in `fast_pos_embed_interpolate`, but that was a loader bug (missing `.tolist()` patches) that caused the pending Conv3d graph to compile and fail.

## Root cause
**Two issues were found: one loader bug (fixed) and one compiler-stack bug (unfixed).**

### Loader bug 1: `.tolist()` on TT device tensors (FIXED)
`Qwen3VLVisionModel.fast_pos_embed_interpolate` and `rot_pos_emb` call
`grid_thw.tolist()` for Python control-flow. `Qwen3VLModel.get_rope_index`
calls `input_ids.tolist()`, `image_grid_thw.tolist()`, and
`video_grid_thw.tolist()`. `Qwen3VLModel.get_image_features` calls
`image_grid_thw.tolist()` (via the inner method). TT device does not support
eager tensor readback — any `.tolist()` on a TT tensor triggers a device
sync that fails with Error code: 13.

**Layer: loader (tt-forge-models)**

### Loader bug 2: missing processor pixel limits (FIXED)
The processor was initialized without `min_pixels`/`max_pixels`, so the
Qwen demo.jpeg (≈1376×2048) produced `image_grid_thw = [[1, 86, 128]]` =
11,008 patches. All other Qwen VL loaders in the test suite set
`min_pixels=56*56, max_pixels=13*28*1280` as standard practice. With the
limit, the demo.jpeg produces `image_grid_thw = [[1, 34, 52]]` = 1,768
patches.

**Layer: loader (tt-forge-models)**

### Compiler-stack bug: Conv3d L1 overflow (UNFIXED)
`Qwen3VLVisionPatchEmbed` uses `nn.Conv3d(in_channels=3, out_channels=1152,
kernel_size=[2,16,16], stride=[2,16,16])` to compute patch embeddings.
`ttnn.experimental.Conv3dDeviceOperation` statically allocates 2,247,168 B
of circular buffers on the full 11×10 core grid, exceeding the 1,572,864 B
L1 limit. This failure occurs regardless of the input batch size (1,768
patches or 11,008 patches produce the same circular buffer allocation).

The TTNN op: `batch_size=1768, in_channels=32, out_channels=1152,
kernel=[2,16,16], stride=[2,16,16]`.

**Layer: tt-metal (runtime)**

## Fix
### Applied (loader, tt-forge-models):

**Patch `.tolist()` callers:** Monkey-patch four `Qwen3VL` methods to move
metadata tensors to CPU before `.tolist()` calls while returning computed
tensors (`position_ids`, `rope_deltas`) back to the original device:
- `Qwen3VLVisionModel.fast_pos_embed_interpolate(grid_thw)` → `grid_thw.cpu()`
- `Qwen3VLVisionModel.rot_pos_emb(grid_thw)` → `grid_thw.cpu()`
- `Qwen3VLModel.get_rope_index(input_ids, image_grid_thw, ...)` → all args `.cpu()`, return back to orig device
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

The kernel parameters causing the overflow: `in_channels=32, out_channels=1152,
kernel=[2,16,16], stride=[2,16,16], batch=1768` (for the Qwen3-VL-2B vision
patch embedding).

## Tier B justification (FAIL with Tier=B only — omit otherwise)
<cross-repo>
The Conv3d circular buffer overflow requires implementing a tiled/streamed
execution strategy in `tt-metal`'s `Conv3dDeviceOperation` kernel — new
infrastructure that goes beyond a scoped single-function fix.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    179.06s (2:59) after loader fixes
- Tier A attempts: N/A

## Files changed
- `firered_ocr_gguf/image_to_text/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a5d811c7c0d2f9a3aca94740f24ebe3348103ceb |
| tt-forge-models | 092c6fcc2ac01ff9cd0464a1a0836b1fddcc567a |
