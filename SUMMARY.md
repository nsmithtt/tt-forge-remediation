# Remediation Summary: bartowski_qwen_qwen3_vl_2b_instruct_gguf/image_to_text/pytorch-qwen_qwen3_vl_2b_instruct_gguf-single_device-inference

## Skill version
14

## Test
tests/runner/test_models.py::test_all_models_torch[bartowski_qwen_qwen3_vl_2b_instruct_gguf/image_to_text/pytorch-qwen_qwen3_vl_2b_instruct_gguf-single_device-inference]

## Result
FAIL — `ttnn.experimental.Conv3dDeviceOperation` statically allocates circular buffers that exceed L1 (2,247,168 B > L1 max 1,572,864 B) regardless of input batch size. This is a compiler/runtime bug in tt-metal's Conv3d kernel.

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

The original surface error was `grid_thw.tolist()` in `fast_pos_embed_interpolate`, but that was a forced-sync trigger that caused the pending Conv3d graph to compile and fail.

## Root cause
**Two issues were found: one loader bug (fixed) and one compiler-stack bug (unfixed).**

### Loader bug 1: `.tolist()` on TT device tensors (FIXED)
`Qwen3VLVisionModel.fast_pos_embed_interpolate` and `rot_pos_emb` call
`grid_thw.tolist()` for Python control-flow. `Qwen3VLModel.get_rope_index`
calls `input_ids.tolist()`, `image_grid_thw.tolist()`, and
`video_grid_thw.tolist()`. `Qwen3VLModel.get_image_features` calls
`(image_grid_thw.prod(-1) // ...).tolist()`. TT device does not support
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
`Qwen3VLVisionPatchEmbed` uses `nn.Conv3d(in_channels=3, out_channels=embed_dim,
kernel_size=[2,16,16], stride=[2,16,16])` to compute patch embeddings.
`ttnn.experimental.Conv3dDeviceOperation` statically allocates 2,247,168 B
of circular buffers on the full 11×10 core grid, exceeding the 1,572,864 B
L1 limit. This failure occurs regardless of the input batch size (1,768
patches or 11,008 patches produce the same circular buffer allocation).

**Layer: tt-metal (runtime)**

Source: `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/...` or the
Conv3d kernel implementation — the static circular buffer allocation must
be made aware of L1 budget, or tiling/streaming must be implemented for
large Conv3d kernels.

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

The kernel parameters causing the overflow: `in_channels=3, out_channels=1152,
kernel=[2,16,16], stride=[2,16,16], batch=1768` (for the Qwen3-VL-2B vision
patch embedding).

## Verification
FAIL — test still exits with `RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13` after loader fixes are applied. The Conv3d circular buffer overflow is the residual failure on n150 silicon.

Hardware: n150 (blackhole)

## Files changed
- `bartowski_qwen_qwen3_vl_2b_instruct_gguf/image_to_text/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 864525836fda30c62a05aba63cf4fedc0cbb392b |
| tt-forge-models | 80044a0b88f4881962b5c05dfa7e554de95f4197 |
