# Remediation Summary: editscore_qwen3_vl_8b_instruct/image_to_text/pytorch-editscore_qwen3_vl_8b_instruct-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[editscore_qwen3_vl_8b_instruct/image_to_text/pytorch-editscore_qwen3_vl_8b_instruct-single_device-inference]

## Result
FAIL — `ttnn.experimental.Conv3dDeviceOperation` statically allocates circular buffers that exceed L1 (2,247,168 B > 1,572,864 B) regardless of input batch size. This is a compiler/runtime bug in tt-metal's Conv3d kernel.

## Stack layer
tt-metal

  - `loader`         — bug was in tt_forge_models or test inputs
  - `tt-xla`         — bug in compiler frontend (PJRT, torch_xla bridge)
  - `tt-mlir`        — bug in compiler core (StableHLO→TTIR lowering)
  - `tt-metal`       — bug in backend runtime / kernels
  - `hardware-class` — model exceeds single-device capacity (XFAIL)
  - `n/a`            — NO_FIX_NEEDED (could not reproduce)

## Tier
B

  - `N/A` — loader fix, no fix needed, or hardware-class XFAIL
  - `A`   — compiler-stack fix attempted (succeeded → SILICON_PASS,
            ran out of attempts → FAIL with explanation)
  - `B`   — compiler-stack bug filed without attempting fix

## Bug fingerprint
conv3d-l1-circular-buffer-overflow

  Format: `<area>-<short-description>`. Use the same string verbatim
  whenever a later report hits the same bug — this is how the audit
  groups failures.

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

The original surface error was `grid_thw.tolist()` in `fast_pos_embed_interpolate`,
but that was a forced-sync trigger that caused the pending Conv3d graph to compile and fail.

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
Qwen demo.jpeg (≈1376×2048) would produce far more patches than needed.
All other Qwen VL loaders in the test suite set
`min_pixels=56*56, max_pixels=13*28*1280` as standard practice.

**Layer: loader (tt-forge-models)**

### Compiler-stack bug: Conv3d L1 overflow (UNFIXED)
`Qwen3VLVisionPatchEmbed` uses `nn.Conv3d(in_channels=3, out_channels=embed_dim,
kernel_size=[2,16,16], stride=[2,16,16])` to compute patch embeddings.
`ttnn.experimental.Conv3dDeviceOperation` statically allocates 2,247,168 B
of circular buffers on the full 11×10 core grid, exceeding the 1,572,864 B
L1 limit. This failure occurs regardless of the input batch size or pixel
limits applied (same circular buffer allocation).

**Layer: tt-metal (runtime)**

This is the same bug as documented in report
`bartowski_qwen_qwen3_vl_2b_instruct_gguf-image_to_text-pytorch-qwen_qwen3_vl_2b_instruct_gguf-single_device-inference`.

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
kernel=[2,16,16], stride=[2,16,16]` (for the Qwen3-VL-8B vision patch embedding).

## Tier B justification (FAIL with Tier=B only — omit otherwise)
new-infrastructure

The Conv3d kernel in tt-metal requires either a new tiling/streaming execution
strategy or a fallback path when static circular buffers would exceed L1; this
is new infrastructure that does not exist and cannot be added in a scoped single-file fix.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    117.20s
- Tier A attempts: N/A

## Files changed
- `editscore_qwen3_vl_8b_instruct/image_to_text/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 6ebbb82a665817051a919528fa11ff613e26970d |
