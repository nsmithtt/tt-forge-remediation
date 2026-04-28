# Remediation Summary: qwen3_vl_2b_emoji_base_i1_gguf/image_to_text/pytorch-2b_emoji_base_i1_gguf-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[qwen3_vl_2b_emoji_base_i1_gguf/image_to_text/pytorch-2b_emoji_base_i1_gguf-single_device-inference]

## Result
FAIL — `ttnn.experimental.Conv3dDeviceOperation` static circular buffers (2,247,168 B) exceed L1 (1,572,864 B): MLIR pads C_in from 3 to 32 for tile alignment, inflating the K dimension of the patch embedding matmul from 48 to 512 tiles, making the vol2col and weight CBs each 1 MB — together already over the 1.5 MB limit.

## Stack layer
tt-metal, tt-mlir

## Tier
B

## Bug fingerprint
conv3d-small-cin-l1-cb-overflow

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

Underlying device log:
```
TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=10,y=9)] grow to 2247168 B
which is beyond max L1 size of 1572864 B
tt::tt_metal::detail::ProgramImpl::validate_circular_buffer_region(tt::tt_metal::IDevice const*)
```

The test also triggered three loader issues which were fixed before reaching silicon (see Fix section).

## Root cause
**Two classes of issues were found: loader bugs (fixed) and a compiler-stack bug (unfixed).**

### Loader issue 1: `qwen3vl` GGUF architecture not registered (FIXED)
`transformers.modeling_gguf_pytorch_utils.GGUF_SUPPORTED_ARCHITECTURES` does not
include `qwen3vl`, so `Qwen3VLForConditionalGeneration.from_pretrained("...",
gguf_file="...")` raises `ValueError: GGUF model with architecture qwen3vl is not
supported yet.` before reaching silicon.

Additionally, the mradermacher i1-GGUF ships only LM weights (`blk.*`,
`output_norm.*`, `token_embd.*`); no vision encoder tensors are present. Loading
the GGUF with `Qwen3VLForConditionalGeneration` would give a randomly-initialised
vision encoder — garbage PCC for an `image_to_text` task.

Fix: load from the base model `adwel94/Qwen3-VL-2B-Emoji-Base` directly so the
vision encoder has proper trained weights. The GGUF repo has no `config.json`;
`adwel94/Qwen3-VL-2B-Emoji-Base` is the source model for the quantised GGUF.

**Layer: loader (tt-forge-models)**

### Loader issue 2: `.tolist()` on TT device tensors (FIXED)
`Qwen3VLVisionModel.fast_pos_embed_interpolate`, `rot_pos_emb`, and
`Qwen3VLModel.get_rope_index` / `get_image_features` call `.tolist()` on
`grid_thw` / `input_ids` for Python control flow. TT device does not support
eager tensor reads; these trigger a device sync that fails with Error code: 13.

**Layer: loader (tt-forge-models)**

### Loader issue 3: missing pixel limits on processor (FIXED)
Without `min_pixels` / `max_pixels` limits, the demo.jpeg (1376×2048 px) produces
`image_grid_thw = [[1, 86, 128]]` = 11,008 patches. All other Qwen VL loaders
set `min_pixels=56*56, max_pixels=13*28*1280`.

**Layer: loader (tt-forge-models)**

### Compiler-stack bug: Conv3d L1 overflow (UNFIXED)
`Qwen3VLVisionPatchEmbed` uses `nn.Conv3d(in_channels=3, out_channels=1024,
kernel_size=[2,16,16], stride=[2,16,16])`. During `TTIRToTTNN` lowering,
`Conv3dOpConversionPattern` pads C_in from 3 to 32 (the next multiple of
`TILE_WIDTH`) for memory alignment, then passes the padded input to
`ttnn.experimental.Conv3dDeviceOperation`.

This padding inflates:
- `patch_size` from `2×16×16×3 = 1536` to `2×16×16×32 = 16384`
- `matmul_K_t` from `1536/32 = 48` to `16384/32 = 512`

The default Conv3d config uses `C_out_block = TILE_WIDTH = 32` (minimum),
`T/H/W_out_block = 1` (single spatial patch per block). Even at this minimum
blocking, the two largest CBs per core are:
- `cb_vol2col_tiled = tile_size × matmul_M_t × matmul_K_t = 2048 × 1 × 512 = 1,048,576 B`
- `cb_weight_tiled  = tile_size × matmul_K_t × matmul_N_t = 2048 × 512 × 1 = 1,048,576 B`

Together these two CBs sum to 2,097,152 B — already over the 1,572,864 B L1
limit before any other CB is added. There is no block size that fits: reducing
`C_out_block` is already at the minimum (1 tile); reducing spatial block sizes
does not affect `matmul_K_t`. The C_in padding that inflates K is structural.

**Layer: tt-metal (runtime), tt-mlir (lowering)**

## Fix
### Applied (loader, tt-forge-models):

1. **Base model loading:** Changed `pretrained_model_name` from
   `mradermacher/Qwen3-VL-2B-Emoji-Base-i1-GGUF` (GGUF with unsupported
   architecture and missing vision encoder) to `adwel94/Qwen3-VL-2B-Emoji-Base`
   (the full-precision source model with proper vision encoder).
   Removed `gguf_file=` kwarg and added explicit `torch_dtype=torch.bfloat16`.

2. **`.tolist()` patches:** Monkey-patched four `Qwen3VL` methods in
   `_patch_qwen3vl_for_tt_device()` to move metadata tensors to CPU before
   `.tolist()` calls:
   - `Qwen3VLVisionModel.fast_pos_embed_interpolate(grid_thw)` → `grid_thw.cpu()`
   - `Qwen3VLVisionModel.rot_pos_emb(grid_thw)` → `grid_thw.cpu()`
   - `Qwen3VLModel.get_rope_index(input_ids, image_grid_thw, ...)` → all args `.cpu()`, outputs back to original device
   - `Qwen3VLModel.get_image_features(pixel_values, image_grid_thw)` → `image_grid_thw.cpu()`

3. **Pixel limits:** Added `self.processor.image_processor.min_pixels = 56*56` and
   `max_pixels = 13*28*1280` matching the established pattern for all Qwen VL loaders.

Remediation branch in tt-forge-models:
`remediation/qwen3_vl_2b_emoji_base_i1_gguf-image_to_text-pytorch-2b_emoji_base_i1_gguf-single_device-inference`

### Proposed (compiler-stack):
The root fix requires either:
1. **In `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`
   (`Conv3dOpConversionPattern`):** Do not pad C_in to `TILE_WIDTH` when C_in is
   small (e.g., C_in=3). Instead, leave C_in as-is and let the Conv3d kernel
   handle non-aligned C_in via its existing `padded_patch_size` rounding. This
   requires a matching change to the Conv3d kernel reader to support non-tile-
   aligned input channels. The L1 alignment constraint
   (`C_in_block % l1_alignment == 0`) must also be relaxed or worked around.

2. **Alternatively:** In
   `tt-metal/ttnn/cpp/ttnn/operations/experimental/conv3d/device/conv3d_program_factory.cpp`,
   add an L1 budget pre-check before CB allocation. When the static CB sum would
   exceed `hal::get_max_worker_l1_unreserved_size()`, reduce `C_in_block` (and
   adjust weight preparation accordingly) until it fits. This requires coordinated
   changes to weight preparation in `prepare_conv3d_weights.cpp` and the MLIR
   lowering to pass matching `cInBlock`.

## Tier B justification
cross-cutting — fixing the C_in padding that inflates the K dimension requires
coordinated changes to the MLIR lowering (`tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`)
and the tt-metal Conv3d kernel reader/program-factory (`conv3d_program_factory.cpp`,
`prepare_conv3d_weights.cpp`) across two repos. Option 2 (L1 budget pre-check)
also requires coordinating C_in_block selection between MLIR weight preparation
and the runtime program factory, touching ≥3 files.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    84.75s (0:01:24) after loader fixes applied
- Tier A attempts: N/A

## Files changed
- `qwen3_vl_2b_emoji_base_i1_gguf/image_to_text/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | c6ee9c8c9e3f0af8c6d2d5448295356a58df3d8b |
| tt-forge-models | 72464a2cee06036e5d1f16adb26b9f439c73b707 |
