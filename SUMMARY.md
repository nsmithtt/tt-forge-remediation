# Remediation Summary: cosmos_reason2_gguf-image_to_text-pytorch-2b_gguf-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[cosmos_reason2_gguf/image_to_text/pytorch-2b_gguf-single_device-inference]

## Result
FAIL — Conv3d circular-buffer static allocation exceeds per-core L1 after loader fixes applied; Tier B compiler-stack bug

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
conv3d-cin-padding-k-dim-l1-overflow

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (before loader fixes):
```
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```
at `grid_thw.tolist()` inside `Qwen3VLVisionModel.fast_pos_embed_interpolate`.

After loader fixes applied (on remediation branch), new failure:
```
TT_THROW @ ttnn/cpp/ttnn/operations/experimental/conv3d/device/conv3d_program_factory.cpp:275:
info:
tt::exception
what:
Statically allocated circular buffers on core range [(x=0,y=0) - (x=10,y=9)] grow to
2247168 B which is beyond max L1 size of 1572864 B
```
from `Conv3dDeviceOperation`.

## Root cause
Two bugs, one fixed in the loader (SILICON fix not achieved, Tier B compiler bug remains):

**Bug 1 (loader, fixed):** `Qwen3VLVisionModel.fast_pos_embed_interpolate`, `rot_pos_emb`,
`Qwen3VLModel.get_rope_index`, and `get_image_features` all call `.tolist()` on tensors
(`grid_thw`, `input_ids`, `image_grid_thw`) that the test runner has moved to TT device.
TT device does not support eager Python-side tensor reads; `.tolist()` triggers a device
sync that fails with `INTERNAL: Error code: 13`. The fix (applied in the loader remediation
branch) moves these metadata tensors to CPU before the problematic method calls.

**Bug 2 (tt-mlir / tt-metal, Tier B, unfixed):** `TTIRToTTNN.cpp` pads `C_in=3` (RGB
patch embedding input) to `TILE_WIDTH=32` before lowering to the TTNN Conv3d op. With the
Qwen3VL patch-embed kernel `[T=2, H=16, W=16]`, this yields:

```
patch_size  = T × H × W × C_in_padded = 2 × 16 × 16 × 32 = 16 384 elements
matmul_K_t  = patch_size / TILE_WIDTH  = 16 384 / 32        = 512 tiles
```

The Conv3d program factory allocates two large circular buffers per core:
- `cb_vol2col_tiled`: `matmul_M_t × matmul_K_t × tile_size = 1 × 512 × 2048 B = 1 048 576 B`
- `cb_weight_tiled`: `matmul_K_t × matmul_N_t × tile_size = 512 × 1 × 2048 B = 1 048 576 B`

Total for these two alone: **2 097 152 B (2 MB)**, which exceeds the maximum unreserved
L1 per worker core of **1 572 864 B (1.5 MB)**. The `TT_THROW` fires when the runtime
checks static CB allocation during kernel dispatch.

The root cause is that the MLIR lowering (`TTIRToTTNN.cpp`) unconditionally pads
`C_in` to `TILE_WIDTH=32` even when the physical channel count (3) is far smaller.
The default `Conv3dConfig` in `conv3d.cpp` then uses `C_in_block=32`, making the
K-dimension of the matmul proportionally large.

## Fix
**Loader fixes (committed to tt-forge-models remediation branch
`remediation/cosmos_reason2_gguf-image_to_text-pytorch-2b_gguf-single_device-inference`):**

- `tt-xla/third_party/tt_forge_models/cosmos_reason2_gguf/image_to_text/pytorch/loader.py`:
  - Added `_register_qwen3vl_gguf_arch()`: registers `qwen3vl` in
    `GGUF_SUPPORTED_ARCHITECTURES`, maps its config key to `qwen3`, patches
    `get_gguf_hf_weights_map` to translate `qwen3_vl` → `qwen3vl` and to skip
    vision submodules (preventing `output_norm` tensor routing to
    `model.visual.merger.norm.weight`).
  - Added `_patch_qwen3vl_for_tt_device()`: patches 4 Qwen3VL methods
    (`fast_pos_embed_interpolate`, `rot_pos_emb`, `get_rope_index`,
    `get_image_features`) to move metadata tensors (`grid_thw`, `input_ids`,
    `image_grid_thw`) to CPU before `.tolist()` calls, avoiding TT device sync.
  - Updated `load_model` to load the public GGUF checkpoint
    (`apolo13x/Cosmos-Reason2-2B-GGUF`, file `Cosmos-Reason2-2B-Q4_K_M.gguf`)
    using `Qwen/Qwen3-VL-2B-Instruct` as the base config and processor source.
  - Added pixel limits (`min_pixels = 56*56`, `max_pixels = 13*28*1280`) to
    constrain vision-encoder input size to hardware L1 budget.

**Proposed compiler-stack fix (Tier B, not attempted):**

The fix spans `tt-mlir` and `tt-metal` and requires coordinated changes:

1. `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` (Conv3d lowering):
   Stop padding `C_in` to `TILE_WIDTH=32`. Pass the actual channel count and
   let the runtime choose an appropriate `C_in_block` value (≤ actual `C_in`).

2. `tt-metal/ttnn/cpp/ttnn/operations/experimental/conv3d/conv3d.cpp` (default config):
   Change the default `C_in_block` from `TILE_WIDTH=32` to a value derived from
   the actual `C_in`, e.g. `min(C_in, TILE_WIDTH)`.

3. `tt-metal/ttnn/cpp/ttnn/operations/experimental/conv3d/device/prepare_conv3d_weights.cpp`
   (weight preparation): Verify that weight tiling/padding logic handles
   `C_in_block < TILE_WIDTH` correctly without introducing zero-padding artifacts
   that would break PCC.

## Tier B justification
**cross-cutting** — The fix requires coordinated changes across two repos
(`tt-mlir` and `tt-metal`) touching at least three files
(`TTIRToTTNN.cpp`, `conv3d.cpp`, `prepare_conv3d_weights.cpp`).
The MLIR side must stop emitting the padded channel count, and the runtime side
must change its default block-size selection and weight-preparation logic
consistently; the two changes cannot be made independently.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    N/A (failed before inference)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/cosmos_reason2_gguf/image_to_text/pytorch/loader.py`
  (loader remediation branch: 5 commits)

## Submodule hashes
| Submodule       | Commit                                   |
|-----------------|------------------------------------------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 804a583d46efa379e16538836953c4dc76e02f95 |
| tt-forge-models | eb2cab902a9be68b82bd4a21eff34bceb0eb6e54 |
