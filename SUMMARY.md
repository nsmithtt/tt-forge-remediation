# Remediation Summary: glm_ocr-image_to_text-pytorch-glm_ocr-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[glm_ocr/image_to_text/pytorch-glm_ocr-single_device-inference]

## Result
FAIL — Conv3d CB page size 192 less than config tensor page size 392 after Tier A L1 overflow fix

## Stack layer
tt-mlir, tt-metal

## Tier
A

## Bug fingerprint
conv3d-cb-page-size-vs-tensor-mismatch

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

From device log (second run, after Tier A tt-mlir fix):
```
TT_FATAL: CB page size 192 should be greater than the config tensor page size 392
```

First-run device log (before Tier A fix, included for context):
```
TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=10,y=9)]
grow to 1747968 B which is beyond max L1 size of 1572864 B
```

The originally reported failure message (`The image processor of type Glm46VImageProcessor
is now loaded as a fast processor by default...`) is a transformers 5.x warning (loader layer),
fixed in the loader.

## Root cause
Two sequential compiler-stack bugs in the Conv3d path, plus loader-layer issues.

**Bug 1 — Conv3d CB L1 overflow (tt-mlir, fixed):** `GlmOcrVisionPatchEmbed` uses
`nn.Conv3d(in_channels=3, out_channels=1024, kernel_size=[2,14,14])`. The default
`c_in_block=TILE_WIDTH=32` made the Conv3d program factory allocate static CBs of
1,747,968 B (1.75 MB), exceeding the per-core L1 limit of 1,572,864 B (1.5 MB).
A Tier A fix in `TTIRToTTNN.cpp` computes a safe `c_in_block` by halving from the
tile width until it fits within `MAX_CB_TILES * TILE_WIDTH / kernelElements` ≈ 20,
yielding `c_in_block=16` for this kernel.

**Bug 2 — Conv3d CB page size mismatch (tt-metal, blocking):** After the L1 overflow
fix sets `c_in_block=16`, the tt-metal Conv3d kernel computes a CB page size of 192
bytes, which is smaller than the input tensor's effective page size of 392 bytes
(= kernel_depth × kernel_h × kernel_w = 2 × 14 × 14). The assertion
`CB page size >= config tensor page size` fails. The 192-byte CB page size appears to
derive from `c_in_block × dtype_size × factor`, while 392 comes from the kernel volume.
The tt-metal Conv3d kernel does not handle the case where `c_in_block` forces a CB page
size smaller than the kernel's volumetric stride. This is a second compiler-stack bug;
per skill rules (one Tier A fix per report), it is filed without a fix attempt.

**Loader-layer bugs (fixed):**
- `GlmOcrVisionModel.rot_pos_emb` iterates `for t, h, w in grid_thw` on a device
  tensor, producing device scalars passed to `torch.arange` → Error code: 13.
- `GlmOcrModel.get_rope_index` and `get_image_features` call `.tolist()` on device
  tensors.
- `AutoProcessor.from_pretrained` missing `use_fast=False` (transformers 5.x default
  changed for `Glm46VImageProcessor`).

## Fix
**Loader fixes applied** in
`tt-forge-models/glm_ocr/image_to_text/pytorch/loader.py`
(branch `remediation/glm_ocr-image_to_text-pytorch-glm_ocr-single_device-inference`,
commit `4e536955b7d7b7df5d68c6c4b99b3d986bebb835`):
- Added `_patch_glm_ocr_for_tt_device()` which moves `grid_thw`/`input_ids`/`attention_mask`
  to CPU before the `.tolist()` and iteration calls in `rot_pos_emb`, `get_rope_index`,
  `get_image_features`, and `get_video_features`.
- Added `use_fast=False` to `AutoProcessor.from_pretrained` call.

**Tier A compiler fix applied** in `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`
(branch `remediation/glm_ocr-image_to_text-pytorch-glm_ocr-single_device-inference`,
commit `335e49b1070e32f57f173bdad7281f46d022abd2`):
- `Conv3dOpConversionPattern::matchAndRewrite` now computes a safe `c_in_block`:
  `kernelElements = dim2 × dim3 × dim4`, `maxCInBlock = MAX_CB_TILES × TILE_WIDTH /
  kernelElements`, then halves `c_in_block` from TILE_WIDTH until it fits and divides
  the aligned channel count. For GLM-OCR kernel [2,14,14]: kernelElements=392,
  maxCInBlock≈20, resulting in `c_in_block=16` (down from 32).
- This fix resolved the L1 overflow but exposed Bug 2.

**Required fix for Bug 2 (not attempted):** The tt-metal Conv3d kernel
(program factory / reader kernel files) must reconcile the CB page size calculation
with `c_in_block` such that the allocated page size is never smaller than the kernel's
volumetric stride of 392 elements. The fix would live in tt-metal's experimental Conv3d
program factory, likely in the CB configuration step that sets the reader CB page size.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    222.93s (second run, with Tier A fix applied)
- Tier A attempts: 1

## Files changed
- `tt-forge-models/glm_ocr/image_to_text/pytorch/loader.py` — loader fixes (use_fast=False,
  TT device .tolist() patches for rot_pos_emb / get_rope_index / get_image_features /
  get_video_features)
- `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` — safe c_in_block computation for
  Conv3dOpConversionPattern to prevent static CB L1 overflow

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 335e49b1070e32f57f173bdad7281f46d022abd2 |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 4e536955b7d7b7df5d68c6c4b99b3d986bebb835 |
