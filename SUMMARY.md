# Remediation Summary: mask2former/panoptic_segmentation/pytorch-Swin_Tiny_Cityscapes-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mask2former/panoptic_segmentation/pytorch-Swin_Tiny_Cityscapes-single_device-inference]

## Result
FAIL — PCC=0.369 (required 0.99) after loader fix; root cause is a compiler-stack numerical error in Mask2Former's multi-scale deformable attention

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
mask2former-multiscale-deformable-attention-pcc-wrong

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (loader): "The image processor of type `Mask2FormerImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`."

After loader fix: AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.36977503157311065. Required: pcc=0.99.

## Root cause

**Loader bug (fixed):** transformers 5.x changed `AutoImageProcessor.from_pretrained` to load `Mask2FormerImageProcessor` as a fast processor by default. Adding `use_fast=False` resolves the original error.

**Compiler bug (Tier B):** After the loader fix, the model compiles and executes (15 subgraphs, ~22 min) but produces PCC=0.369 — far below any expected BF16 precision floor (which would be ≥0.86 for a 256-dim model). PCC=0.369 indicates genuinely wrong computation, not accumulated BF16 rounding.

The most likely source is the multi-scale deformable attention (MSDA) in the Mask2FormerPixelDecoder (pixel decoder). MSDA calls `F.grid_sample(..., mode="bilinear", padding_mode="zeros", align_corners=False)` which is decomposed via `aten.grid_sampler_2d` into advanced indexing + bilinear interpolation (`a[N_idx, C_idx, idx_y, idx_x] * w`). This decomposition lowers to `stablehlo.gather` operations. If the gather lowering is incorrect for these specific shapes, the pixel decoder features — which are the keys/values for all transformer decoder layers — would be wrong, cascading through all 9 decoder layers.

The MSDA feature shapes: batch=1, num_queries=1024+256+64+64 (multi-scale flat), C=256, and sampling across 3 feature levels. These are non-trivial shapes that may exercise gather edge cases.

The dynamic boolean attention mask used in the transformer decoder's masked cross-attention is a secondary suspect (bool mask → float -inf conversion via `masked_fill_`), but this pattern is commonly used across many passing models.

## Fix
**Loader fix (committed):** `tt_forge_models` commit `029efb4d06` — added `use_fast=False` to `AutoImageProcessor.from_pretrained` call in `mask2former/panoptic_segmentation/pytorch/loader.py`.

**Proposed compiler fix:** Investigate whether `stablehlo.gather` produced from the `aten.grid_sampler_2d` decomposition returns incorrect results for the specific index tensors and shapes used in MSDA. The fix would likely be in `tt-mlir`'s gather lowering or in the `decompositions.py` decomposition of `grid_sampler_2d`. To isolate, run the Mask2Former pixel decoder in isolation against CPU reference and compare outputs per layer.

## Tier B justification
Indicator: **cross-cutting** — the root cause (wrong gather computation in the grid_sampler_2d decomposition) requires investigation to confirm (vs. other candidates: Swin window attention, masked cross-attention). Even once confirmed, a fix would span the decomposition (tt-xla), MLIR lowering (tt-mlir), and potentially the kernel (tt-metal). No single scoped change in one function addresses the issue.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    1297.91s (0:21:37)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/mask2former/panoptic_segmentation/pytorch/loader.py` — added `use_fast=False`
- `tt-xla/tests/runner/utils/dynamic_loader.py` — removed `sys.path.insert(models_root)` that shadowed spacy package

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 9982fac88249b56efafea3a95829d2200197e1be |
| tt-forge-models | 029efb4d0668871564be6d9787ca73927356af3a |
