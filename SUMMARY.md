# Remediation Summary: mask2former-panoptic_segmentation-pytorch-Swin_Large_Mapillary_Vistas-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mask2former/panoptic_segmentation/pytorch-Swin_Large_Mapillary_Vistas-single_device-inference]

## Result
FAIL — Tier B PCC=0.679 (required 0.99) after loader fixes; same multi-scale deformable attention stablehlo.gather compiler bug as Swin_Tiny_Cityscapes variant

## Stack layer
loader, tt-mlir

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
Original failure:
```
The image processor of type `Mask2FormerImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.
```

After loader fix 1 (use_fast=False):
```
AttributeError: module 'spacy' has no attribute 'Language'
```

After loader fix 2 (PIL.Image replacement):
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.6785098996034006. Required: pcc=0.99.
```

## Root cause
Two loader bugs and one Tier B compiler bug:

1. **Loader bug 1** (`tt_forge_models/mask2former/panoptic_segmentation/pytorch/loader.py`): `AutoImageProcessor.from_pretrained()` now defaults to fast processor in transformers 5.x. Added `use_fast=False`.

2. **Loader bug 2** (same file): `load_dataset("huggingface/cats-image")` triggers `datasets._dill` to pickle `spacy.Language`, but the `tt_forge_models/spacy/` directory is a namespace package that shadows the real `spacy` module when `models_root` is on `sys.path` (via `dynamic_loader.py`). Replaced with `PIL.Image.new("RGB", (480, 640), color=(128, 128, 128))`.

3. **Compiler bug (Tier B)**: After loader fixes pass, PCC=0.679 (required 0.99). This is not a BF16 precision floor (which would be ≥0.86). The `Mask2FormerPixelDecoder` uses `F.grid_sample` in multi-scale deformable attention (MSDA), which decomposes via `aten.grid_sampler_2d` → advanced indexing + bilinear weights → `stablehlo.gather`. The gather produces incorrect results for these shapes, propagating errors through all decoder layers. This is the same bug seen in the Swin_Tiny_Cityscapes variant (PCC=0.369) and is Tier B.

## Fix
- **Loader fixes** committed in `tt_forge_models` on branch `remediation/mask2former-panoptic_segmentation-pytorch-Swin_Large_Mapillary_Vistas-single_device-inference`:
  - Commit 1: `8a5b37fa37` — `use_fast=False` in `_load_image_processor`
  - Commit 2: `6ddb5fb5dd` — Replace `load_dataset` with `PIL.Image.new`

- **Proposed compiler fix** (not attempted, Tier B): The `stablehlo.gather` lowering for the bilinear interpolation in `F.grid_sample` / `aten.grid_sampler_2d` produces incorrect results. The fix would require implementing correct grid sampling in tt-mlir (new infrastructure, multi-file change).

## Tier B justification
- **new-infrastructure**: `F.grid_sample` decomposition to bilinear interpolation via `stablehlo.gather` requires new or significantly reworked lowering passes in tt-mlir.
- Likely cross-cutting: affects any model using bilinear `grid_sample` (deformable attention, spatial transformer networks).

## Verification
- pytest exit: FAIL
- Hardware: blackhole-p150b
- Duration: 1560.59s (0:26:00)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/mask2former/panoptic_segmentation/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | df3d14f769d39d7801ebfaf01f324186ff78b1e3 |
| tt-forge-models | 6ddb5fb5dd027ffaa58d14fa3a4173e081c2e8f4 |
