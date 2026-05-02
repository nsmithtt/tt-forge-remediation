# Remediation Summary: oneformer-pytorch-Swin_Tiny_ADE20k-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[oneformer/pytorch-Swin_Tiny_ADE20k-single_device-inference]

## Result
FAIL — stablehlo.gather with 2D start_index_map and partial slices in both indexed dims has no supported TTNN lowering

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
stablehlo-gather-2d-partial-slice-no-lowering

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original reported failure (transformers 5.x warning):
```
The image processor of type `OneFormerImageProcessor` is now loaded as a fast processor by default,
even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce
slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.
```

After loader fix, second failure (PJRT kInternal error code 13):
```
ValueError: Error code: 13
```
Root cause traced to:
```
loc("gather.83"): error: failed to legalize operation 'stablehlo.gather'
module_builder.cc:889: Failed to convert from SHLO to TTIR module
```

## Root cause

Three issues were found and two were fixed:

**Issue 1 (loader, FIXED):** OneFormerProcessor was loaded without `use_fast=False`, causing a
transformers 5.x breaking change warning that propagated as an error. Fixed by adding `use_fast=False`
to the `OneFormerProcessor.from_pretrained()` call.

**Issue 2 (loader, FIXED):** `load_dataset("huggingface/cats-image")` triggered the `spacy` namespace
shadowing bug — `tt_forge_models/spacy/` shadows the real `spacy` package when `models_root` is
inserted into `sys.path`. Fixed by replacing `load_dataset` with `PIL.Image.new`.

**Issue 3 (tt-mlir, FIXED — Tier A):** `ElementTypeNormalization` pass converts `tensor<f64>` to
`tensor<f32>` but was not updating the `ttcore.local_shape` annotation on function result attrs.
The annotation retained the original f64 type, causing PJRT to prepare an 8-byte output buffer for
what the runtime actually wrote as 4-byte f32 data, producing kInternal error 13. Fixed by adding a
post-pass walk in `runOnOperation()` that updates `local_shape` element types to match the normalized
result types.

**Issue 4 (tt-mlir, Tier B — NOT FIXED):** OneFormer's pixel decoder (likely the feature pyramid
deformable attention mechanism) produces a `stablehlo.gather` with:
- Input: `tensor<1x64x64x256xf32>`
- Indices: `tensor<128x128x2xi32>`
- Output: `tensor<1x3x3x256x128x128xf32>` (rank 6)
- `dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [1, 2], index_vector_dim = 2>`
- `slice_sizes = [1, 3, 3, 256]`

The `StableHLOGatherToEmbeddingPattern` fails because `checkBasicLegality` requires at most one
partially-sliced indexed dim (`partialIndexedDims=2`, `remainingIndexedDims=2 != 1`). The
`StableHLOGatherToSliceRepeatConcatPattern` fails because it only handles `startIndexMap.size() == 1`.
No pattern handles a 2D spatial gather (multi-dim `start_index_map` with partial slices in both
indexed dims). This is a 2D sliding-window extraction (unfold-style op) for which no corresponding
TTNN primitive exists.

## Fix

**Loader fix** (`tt-xla/third_party/tt_forge_models`, `oneformer/pytorch/loader.py`):
- Added `use_fast=False` to `OneFormerProcessor.from_pretrained()`
- Replaced `load_dataset("huggingface/cats-image")` with `PIL.Image.new("RGB", (512, 512), color=(128, 128, 128))`
- Branch: `remediation/oneformer-pytorch-Swin_Tiny_ADE20k-single_device-inference`

**Tier A compiler fix** (`tt-mlir`, `lib/Dialect/TTIR/Transforms/ElementTypeNormalization.cpp`):
- Added post-normalization walk in `runOnOperation()` that updates `ttcore.local_shape` element types
  on function results to match their normalized result types (e.g., f64→f32, i64→i32)
- Branch: `remediation/oneformer-pytorch-Swin_Tiny_ADE20k-single_device-inference`

**Proposed fix for Tier B gather bug** (would live in `tt-mlir`,
`lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`):
- A new `StableHLOGatherTo2DSpatialPattern` that handles `startIndexMap.size() >= 2` with partial
  slices in multiple indexed dims. Would need to be lowered to TTNN slice+concat decomposition
  (expensive for large index grids) or require a new TTNN gather primitive that accepts multi-dim
  index maps.

## Tier B justification

new-infrastructure: The 2D spatial gather (multi-dim `start_index_map` with partial slice sizes in
both indexed dims) has no corresponding TTNN primitive. Implementing it requires either: (a) generating
O(H×W)=16384 individual slice operations which is impractical at 128×128 index resolution, or (b)
a new TTNN gather op that accepts multidimensional index maps, which does not currently exist.

## Verification
- pytest exit: FAIL
- Hardware: blackhole-p150b
- Duration: ~126s (to failure)
- Tier A attempts: 1

## Files changed
- `tt-xla/third_party/tt_forge_models/oneformer/pytorch/loader.py` — `use_fast=False`, replace `load_dataset` with PIL
- `tt-mlir/lib/Dialect/TTIR/Transforms/ElementTypeNormalization.cpp` — update `local_shape` attrs after type normalization

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | a5c9d3d94392a1d54211824076e281b515837009 |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 5d2566a1ebc0dcb23823262f668b167bcdecf4ba |
