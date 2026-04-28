# Remediation Summary: idefics2-conditional_generation-pytorch-tiny_idefics2-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[idefics2/conditional_generation/pytorch-tiny_idefics2-single_device-inference]

## Result
FAIL — ttnn.embedding CB overflow: boolean-indexing pixel_values row size (4.3 MB) exceeds Blackhole L1 (1.5 MB)

## Stack layer
tt-metal

## Tier
B

## Bug fingerprint
embedding-cb-overflow-row-size-exceeds-l1

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The stated failure message is a transformers 5.x `UserWarning`:
> The image processor of type `Idefics2ImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.

After applying `use_fast=False` (loader fix, committed), the test fails with a Tier B compiler-stack error:
```
TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=0,y=4)]
grow to 4433472 B which is beyond max L1 size of 1572864 B
```
from `ttnn::prim::embedding` via `tt::runtime::ttnn::operations::embedding::run`.

## Root cause
The idefics2 model (`modeling_idefics2.py:849-850`) performs boolean tensor indexing to filter out zero-padded image tiles:
```python
real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image
pixel_values = pixel_values[real_images_inds].contiguous()
```

With the default `Idefics2ImageProcessor` configuration (`longest_edge=980`) and the test image (4032×3024 candy.JPG), the image is resized and tiled into 5 tiles of shape 980×735. The processor output has shape `(1, 5, 3, 980, 735)`.

The TT compiler lowers the boolean indexing as an `ttnn.embedding` operation:
- Embedding table: `pixel_values` reshaped to `(5, 2160900)` where 2160900 = 3×980×735
- Each embedding row: 2160900 × 2 bytes (bf16) = **4,321,800 bytes ≈ 4.1 MB**
- Blackhole L1 size: **1,572,864 bytes = 1.5 MB**

The embedding kernel allocates circular buffers (CBs) sized for one row. At 4.1 MB per row, the CBs grow to 4,433,472 bytes — nearly 3× the L1 limit — causing a fatal `TT_THROW`.

Additionally, the same model code contains `pixel_values == 0.0` where `0.0` is a Python float64 literal, producing a `stablehlo.constant dense<0.0> : tensor<5x3x980x735xf64>` and a `stablehlo.convert bf16→f64` in the StableHLO graph. The `ElementTypeNormalization` pass in tt-mlir demotes f64→f32 at the TTIR level, so this graph itself compiles and executes successfully. The fatal failure is downstream in the boolean indexing gather.

The `use_fast=False` loader fix was applied and committed; both slow and fast processors produce identical output shapes `(1, 5, 3, 980, 735)`, so the loader fix does not resolve the CB overflow.

## Fix
**Applied (loader)**: Added `use_fast=False` to `AutoProcessor.from_pretrained()` in `idefics2/conditional_generation/pytorch/loader.py` to silence the transformers 5.x breaking-change warning and use the intended slow processor.

**Proposed (compiler stack)**: The `ttnn::prim::embedding` kernel (tt-metal) should detect when the embedding row size exceeds L1 and fall back to a different lowering — for example, a DRAM-to-DRAM copy path or a chunked tile-streaming approach. Alternatively, tt-mlir could detect this pattern (gather on a reshaped image tensor with row_size > L1) during lowering and emit a different TTNN operation that is not bound by the L1 single-row constraint.

## Tier B justification
**Indicator: new-infrastructure**

The `ttnn::prim::embedding` kernel allocates one L1 CB per row and has no code path for rows that exceed L1 size. Supporting large-row embedding would require implementing a blocked/streaming approach in the embedding kernel and its dispatch code, which is new kernel infrastructure. Alternatively, the tt-mlir boolean-indexing lowering would need a new code path that routes to a non-embedding gather implementation when row_size > L1; this cross-cuts embedding lowering and gather lowering. Neither fits in the "single scoped pattern fix" Tier A threshold.

## Verification
- pytest exit: FAIL
- Hardware: blackhole-p150b
- Duration: ~32s
- Tier A attempts: N/A

## Files changed
- `idefics2/conditional_generation/pytorch/loader.py` (tt_forge_models — add `use_fast=False`)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 5cae9217a968a6f4c1f2b33a55e276b6826d1a51 |
| tt-forge-models | f3617c75b46bb10779399a3dc240231727aa1cb9 |
