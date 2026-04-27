# Remediation Summary: code_formula_v2-ocr-pytorch-code_formula_v2-single_device-inference

## Skill version
11

## Test
tests/runner/test_models.py::test_all_models_torch[code_formula_v2/ocr/pytorch-code_formula_v2-single_device-inference]

## Result
FAIL — StableHLO gather with 786432-element rows is lowered to a TTNN embedding op whose circular buffer (3 MB) exceeds Wormhole L1 size (1.5 MB)

## Failure
```
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

Full Python traceback:
```
transformers/models/idefics3/modeling_idefics3.py:596: in get_image_features
    pixel_values = pixel_values[real_images_inds].contiguous()
transformers/models/idefics3/modeling_idefics3.py:596: in torch_dynamo_resume_in_get_image_features_at_596
    pixel_values = pixel_values[real_images_inds].contiguous()
python_package/tt_torch/backend/backend.py:225: in __call__
    return self._call_experimental_compile(*args)
torch_xla/_dynamo/dynamo_bridge.py:859: in extract_compiled_graph_helper
    return partition_fx_graph_for_cpu_fallback(...)
torch_xla/_dynamo/dynamo_bridge.py:346: in extract_graph_helper
    torch_xla.sync(reset_scope=False)
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

Relevant C++ stack at failure:
```
 --- tt::runtime::ttnn::operations::embedding::run(tt::target::ttnn::EmbeddingOp const*, ...)
 --- ttnn::embedding(...)
 --- ttnn::prim::embedding(...)
 --- EmbeddingsDeviceOperation::launch
 --- create_and_cache_mesh_workload
 --- enqueue_mesh_workload
 --- MeshWorkloadImpl::compile
```

## Root cause

**Layer: tt-mlir** (StableHLO→TTIR conversion, `StableHLOGatherToEmbeddingPattern`)

The Idefics3 model's `get_image_features` method contains a padding-image filter:

```python
real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image
pixel_values = pixel_values[real_images_inds].contiguous()   # line 596
```

`pixel_values` has shape `[1, 3, 512, 512]` (1 image, 3 channels, 512×512). The boolean index `real_images_inds` selects along dim 0, keeping only non-all-zero images. With a single real image, `real_images_inds = [True]` and the operation is a runtime no-op — but the compiler must still compile it.

The compilation pipeline lowers `pixel_values[real_images_inds]` as:

1. **XLA/StableHLO**: boolean index → `stablehlo.gather`
2. **tt-mlir** (`StableHLOGatherToEmbeddingPattern`): gather → TTIR/TTNN embedding op

`StableHLOGatherToEmbeddingPattern::checkBasicLegality` passes for this gather because:
- `startIndexMap = [0]`, `inputShape[0] = 1` → counted as `singletonIndexedDims = 1`
- All other slice sizes equal the full dimension → legality checks pass

The pattern reshapes `pixel_values [1, 3, 512, 512]` into an embedding table `[1, 786432]`, where:
- **embed_dim = 3 × 512 × 512 = 786,432**
- **weight_page_size = 786,432 × 2 (bfloat16) = 1,572,864 bytes = 1.5 MB**
- **out_cb_size = 2 × 1,572,864 = 3,145,728 bytes = 3 MB**

Wormhole L1 size per core is 1,572,864 bytes (1.5 MB). The output circular buffer (3 MB) exceeds L1, causing `validate_circular_buffer_region` to throw inside `MeshWorkloadImpl::compile`. This exception is caught by the PJRT plugin and wrapped as "Bad StatusOr access: INTERNAL: Error code: 13".

## Fix
**Proposed fix in tt-mlir** (`lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`):

Add a row-size guard to `StableHLOGatherToEmbeddingPattern::checkBasicLegality`. After computing `reshapedInput` shape, check that the embedding row size (product of all non-indexed dimensions) does not exceed a safe L1 threshold (e.g., half of Wormhole L1 = 786,432 bytes). If the row exceeds this threshold, `notifyMatchFailure` so the gather falls back to an alternative lowering (CPU fallback or scatter/concat decomposition).

Concretely, before line 4687 in `StableHLOToTTIRPatterns.cpp`:

```cpp
// Guard: reject if embedding row size exceeds half L1 (786432 bytes = 393216 elements bfloat16)
// Large rows cause out_cb_size > L1 in EmbeddingsRMProgramFactory.
const int64_t maxRowElements = 393216;  // conservative L1 limit
int64_t rowSize = 1;
for (size_t i = 0; i < inputShape.size(); ++i) {
  if (!llvm::is_contained(startIndexMap, i))
    rowSize *= inputShape[i];
}
if (rowSize > maxRowElements) {
  return rewriter.notifyMatchFailure(
      srcOp, "Gather row size exceeds safe L1 circular buffer limit");
}
```

This change lives in tt-mlir and does not touch the model loader.

## Verification
FAIL — test not passing. No silicon run performed.

## Files changed
None (compiler-stack bug, no loader fix applied)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
