# Remediation Summary: graphormer_base_pcqm4mv2-graph_classification-pytorch-clefourrier-graphormer-base-pcqm4mv2-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[graphormer_base_pcqm4mv2/graph_classification/pytorch-clefourrier/graphormer-base-pcqm4mv2-single_device-inference]

## Result
SILICON_PASS — boolean index_put decomposed to torch.where, avoiding stablehlo.scatter with dynamic indices

## Stack layer
tt-xla

## Tier
A

## Bug fingerprint
boolean-index-put-stablehlo-scatter-dynamic-k

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
loc("scatter.23"): error: failed to legalize operation 'stablehlo.scatter'
ValueError: Error code: 13
```

The originally reported failure `/bin/bash: line 1: python: command not found` came from `silicon_validate.py:312` hardcoding `"python"` instead of `sys.executable`; that was fixed separately in `.hf-bringup/silicon_validate.py`.

## Root cause
`GraphormerGraphAttnBias.forward()` in `modeling_graphormer.py:277` contains:

```python
spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
```

This boolean-indexed in-place assignment traces to `aten.index_put_.default` in the FX graph.  `torch.export.export` with `strict=False` does not fully functionalize in-place ops — `run_decompositions` functionalizes it to `aten.index_put.default` before applying the custom decompositions table. Neither variant had a custom decomposition, so both fell through to XLA's default lowering: `stablehlo.scatter` with shape `[K, 3]` where `K` is dynamic (the number of zeros in the mask).

In `StableHLOToTTIRPatterns.cpp`, `flattenMultiDimScatterIndices` (line ~6066) does:
```cpp
llvm::SmallVector<int32_t> ends = {
    static_cast<int32_t>(expandedNumIndices),  // ShapedType::kDynamic → INT32_MIN-ish
    static_cast<int32_t>(idxPos + 1)};
```
When `expandedNumIndices == ShapedType::kDynamic`, the cast produces an invalid large-magnitude value used as a `SliceStaticOp` end bound, causing legalization failure.

## Fix
Added `_index_put_boolean` decomposition to `tt-xla/python_package/tt_torch/backend/decompositions.py`.  The function decomposes `index_put[_](self, [bool_mask], values)` → `torch.where(bool_mask, values.expand_as(self), self)`, which lowers to `stablehlo.select` (fully supported). Both the in-place (`index_put_.default`) and out-of-place (`index_put.default`) variants are registered because `run_decompositions` functionalizes the in-place form before applying the custom table.

File changed: `python_package/tt_torch/backend/decompositions.py`

Commit: `115991a79734ba27592466599fffe53f26b41a43` on branch `remediation/graphormer_base_pcqm4mv2-graph_classification-pytorch-clefourrier-graphormer-base-pcqm4mv2-single_device-inference` in tt-xla.

## Verification
- pytest exit: PASS
- Hardware: blackhole-p150b
- Duration: 93.07s (0:01:33)
- Tier A attempts: 2

## Files changed
- `tt-xla/python_package/tt_torch/backend/decompositions.py` — added `_index_put_boolean` and registered for both `aten.index_put_.default` and `aten.index_put.default`
- `.hf-bringup/silicon_validate.py` — fixed `"python"` → `sys.executable` (infrastructure fix, not in any submodule)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 115991a79734ba27592466599fffe53f26b41a43 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
