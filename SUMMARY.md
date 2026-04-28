# Remediation Summary: granite_4_0_h-causal_lm-pytorch-4.0_h_small_base-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[granite_4_0_h/causal_lm/pytorch-4.0_H_Small_Base-single_device-inference]

## Result
XFAIL — granite-4.0-h-small-base (hidden_size=4096, 40 hybrid Mamba/MoE layers) in BF16 exceeds single-device DRAM capacity; OOM occurs during inference after all model weights and compiled kernels are loaded

## Stack layer
hardware-class

## Tier
A

## Bug fingerprint
oom-model-exceeds-single-device-dram

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

Original failure (before Tier A fix): `TT_FATAL: Number of rows in gradient tensor must be equal to number of indices in index tensor` in `embedding_backward_device_operation.cpp:67`, triggered by rank-1 scatter indices passed without reshaping.

After Tier A fix: `TT_FATAL: Out of Memory: Not enough space to allocate 1073741824 B DRAM buffer across 8 banks, where each bank needs to store 134217728 B, but bank size is 4273390016 B (allocated: 4124054528 B, free: 149335488 B, largest free block: 121634944 B)` during `ttnn::sum` execution in `bank_manager.cpp:439`.

## Root cause

### Original compiler bug (Tier A — fixed)

XLA generates `stablehlo.scatter` with rank-1 indices `tensor<70xi64>` and `index_vector_dim=1` (= rank, the scalar-index form) for `torch.index_add` with 1D integer indices inside the MoE layer of Granite 4.0H. The `StableHLOToTTIREmbeddingBackwardOpConversionPattern` in `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp` only handled the rank-2 case `[N, 1]` with `index_vector_dim=1`, inserting a reshape to `[1, N, 1]`. The rank-1 case `[N]` with `index_vector_dim=rank` was not handled — indices passed unchanged to `ttir::EmbeddingBackwardOp`. Downstream in `TTIRToTTNNEmbeddingBackwardOp`, the 1D index tensor `[N]` is interpreted with `batch_size=N, sentence_size=N`, causing the assertion `grad_tensor_shape[2] == index_tensor_shape[0] * index_tensor_shape[-1]` to evaluate `N == N*N` → false.

### Hardware capacity ceiling

After the scatter fix was applied, the test ran for ~58 minutes (3513 seconds), completing compilation of all 40 layers and beginning inference execution. At that point, 4124054528 B (3.84 GB out of ~3.98 GB) of DRAM was already allocated for model weights and compiled program buffers. A `ttnn::sum` (reduce) operation tried to allocate a 1 GB output tensor, leaving only 142 MB free — insufficient. This is a genuine hardware capacity ceiling: `ibm-granite/granite-4.0-h-small-base` has `hidden_size=4096` across 40 hybrid Mamba+MoE layers and in BF16 requires more DRAM than a single wormhole device provides.

## Fix

### Tier A fix in tt-mlir (applied)

**File**: `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp` (lines 5635–5643)

Added a branch to `StableHLOToTTIREmbeddingBackwardOpConversionPattern::matchAndRewrite` for rank-1 scatter indices with `indexVectorDim == rank`:

```cpp
} else if (scatterIndicesType.getRank() == 1 &&
           indexVectorDim == scatterIndicesType.getRank()) {
  // [N] with indexVectorDim=rank (scalar indices) → [1, N]
  llvm::SmallVector<int64_t> newShape{1, scatterIndicesType.getDimSize(0)};
  scatterIndices = ttir::utils::createReshapeOp(rewriter, srcOp.getLoc(),
                                                scatterIndices, newShape);
}
```

A test case `@test_rank1_scalar_indices` was added to `tt-mlir/test/ttmlir/Conversion/StableHLOToTTIR/scatter_op_to_embedding_bw.mlir`.

Branch: `tenstorrent/tt-mlir@remediation/granite_4_0_h-causal_lm-pytorch-4.0_h_small_base-single_device-inference`

### XFAIL marking in tt-xla (applied)

`tests/runner/test_config/torch/test_config_inference_single_device.yaml` updated to mark `granite_4_0_h/causal_lm/pytorch-4.0_H_Small_Base-single_device-inference` as `KNOWN_FAILURE_XFAIL`.

Branch: `tenstorrent/tt-xla@remediation/granite_4_0_h-causal_lm-pytorch-4.0_h_small_base-single_device-inference`

## Verification
- pytest exit: FAIL (OOM after Tier A fix; original INTERNAL error resolved)
- Hardware:    n150
- Duration:    3513.54s (0:58:33) — with Tier A fix applied
- Tier A attempts: 1

## Files changed
- `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`
- `tt-mlir/test/ttmlir/Conversion/StableHLOToTTIR/scatter_op_to_embedding_bw.mlir`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 1d7853791cf2dc3ea0832b12a18ee2d092431afb |
| tt-xla          | 86231eda5a81fc35d5ec0f8bb18c82404a3f7ec1 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
