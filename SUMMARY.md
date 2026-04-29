# Remediation Summary: deberta_v2-seq_cls-pytorch-scaleTech_nsfw_classifier-single_device-inference

## Skill version
6

## Test
`tests/runner/test_models.py::test_all_models_torch[deberta_v2/seq_cls/pytorch-scaleTech_nsfw_classifier-single_device-inference]`

## Result
SILICON_PASS

## Stack layer
tt-mlir

## Tier
A

## Bug fingerprint
gather-embedding-f32-indices-crash

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
python3: /opt/ttmlir-toolchain/include/llvm/ADT/ArrayRef.h:253: const T &llvm::ArrayRef<long>::operator[](size_t) const [T = long]: Assertion `Index < Length && "Invalid index!"' failed.
Fatal Python error: Aborted
```

Crash occurs during MLIR compilation in `extract_graph_helper` (dynamo_bridge.py:483) while compiling the DeBERTa-v2 `forward` method (modeling_deberta_v2.py:1039).

## Root cause

DeBERTa-v2 uses disentangled attention with 3D gather indices (shape `[12, 512, 512, 3]`, indexing a `[12, 512, 512]` tensor across all three dimensions). `StableHLOGatherToEmbeddingPattern` handles multi-dim gather (`numIndexingDims > 1`) by calling `flattenStartIndices`, which:

1. Typecasts `ui32` indices → `f32` to enable matmul
2. Multiplies by row-major stride constants `[262144, 512, 1]` via matmul
3. Returns the `f32` result directly

Two bugs:

**Bug 1 — f32 indices to embedding (crash):** The `f32` result is used as the `input` (index) argument to `ttir.embedding` without casting back to integer. When TTIR→TTNN lowering runs, `ttnn.embedding` receives an f32 index tensor, reinterprets the bits as int64, and triggers an OOB assertion in `ArrayRef<long>::operator[]` → SIGSEGV.

**Bug 2 — FP16 accumulation (precision):** TT hardware matmul uses FP16 intermediates. Stride products (e.g. `511 × 512 = 261632`) exceed FP16's exact integer range (2^11 = 2048), causing off-by-1 errors in the flattened index.

## Fix

Both bugs fixed in `StableHLOToTTIRPatterns.cpp` (`StableHLOGatherToEmbeddingPattern`):

**Fix 1 — cast back to integer at call site:** After `flattenStartIndices` returns the sum result (f32), insert a `ttir::TypecastOp` converting from f32 back to the original integer element type of `startIndices` (e.g. `ui32`). This ensures `ttir.embedding` always receives integer indices.

**Fix 2 — replace matmul with multiply+reduce-sum:** Changed `flattenStartIndices` to use element-wise multiply followed by reduce-sum over the last dimension instead of a float matmul. The strides constant is reshaped to `[1, ..., 1, numIndexingDims]` for broadcast. Element-wise f32 multiply is exact for values in this range; reduce-sum uses f32 accumulation (exact up to 2^24 = 16,777,216, well above the max flat index of 3,145,727).

## Verification
- pytest exit: PASS
- Hardware: n150
- Duration: 66.35s
- Tier A attempts: 1

## Files changed
- `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`
  — `flattenStartIndices`: replace matmul with multiply+reduce-sum (fix FP16 precision)
  — `matchAndRewrite`: add TypecastOp from f32 → original integer type after
    `flattenStartIndices` call (fix embedding index dtype crash)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 4d223e1aeff99a40538b2d560f836d9f74360a83 |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
