# Remediation Summary: deberta-question_answering-pytorch-deepset_deberta_v3_base_squad2-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deberta/question_answering/pytorch-deepset_deberta_v3_base_squad2-single_device-inference]

## Result
SILICON_PASS

## Stack layer
tt-mlir

## Tier
A

## Bug fingerprint
shared-lhs-matmul-fusion-mixed-rank-oob

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Process crash (SIGABRT) during torch.compile compilation of DeBERTa-v3-base:

```
Extension modules: numpy._core._multiarray_umath, ...  (total: 222)
```

Stack trace shows crash in `dynamo_bridge.py:483 in extract_graph_helper` →
`partition_fx_graph_for_cpu_fallback` → MLIR compilation.

## Root cause
Two bugs in tt-mlir, both in the DeBERTa-v2 disentangled attention path:

**Bug 1 (primary crash): `SharedLHSMatmulFusion<LinearOp>` in
`tt-mlir/lib/Dialect/TTIR/Transforms/TTIRFusing.cpp`**
crashes with `Assertion 'Index < Length && "Invalid index!"' failed` (SIGABRT)
when the fusion candidate set contains `LinearOp`s with different output ranks.
`collectCandidates` checked that all LHS-sharing `LinearOp`s have the same RHS
rank but not that their *output* ranks agree with the root op. DeBERTa-v3
disentangled attention mixes rank-3 attention projections (input `[1,384,768]`)
with rank-2 ops sharing the same LHS. `replaceWithSlices` then accesses
`shape[outputFusedDim]` where `outputFusedDim = rootOutputRank - 1 = 2` but
the rank-2 candidate's shape has only 2 elements (indices 0 and 1), triggering
the out-of-bounds assertion.

**Bug 2 (secondary): `StableHLOGatherToEmbeddingPattern.flattenStartIndices`
in `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`**
had two problems for the multi-dim gather path (numIndexingDims > 1) used by
DeBERTa-v2 disentangled attention:
1. `flattenStartIndices` returned f32 from its matmul but `ttir.embedding`
   requires integer indices; TTNN reinterpreted f32 bits as int64, producing
   values like 1162027008 instead of 3072, causing OOB crashes.
2. FP16 accumulation error in the TT matmul: strides like 512 and 262144
   caused accumulated products (e.g. 511×512=261632) to exceed the FP16
   representable integer range, giving off-by-one index errors.

## Fix
Two fixes in `tt-mlir` on branch
`remediation/deberta-question_answering-pytorch-deepset_deberta_v3_base_squad2-single_device-inference`
(commits `1a5152c3a` and `e79290fbf`):

**Fix 1** — `tt-mlir/lib/Dialect/TTIR/Transforms/TTIRFusing.cpp`
(commit `e79290fbf`): Added output-rank guard in `collectCandidates` to skip any
candidate whose result rank differs from the root op's result rank, preventing
the OOB access in `replaceWithSlices`:
```cpp
int64_t rootOutputRank =
    mlir::cast<RankedTensorType>(rootOp.getType()).getRank();
// Output rank must match the root op so replaceWithSlices can index the
// fused output dimension uniformly across all candidates.
if (mlir::cast<RankedTensorType>(op.getType()).getRank() != rootOutputRank) {
  continue;
}
```

**Fix 2** — `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`
(commit `1a5152c3a`): Cast the result of `flattenStartIndices` back to the
original integer element type (instead of leaving it as f32 after the matmul),
and replaced the matmul-based stride computation with element-wise
multiply + reduce-sum to avoid FP16 precision loss at large stride values.

## Verification
- pytest exit: PASS
- Hardware:    p150b
- Duration:    105.16s (0:01:45)
- Tier A attempts: 1

## Files changed
- `tt-mlir/lib/Dialect/TTIR/Transforms/TTIRFusing.cpp`
- `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | e79290fbfc868f883957c5d3482b891c3bc6da13 |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
