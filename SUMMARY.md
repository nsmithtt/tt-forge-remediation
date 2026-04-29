# Remediation Summary: deberta-question_answering-pytorch-deepset_deberta_v3_large_squad2-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deberta/question_answering/pytorch-deepset_deberta_v3_large_squad2-single_device-inference]

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
Process crash (SIGABRT) during torch.compile compilation of DeBERTa-v3-large:

```
Extension modules: numpy._core._multiarray_umath, ...  (total: 222)
```

Stack trace shows crash in `dynamo_bridge.py:483 in extract_graph_helper` →
`partition_fx_graph_for_cpu_fallback` → MLIR compilation.

## Root cause
`SharedLHSMatmulFusion<LinearOp>` in `tt-mlir/lib/Dialect/TTIR/Transforms/TTIRFusing.cpp`
crashes with `Assertion 'Index < Length && "Invalid index!"' failed` (SIGABRT) when
the fusion candidate set contains `LinearOp`s with different output ranks.

`collectCandidates` checked that all LHS-sharing `LinearOp`s have the same RHS rank
but not that their *output* ranks agree with the root op. DeBERTa-v3 disentangled
attention mixes rank-3 attention projections (input `[1,128,1024]`) with rank-2 ops
sharing the same LHS. `replaceWithSlices` then accesses `shape[outputFusedDim]` where
`outputFusedDim = rootOutputRank - 1 = 2` but the rank-2 candidate's shape has only
2 elements (indices 0 and 1), triggering the out-of-bounds assertion.

## Fix
Added output-rank guard in `collectCandidates` in
`tt-mlir/lib/Dialect/TTIR/Transforms/TTIRFusing.cpp` (tt-mlir commit `e27855f2e`,
cherry-picked from `bec72757a` on branch
`remediation/deberta-question_answering-pytorch-deepset_deberta_v3_large_squad2-single_device-inference`):

```cpp
int64_t rootOutputRank =
    mlir::cast<RankedTensorType>(rootOp.getType()).getRank();
// Output rank must match the root op so replaceWithSlices can index the
// fused output dimension uniformly across all candidates.
if (mlir::cast<RankedTensorType>(op.getType()).getRank() != rootOutputRank) {
  continue;
}
```

This skips any candidate whose output rank differs from the root op, preventing the
out-of-bounds access in `replaceWithSlices`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    245.62s (0:04:05)
- Tier A attempts: 1

## Files changed
- `tt-mlir/lib/Dialect/TTIR/Transforms/TTIRFusing.cpp`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | e27855f2e30bb7c09a3b623db8923da4409c3873 |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
