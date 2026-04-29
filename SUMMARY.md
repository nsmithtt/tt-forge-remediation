# Remediation Summary: deberta-zero_shot_cls-pytorch-V3_Base_Zeroshot_v2.0-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deberta/zero_shot_cls/pytorch-V3_Base_Zeroshot_v2.0-single_device-inference]

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
Process crash (SIGABRT) during torch.compile compilation of DeBERTa-v3-base-zeroshot-v2.0:

```
python: /opt/ttmlir-toolchain/include/llvm/ADT/ArrayRef.h:253:
const T &llvm::ArrayRef<long>::operator[](size_t) const [T = long]:
Assertion `Index < Length && "Invalid index!"' failed.
Fatal Python error: Aborted
```

Stack trace: `dynamo_bridge.py:483 extract_graph_helper` ->
`partition_fx_graph_for_cpu_fallback` -> MLIR compilation ->
`SharedLHSMatmulFusion<LinearOp>` -> `ArrayRef::operator[]` OOB -> SIGABRT.

## Root cause
Two bugs in tt-mlir, both in the DeBERTa-v2 disentangled attention path
(identical root cause to the previously reported
`deberta/question_answering` failure):

**Bug 1 (primary crash): `SharedLHSMatmulFusion<LinearOp>` in
`tt-mlir/lib/Dialect/TTIR/Transforms/TTIRFusing.cpp`**
crashes when the fusion candidate set contains `LinearOp`s with different
output ranks. `collectCandidates` checked that all LHS-sharing `LinearOp`s
share the same RHS rank but not that their output ranks agree with the root
op. DeBERTa-v3 disentangled attention mixes rank-3 projections with rank-2
ops sharing the same LHS. `replaceWithSlices` then accesses
`shape[outputFusedDim]` where `outputFusedDim = rootOutputRank - 1 = 2` but
the rank-2 candidate has only indices 0 and 1, triggering the OOB assertion.

**Bug 2 (secondary): `StableHLOGatherToEmbeddingPattern.flattenStartIndices`
in `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`**
returned f32 from its matmul stride computation but `ttir.embedding` requires
integer indices. TTNN reinterpreted f32 bits as int64, producing garbage
index values and OOB embedding lookups.

## Fix
Cherry-picked two commits from the remediation branch for the sibling
`deberta/question_answering` failure onto a new remediation branch in tt-mlir:

**Fix 1** - commit `0fa4e5a4b` (cherry-pick of `1a5152c3a`):
`tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`:
Cast the result of `flattenStartIndices` back to the original integer
element type and replaced the matmul-based stride computation with
element-wise multiply + reduce-sum to avoid FP16 precision loss.

**Fix 2** - commit `1b339b35b` (cherry-pick of `e79290fbf`):
`tt-mlir/lib/Dialect/TTIR/Transforms/TTIRFusing.cpp`:
Added output-rank guard in `collectCandidates` to skip any candidate
whose result rank differs from the root op's result rank.

Remediation branch: `remediation/deberta-zero_shot_cls-pytorch-V3_Base_Zeroshot_v2.0-single_device-inference`
in tt-mlir (commits `0fa4e5a4b` and `1b339b35b`).

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    99.85s (0:01:39)
- Tier A attempts: 1

## Files changed
- `tt-mlir/lib/Dialect/TTIR/Transforms/TTIRFusing.cpp`
- `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 1b339b35b0a05c21a6f3cc4c74f2b1da42a5c68a |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
