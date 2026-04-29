# Remediation Summary: cross_encoder-nli-pytorch-nli-deberta-v3-xsmall-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[cross_encoder/nli/pytorch-nli-deberta-v3-xsmall-single_device-inference]

## Result
SILICON_PASS — fixed SharedLHSMatmulFusion mixed-rank OOB crash in TTIRFusing.cpp

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
```
python: /opt/ttmlir-toolchain/include/llvm/ADT/ArrayRef.h:253: const T &llvm::ArrayRef<long>::operator[](size_t) const [T = long]: Assertion `Index < Length && "Invalid index!"' failed.
```

SIGABRT in `replaceWithSlices` during `SharedLHSMatmulFusion` rewrite. DeBERTa-v3 disentangled attention creates LinearOps where some candidates have 2-D outputs while the root op has a 3-D output. The fused-output-dimension index, derived from the root op rank, exceeded the rank of a 2-D candidate, triggering an out-of-bounds assertion inside `replaceWithSlices`.

## Root cause
In `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp`, `collectCandidates` for `SharedLHSMatmulFusion` verified that candidates share the same LHS and have the same RHS rank, but did not check that the candidate's result rank matches the root op's result rank. DeBERTa-v3's disentangled attention introduces `ttir.linear` calls that share the same LHS but have different output ranks (2-D vs 3-D). When `replaceWithSlices` indexed the fused output using the root op's dimension count, it overflowed the candidate op's lower-rank result, triggering the LLVM ArrayRef assertion (SIGABRT).

## Fix
Added an output-rank guard in `collectCandidates` in `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp` (tt-mlir commit `bec72757a`). The guard skips any candidate whose result rank differs from the root op's result rank, preventing mixed-rank candidates from entering the same fusion group.

The fix lives on branch `remediation/cross_encoder-nli-pytorch-nli-deberta-v3-xsmall-single_device-inference` in tt-mlir.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    66.28s (0:01:06)
- Tier A attempts: 1

## Files changed
- `tt-mlir/lib/Dialect/TTIR/Transforms/TTIRFusing.cpp` (+9 lines, output-rank guard in `collectCandidates`)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | bec72757a1e17219bf6e99902aaff29de49a6b69 |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
