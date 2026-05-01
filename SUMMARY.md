# Remediation Summary: deberta-reward_model-pytorch-V3_Large_V2-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deberta_v2/seq_cls/pytorch-henokyemam_llama_guard_safegate_ss_august15-single_device-inference]

## Result
SILICON_PASS

## Stack layer
tt-mlir

## Tier
A

## Bug fingerprint
ttir-shared-lhs-matmul-fusion-output-rank-oob

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
Crash in `partition_fx_graph_for_cpu_fallback` -> `extract_graph_helper` within `dynamo_bridge.py` while compiling `modeling_deberta_v2.py:1039 forward`.

## Root cause
`SharedLHSMatmulFusion<LinearOp>::collectCandidates` in `TTIRFusingPass` (`tt-mlir/lib/Dialect/TTIR/Transforms/TTIRFusing.cpp`) collected all ops that share a LHS operand and have matching RHS rank, but did **not** check that candidate output rank matches the root output rank.

`MatmulWithBiasFusionPattern` can produce `LinearOp`s with varying output ranks when a bias tensor is shared across multiple adds and bias peeling stops early (broadcast shape exceeds matmul rank). When DeBERTa-V3's shared Q/K/V projections produce a mix of rank-2 and rank-3 `LinearOp`s in the same fusion group, `replaceWithSlices` accesses `shape[rootOutputRank-1]` on a candidate whose output rank is lower, triggering the LLVM `ArrayRef::operator[]` out-of-bounds assertion and SIGABRT.

## Fix
Added an output-rank equality guard in `collectCandidates` (after the batch-dimensions check):

```cpp
auto rootOutputType = mlir::cast<RankedTensorType>(rootOp.getType());
int64_t rootOutputRank = rootOutputType.getRank();
// ...
auto outputType = mlir::cast<RankedTensorType>(op.getType());
if (outputType.getRank() != rootOutputRank) {
    continue;
}
```

`rootOutputRank` is computed from `rootOp.getType()` at the top of `collectCandidates`. Candidates with mismatched output rank are simply excluded from the fusion group instead of crashing.

**Repo:** `tt-mlir`
**Branch:** `remediation/deberta-reward_model-pytorch-V3_Large_V2-single_device-inference`
**File:** `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp`
**Commit:** `b12c792a9`

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    108.93s (0:01:48)
- Tier A attempts: 1

## Files changed
- `tt-mlir/lib/Dialect/TTIR/Transforms/TTIRFusing.cpp` — output-rank guard in `collectCandidates`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | b12c792a9ebee12b3b5a403d6b50fcf34d420998 |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
