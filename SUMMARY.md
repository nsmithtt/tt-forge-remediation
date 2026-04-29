# Remediation Summary: deberta/masked_lm/pytorch-V3_Base-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deberta/masked_lm/pytorch-V3_Base-single_device-inference]

## Result
FAIL — PCC 0.9866 below required 0.99 after crash fix; second compiler-stack bug, not chaining Tier A fixes per skill rules

## Stack layer
tt-mlir

## Tier
A

## Bug fingerprint
shared-lhs-linear-fusion-output-rank-oob

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Process crashed with Python fault-handler dump (total: 223 extension modules listed) — a hard abort in the PJRT compilation subprocess. The original failure message excerpt:

```
Extension modules: numpy._core._multiarray_umath, numpy.linalg._umath_linalg, ...
torch._C, torch._C._dynamo.autograd_compiler, ... (total: 223)
```

The crash was reproduced and isolated to `SharedLHSMatmulFusion<LinearOp>` in the `ttir-fusing` pass:

```
ttmlir-opt --pass-pipeline="builtin.module(ttir-fusing)" /tmp/deberta_canonicalized.mlir
```
→ Aborted (core dumped) — LLVM `ArrayRef<long>::operator[]` out-of-bounds assertion in debug build.

After the Tier A fix the crash is gone. The test now fails with:
```
AssertionError: Evaluation result 0 failed: PCC comparison failed.
Calculated: pcc=0.986602805459758. Required: pcc=0.99.
```

## Root cause

**Bug 1 (fixed): Out-of-bounds access in `SharedLHSMatmulFusion<LinearOp>::collectCandidates`**

`TTIRToTTIRDecompositionPass` converts `ttir.dot_general` ops to `ttir.matmul`
ops. `MatmulWithBiasFusionPattern` then converts `matmul → [reshape] → add`
sequences into `LinearOp`. In DeBERTa-V3, parameter `%arg71` (shape
`[1,1,768]`) is reused as bias for both the content V-projection add and the
position K-projection add (multiple uses). `peelBiasTransformations` stops
peeling when it encounters a value with >1 use, leaving the bias as rank-3.
The resulting broadcast shape becomes `[1, 512, 768]`, so that LinearOp has a
rank-3 output while the other LinearOps sharing the same LHS `%53` have
rank-2 outputs.

`collectCandidates` checked only that candidate RHS rank matches the root op,
not output rank. It collected both rank-2 and rank-3 LinearOps into the same
fusion group. `replaceWithSlices` computes `outputFusedDim = rootOp.getRank()-1`
and accesses `shape[outputFusedDim]` on each candidate. When a rank-3 root op
is matched with a rank-2 candidate (2-element shape array), `shape[2]` is
out-of-bounds → crash.

**Bug 2 (unfixed): PCC degradation**

After the crash fix, the model compiles and runs but produces outputs with
PCC = 0.9866 against the CPU reference, below the 0.99 threshold. Root cause
is undiagnosed. Possibilities include residual incorrect fusion (the rank-3
LinearOp group that was excluded from fusion may itself be fused incorrectly
elsewhere), or BF16 accumulation error across 12 transformer layers. Not
investigated further per the "no chaining Tier A fixes" rule.

## Fix

**Bug 1 fix — `tt-mlir/lib/Dialect/TTIR/Transforms/TTIRFusing.cpp`**

In `collectCandidates<LinearOp>` (around line 1737), capture the root op's
output rank before the loop:

```cpp
auto rootOutputType = mlir::cast<RankedTensorType>(rootOp.getType());
int64_t rootOutputRank = rootOutputType.getRank();
```

Inside the loop, after the RHS-rank and batch-dims checks, add:

```cpp
auto outputType = mlir::cast<RankedTensorType>(op.getType());
if (outputType.getRank() != rootOutputRank) {
    continue;
}
```

This prevents mixing LinearOps of different output ranks in one fusion group.
The rank-3 candidate is excluded; the remaining rank-2 candidates fuse
correctly and `replaceWithSlices` no longer accesses an out-of-bounds index.

Committed on branch `remediation/deberta-masked-lm-pytorch-V3_Base-single-device-inference`
in `tt-mlir` (commit `ed8f5ee59`). Verified: `ttmlir-opt --pass-pipeline="builtin.module(ttir-fusing)" /tmp/deberta_canonicalized.mlir` exits 0 after the fix (previously aborted).

**Bug 2 — proposed investigation**

Compare output tensors layer-by-layer (pre/post attention, pre/post feed-forward)
between TT and CPU to identify where PCC first diverges. If the gap is
concentrated in the attention output the likely culprit is the remaining
rank-3 LinearOp (excluded from shared-LHS fusion, so now computed as a plain
LinearOp — check whether that path itself is numerically correct). If spread
evenly across all layers, suspect BF16 accumulation and measure
`TTXLA_REQUIRED_PCC`-class noise floor on a simpler BERT model for comparison.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
N/A — Tier is A (crash was fixed; PCC failure is a second, uninvestigated bug left per the no-chaining rule, not a Tier B classification)

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    105.59s (1:45)
- Tier A attempts: 1

## Files changed
- `tt-mlir/lib/Dialect/TTIR/Transforms/TTIRFusing.cpp` — output-rank guard in `collectCandidates<LinearOp>`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | ed8f5ee59e79ff89247aeb40ff51a642147c0151 |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
