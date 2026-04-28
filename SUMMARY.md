# Remediation Summary: jtapsa_moep-causal_lm-pytorch-moep-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[jtapsa_moep/causal_lm/pytorch-moep-single_device-inference]

## Result
FAIL — stablehlo gather multi-dim lowering is incorrect; ParallelLayer torch.gather produces PCC=0.138

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
stablehlo-gather-multidim-flattenindices-incorrect

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

## Root cause
Two stacked bugs in tt-mlir were discovered:

**Bug 1 (Tier A — fixed):** `StableHLOToTTIREmbeddingBackwardOpConversionPattern` passes the
stablehlo scatter's `scatter_indices` tensor directly to `ttnn::EmbeddingBackwardOp`. For the
`index_add_()` scatter emitted by the MoE dispatch loop the indices are 1D `[Ne]`, but
`embedding_backward_device_operation.cpp:validate_on_program_cache_miss()` asserts the index
tensor must be at least 2D with `shape[1] == 1 && shape[2] == 1`. The missing dimensions cause
a `TT_FATAL` assertion which propagates as Error code 13 (`INTERNAL`).

A second semantic error compounds this: `ttnn::embedding_bw` always zero-initialises its output
before accumulating updates at indexed positions. The MoE model chains 6 `index_add_()` calls
that scatter into an accumulated `y_flat` tensor; each call must add into the *existing*
non-zero values. With the plain `embedding_bw(indices, weight, grad)` lowering only the *last*
scatter's contributions survive, giving PCC ≈ 0.077.

**Bug 2 (Tier B — not fixed):** `ParallelLayer.forward()` calls
`torch.gather(Y, dim=1, index=gather_idx)` with `Y:[128,4,192]` and
`gather_idx:[128,2,192]`. XLA lowers this to a `stablehlo.gather` with
`start_index_map=[0,1,2]` (three indexed dimensions). The
`StableHLOGatherToEmbeddingPattern` in `StableHLOToTTIRPatterns.cpp`
reaches `flattenStartIndices` with `numIndexingDims=3` and emits its own
warning:

```
"End results might be incorrect when indexing multiple dimensions of input
because of typecast ops."
```

An isolated test (`test_gather_isolated.py`) confirms PCC = 0.138 for this
exact shape. `ParallelLayer` is applied 10 times per forward pass (once per
transformer block), making this the dominant source of error. The overall
model test ends at PCC ≈ 0.283.

## Fix
**Bug 1 fix** — committed on branch
`remediation/jtapsa_moep-causal_lm-pytorch-moep-single_device-inference` in tt-mlir
(commit `b74c0e7dfc3b61ddc77fc9f7175dce514f56780c`):

File: `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`
`EmbeddingBackwardOpConversionPattern::matchAndRewrite`

1. **1D→2D reshape for indices** (Fix #1): When the indices operand is rank-1
   `[Ne]`, insert a reshape to `[1, Ne]` before passing to `ttnn::EmbeddingBackwardOp`.
   This satisfies the kernel's shape assertion and resolves the Error code 13 crash.

2. **Add original operand back** (Fix #2): Replace the single `replaceOpWithNewOp`
   with a two-op sequence:
   ```
   embBw = EmbeddingBackwardOp(indices, weight, grad)
   result = AddOp(weight, embBw)
   ```
   Because `embedding_bw` always starts from zeros, adding the original `weight`
   (which holds the accumulated `y_flat` value) back in correctly implements
   `index_add_` semantics for non-zero operands. For the standard embedding-backward
   case the operand is always zero so the add is a no-op.

**Bug 2 proposed fix** — Tier B, not attempted:

The correct fix is to add a legality guard in `StableHLOGatherToEmbeddingPattern`
that rejects (via `notifyMatchFailure`) any gather where `numIndexingDims > 1`,
and then implement a new lowering pattern in `StableHLOToTTIRPatterns.cpp` that
correctly handles the general `stablehlo.gather` case by forwarding it to a TTIR
gather or slice+concat chain. Alternatively, `flattenStartIndices` needs to be
fixed to correctly handle multi-dimensional start index maps. Both paths require
changes across multiple files / patterns and may affect other models.

## Tier B justification
`cross-cutting` — `flattenStartIndices` is a shared helper used by the same
`StableHLOGatherToEmbeddingPattern` for all gather ops. Fixing the multi-dim
case requires either correctly implementing N-dim index flattening (a fundamental
algorithm change) or adding a new general-purpose gather lowering path. Either
approach touches multiple files and patterns and risks breaking the existing
single-dimension gather path used by embedding lookups throughout the model zoo.
The bug is also self-documented (`emitWarning` already in-tree), indicating the
team is aware but has not yet scoped the fix.

## Verification
- pytest exit: FAIL
- Hardware:    not-run
- Duration:    N/A
- Tier A attempts: 1 (scatter fix; resolves Error code 13 and index_add_ semantics)

## Files changed
- `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` — Fix #1 (1D→2D indices reshape) and Fix #2 (weight + embBw add pattern)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | b74c0e7dfc3b61ddc77fc9f7175dce514f56780c |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
