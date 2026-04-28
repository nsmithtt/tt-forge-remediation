# Remediation Summary: ara_prompt_guard-sequence_classification-pytorch-V0-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[ara_prompt_guard/sequence_classification/pytorch-V0-single_device-inference]

## Result
SILICON_PASS

## Stack layer
tt-mlir

## Tier
A

## Bug fingerprint
shared-lhs-matmul-fusion-mixed-output-rank

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Fatal Python error: Aborted (SIGABRT) during `torch.compile(model, backend='tt')`.
GDB showed the crash inside `SharedLHSMatmulFusion<LinearOp>::matchAndRewrite` →
`ArrayRef<long>::operator[]` assertion failure: index out of bounds.

Diagnostic output confirming the bad state:
```
SharedLHSMatmulFusion: outputFusedDim=2 but candidate output rank=2 (rootOp rank=3)
```

## Root cause
In `TTIRFusing.cpp`, `SharedLHSMatmulFusion<LinearOp>::collectCandidates` collects
all LinearOps that share the same LHS (A) operand. It correctly guards on matching
`transpose_a` flag, matching RHS rank, and matching RHS batch dimensions. However it
does **not** guard on the output rank of each candidate matching the root op's output rank.

`matchAndRewrite` computes `outputFusedDim = rootOutputType.getRank() - 1` from the
root op, then passes it to `replaceWithSlices`, which uses it to index into each
candidate's shape array (`shape[outputFusedDim]`). If a candidate's output rank is
lower than the root op's (e.g. rank 2 vs root rank 3), the index is out of bounds →
`ArrayRef` assertion → SIGABRT.

The model that triggers this is `NAMAA-Space/Ara-Prompt-Guard_V0`, a DeBERTa-v2
sequence classifier with disentangled attention (`share_att_key=True`,
`pos_att_type=['p2c','c2p']`). Its attention projections create multiple LinearOps
sharing the same A tensor but with different LHS ranks (some 2-D, some 3-D), causing
the output rank mismatch.

## Fix
In `tt-mlir/lib/Dialect/TTIR/Transforms/TTIRFusing.cpp`, inside
`SharedLHSMatmulFusion<LinearOp>::collectCandidates`, capture the root op's output
rank before the loop and skip any candidate whose output rank differs:

```cpp
int64_t rootOutputRank =
    mlir::cast<RankedTensorType>(rootOp.getType()).getRank();
// ... inside the per-candidate loop, after the batch-dim check:
if (mlir::cast<RankedTensorType>(op.getType()).getRank() != rootOutputRank) {
  continue;
}
```

Remediation branch: `remediation/ara_prompt_guard-sequence_classification-pytorch-V0-single_device-inference`
in `tt-mlir`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    128.60s
- Tier A attempts: 1

## Files changed
- `tt-mlir/lib/Dialect/TTIR/Transforms/TTIRFusing.cpp`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 0061ba367227d04853f1fcfda8c739a8b4d93f39 |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
