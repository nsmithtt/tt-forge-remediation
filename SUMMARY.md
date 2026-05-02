# Remediation Summary: openmed-ner_pharma_detect-pytorch-OpenMed-OpenMed-NER-PharmaDetect-SuperClinical-141M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[openmed/ner_pharma_detect/pytorch-OpenMed/OpenMed-NER-PharmaDetect-SuperClinical-141M-single_device-inference]

## Result
SILICON_PASS

## Stack layer
tt-mlir

## Tier
A

## Bug fingerprint
ttir-shared-lhs-matmul-fusion-mixed-rank-oob

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

Crash inside `partition_fx_graph_for_cpu_fallback` → `extract_graph_helper` → `TTIRFusing.cpp::SharedLHSMatmulFusion::collectCandidates/replaceWithSlices`.

## Root cause
`SharedLHSMatmulFusion<LinearOp>` in `tt-mlir/lib/Dialect/TTIR/Transforms/TTIRFusing.cpp`
`collectCandidates` verified that all LHS-sharing LinearOps agree on RHS rank but NOT on output rank.
OpenMed-NER-PharmaDetect-SuperClinical-141M is a DeBERTa-v2-based model; its disentangled
attention mixes rank-3 projections (input `[1,128,384]`) with rank-2 projections that share
the same LHS. When `replaceWithSlices` computes `outputFusedDim = rootOutputRank - 1 = 2`
it accesses `shape[2]` on a rank-2 output tensor (valid indices: 0, 1), triggering the
ArrayRef OOB assertion and SIGABRT.

## Fix
Cherry-picked commit `bec72757a` ("[TTIRFusing] Guard SharedLHSMatmulFusion against mixed-rank outputs")
from the existing `remediation/cross-encoder-nli-pytorch-nli-deberta-v3-xsmall-single_device-inference`
branch onto the current tt-mlir HEAD.

Added an output-rank guard in `collectCandidates` in
`tt-mlir/lib/Dialect/TTIR/Transforms/TTIRFusing.cpp`:
```cpp
int64_t rootOutputRank =
    mlir::cast<RankedTensorType>(rootOp.getType()).getRank();
// Output rank must match the root op so replaceWithSlices can index the
// fused output dimension uniformly across all candidates.
if (mlir::cast<RankedTensorType>(op.getType()).getRank() != rootOutputRank) {
  continue;
}
```

Remediation branch in tt-mlir:
`remediation/openmed-ner_pharma_detect-pytorch-OpenMed-OpenMed-NER-PharmaDetect-SuperClinical-141M-single_device-inference`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    68.11s
- Tier A attempts: 1

## Files changed
- `tt-mlir/lib/Dialect/TTIR/Transforms/TTIRFusing.cpp` (+9 lines)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 5a9d6e68a292a63734f3c583f09fd19e453bdf17 |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
