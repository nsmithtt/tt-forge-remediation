# Remediation Summary: openmed-ner_pathology_detect-pytorch-OpenMed-OpenMed-NER-PathologyDetect-SuperClinical-434M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[openmed/ner_pathology_detect/pytorch-OpenMed/OpenMed-NER-PathologyDetect-SuperClinical-434M-single_device-inference]

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
Fatal Python error: Aborted

python3: /opt/ttmlir-toolchain/include/llvm/ADT/ArrayRef.h:253: const T &llvm::ArrayRef<long>::operator[](size_t) const [T = long]: Assertion `Index < Length && "Invalid index!"' failed.

Stack: dynamo_bridge.py partition_fx_graph_for_cpu_fallback → extract_graph_helper → extract_internal → transformers/models/deberta_v2/modeling_deberta_v2.py:1136

## Root cause
`SharedLHSMatmulFusion<LinearOp>` in `tt-mlir/lib/Dialect/TTIR/Transforms/TTIRFusing.cpp`
crashes with an ArrayRef out-of-bounds assertion (SIGABRT) when the fusion
candidate set contains LinearOps with different output ranks.

`collectCandidates` guards that all LHS-sharing ops have the same RHS rank
but does NOT check that their output ranks match the root op. DeBERTa-v2
disentangled attention has both rank-3 and rank-2 projection outputs that
share the same LHS. When the root op is rank-3, `outputFusedDim = 2`, but
a rank-2 candidate's output shape has only 2 elements (indices 0 and 1),
so `replaceWithSlices` accesses `shape[2]` and triggers the OOB assertion.

## Fix
Added an output-rank guard in `collectCandidates` in
`lib/Dialect/TTIR/Transforms/TTIRFusing.cpp`:

```cpp
int64_t rootOutputRank =
    mlir::cast<RankedTensorType>(rootOp.getType()).getRank();
// ...
// Output rank must match the root op so replaceWithSlices can index the
// fused output dimension uniformly across all candidates.
if (mlir::cast<RankedTensorType>(op.getType()).getRank() !=
    rootOutputRank) {
  continue;
}
```

tt-mlir commit: `0758ba76a871c715ebe91712baf03a2e762b2247`
Branch: `remediation/openmed-ner_pathology_detect-pytorch-OpenMed-OpenMed-NER-PathologyDetect-SuperClinical-434M-single_device-inference`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    260.77s (0:04:20)
- Tier A attempts: 1

## Files changed
- `tt-mlir/lib/Dialect/TTIR/Transforms/TTIRFusing.cpp`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 0758ba76a871c715ebe91712baf03a2e762b2247 |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
