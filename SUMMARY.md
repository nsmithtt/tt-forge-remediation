# Remediation Summary: gliner2-pytorch-multi_v1-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gliner2/pytorch-Multi_v1-single_device-inference]

## Result
FAIL — residual Tier B: TT hardware BF16 matmul precision floor in 12-layer DeBERTa-v3-base encoder gives PCC=0.9898 vs required 0.99 (BF16 CPU floor is 0.9994)

## Stack layer
tt-xla, tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-bf16-matmul-precision-floor

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure

**Original failure (fixed — loader):**
```
ModuleNotFoundError: No module named 'llm2vec.models'
```
in `gliner/modeling/encoder.py:21`, at module import time.

**Secondary failure (fixed — Tier A, tt-mlir):**
The SharedLHSMatmulFusion SIGABRT (same root cause as prior gliner reports) was
fixed proactively since DeBERTa-v3-base is used by this model.

**Terminal failure (Tier B, unfixed):**
```
AssertionError: Evaluation result 0 failed: PCC comparison failed.
Calculated: pcc=0.9897966290567508. Required: pcc=0.99.
```

## Root cause

**Bug 1 (fixed — tt-xla loader):** `DynamicLoader.setup_models_path` adds
`models_root` (the `tt_forge_models/` directory) to `sys.path` via
`sys.path.insert(0, models_root)`. This makes every subdirectory visible as a
top-level Python namespace package. The `tt_forge_models/llm2vec/` directory (the
llm2vec model loader directory, not the real llm2vec library) is thus importable as
`llm2vec`. When gliner2 imports gliner, which imports `gliner.modeling.encoder`, the
module-level code calls `is_module_available("llm2vec")` which finds this namespace
package and returns True. The subsequent `from llm2vec.models import ...` fails
because `tt_forge_models/llm2vec/` has no `models` subpackage.

Fix: remove `sys.path.insert(0, models_root)` from `setup_models_path`. Relative
imports in loaders work via `__package__` and the manually-registered `tt_forge_models`
namespace module; the sys.path insertion is not needed.

**Bug 2 (fixed — Tier A, tt-mlir):** `SharedLHSMatmulFusion<LinearOp>::collectCandidates`
in `TTIRFusing.cpp` verified that fusion candidates share the same LHS and have the same
RHS rank, but did not check that the candidate's output rank matches the root op's output
rank. For DeBERTa-v3 disentangled attention, some `ttir.linear` ops share the same LHS
but have different output ranks (2-D vs 3-D). `replaceWithSlices` then indexed
`shape[rootOutputRank - 1]` on a rank-2 candidate, causing an out-of-bounds
`ArrayRef::operator[]` assertion (SIGABRT).

Fix: added output-rank equality guard in `collectCandidates`.

**Bug 3 (unfixed — Tier B):** After both fixes, the DeBERTa-v3-base encoder (12 layers,
hidden_size=768) gives TT hardware PCC=0.9898 against FP32 CPU, while BF16 CPU gives
0.9994. The additional ~0.01 precision deficit on TT hardware is caused by accumulated
BF16 matmul errors across 12 transformer layers on Blackhole architecture. The required
threshold is 0.99 and the measured TT PCC falls short by ~0.002.

## Fix

**Applied (tt-xla):** Removed `sys.path.insert(0, models_root)` from
`DynamicLoader.setup_models_path` in `tests/runner/utils/dynamic_loader.py`.
Commit: `87d5ae2e2` on branch
`remediation/gliner2-pytorch-multi_v1-single_device-inference` in tt-xla.

**Applied (tt-mlir):** Added output-rank guard in
`SharedLHSMatmulFusion::collectCandidates` in
`lib/Dialect/TTIR/Transforms/TTIRFusing.cpp`. Commit: `71918235a` on branch
`remediation/gliner2-pytorch-multi_v1-single_device-inference` in tt-mlir.

**Not applied (Tier B):** The residual PCC=0.9898 deficit requires improving BF16
matmul accumulation precision on WH/BH hardware or using F32 intermediate accumulation
throughout the compiler lowering path. This is a cross-cutting change.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
cross-cutting

The BF16 matmul precision floor manifests across all layers of the 12-layer
DeBERTa-v3-base encoder. Fixing it requires changing either the hardware BF16
accumulation behavior or inserting F32 accumulation at every matmul site throughout
the compiler lowering pipeline — a cross-cutting change across multiple files and
multiple passes in tt-mlir.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    110.11s (0:01:50)
- Tier A attempts: 1

## Files changed
- `tt-xla/tests/runner/utils/dynamic_loader.py` — removed sys.path.insert(0, models_root) from setup_models_path
- `tt-mlir/lib/Dialect/TTIR/Transforms/TTIRFusing.cpp` — added output-rank guard in SharedLHSMatmulFusion::collectCandidates

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d75355 |
| tt-mlir         | 71918235a |
| tt-xla          | 87d5ae2e2 |
| tt-forge-models | 0f7b734348 |
