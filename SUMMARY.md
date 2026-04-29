# Remediation Summary: esm2/masked_lm/pytorch-facebook/esm2_t33_650M_UR50D-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[esm2/masked_lm/pytorch-facebook/esm2_t33_650M_UR50D-single_device-inference]

## Result
NO_FIX_NEEDED — test passes cleanly on the configured branch

## Stack layer
n/a

## Tier
N/A

## Bug fingerprint
n/a

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Extension modules: numpy._core._multiarray_umath, numpy.linalg._umath_linalg, psutil._psutil_linux, torch._C, ... (total: 222)

The provided failure message is the Python interpreter's crash-report extension-module dump. This indicates a prior Python-level crash (segfault or fatal error) during a previous CI run.

## Root cause
No root cause to investigate. The ESM-2 masked LM loader was added to tt-forge-models at commit 91a57845ac and is present in the pinned submodule at 0f7b734348. The test passed cleanly in three consecutive runs on n150 silicon.

## Fix
No fix required. Test already passes on the configured branch (tt-forge-models at `ip-172-31-30-232-tt-xla-dev/ubuntu/hf-bringup-range-1500-500-5`, pinned to 0f7b734348 via configure.sh submodule reset).

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    148.97s
- Tier A attempts: N/A

## Files changed
None

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
