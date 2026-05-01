# Remediation Summary: bert/masked_lm/pytorch-Base_Multilingual_Cased-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bert/masked_lm/pytorch-Base_Multilingual_Cased-single_device-inference]

## Result
NO_FIX_NEEDED — test passes on the configured branch; the reported "failure message" is a harmless DeprecationWarning, not an error

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
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute

## Root cause
The "failure message" is a Python DeprecationWarning emitted by the SWIG runtime when a built-in SWIG proxy type (swigvarlink) lacks a __module__ attribute. This is a cosmetic warning printed to stderr after the pytest session exits; it does not cause any test to fail. The test itself ran to completion and exited PASS with a PCC above the required threshold. No bug exists in the compiler stack, the loader, or the test harness for this test on the configured branch (worktree-aus-wh-07-tt-xla-dev+nsmith+hf-bringup-start500-3).

## Fix
No fix required. The test passes as-is.

## Verification
- pytest exit: PASS
- Hardware:    wormhole
- Duration:    48.48s
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
