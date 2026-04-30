# Remediation Summary: detr/object_detection/pytorch-ResNet101_Backbone-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[detr/object_detection/pytorch-ResNet101_Backbone-single_device-inference]

## Result
NO_FIX_NEEDED — test passes on the configured branch; the "failure message" is a spurious post-exit DeprecationWarning, not a test failure

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
The reported "failure message" is a Python interpreter-level DeprecationWarning printed to stderr by the SWIG runtime at process exit, after pytest has already reported the test result. It is not associated with a test failure. On branch ip-172-31-30-236-tt-xla-dev/ubuntu/hf-bringup-43, the test runs to completion and pytest exits PASS in 129.65s.

## Fix
No fix required. The test passes as-is on the configured branch.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    129.65s (0:02:09)
- Tier A attempts: N/A

## Files changed
None

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 4c72883ecb3497de9c208017fd1ec6565db112d4 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
