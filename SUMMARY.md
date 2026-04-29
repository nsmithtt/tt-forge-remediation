# Remediation Summary: camembert-pytorch-Base_Legacy-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[camembert/pytorch-Base Legacy-single_device-inference]

## Result
NO_FIX_NEEDED — test passes when the pytest node ID is properly quoted on the command line

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
ERROR: file or directory not found: Legacy-single_device-inference]

## Root cause
The test ID `tests/runner/test_models.py::test_all_models_torch[camembert/pytorch-Base Legacy-single_device-inference]` contains a space inside the brackets ("Base Legacy"). When pytest is invoked without quoting this argument in the CI shell invocation, the shell splits the argument at the space, and pytest sees `Legacy-single_device-inference]` as a separate positional argument (a file/directory path), which it cannot find. This is a CI invocation quoting issue, not a compiler stack or loader bug. When the full test ID is passed as a single properly-quoted argument, the test runs and passes on silicon in 44.47s.

## Fix
No code changes required. The fix is to quote the pytest node ID in the CI invocation:
```
pytest -svv "tests/runner/test_models.py::test_all_models_torch[camembert/pytorch-Base Legacy-single_device-inference]"
```

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    44.47s
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
