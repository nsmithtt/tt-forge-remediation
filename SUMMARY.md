# Remediation Summary: bert-masked-lm-pytorch-nlpaueb-legal-bert-base-uncased-single-device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bert/masked_lm/pytorch-nlpaueb/legal-bert-base-uncased-single_device-inference]

## Result
NO_FIX_NEEDED

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
The failure message was:
```
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```
Which surfaced because pytest was invoked without `PYTHONPATH` set to include the `tests/` directory, causing conftest.py to fail with `ModuleNotFoundError: No module named 'infra'` before any test ran. The swig deprecation warning was the only visible output at tail-200 of the original invocation from the wrong directory.

## Root cause
Not a code bug. The test invocation in the failure report ran pytest from a directory that did not include `tests/` on `sys.path`. The `tests/conftest.py` does `from infra import DeviceConnectorFactory, Framework` where `infra` is `tests/infra/` — a local package, not an installed one. Without `PYTHONPATH=/home/ttuser/tt-forge-remediation/tt-xla/tests`, every pytest session aborts at conftest import before collecting any test. Running with the correct `PYTHONPATH` resolves the import and the test executes and passes on silicon.

## Fix
No fix applied. The test passes as-is when invoked correctly.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    44.10s
- Tier A attempts: N/A

## Files changed
None.

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
