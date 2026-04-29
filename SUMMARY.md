# Remediation Summary: bert-masked_lm-pytorch-Large_Cased-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bert/masked_lm/pytorch-Large_Cased-single_device-inference]

## Result
NO_FIX_NEEDED — test already passes on the configured branch when pytest is run with the correct PYTHONPATH from `venv/activate`

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
```
ImportError while loading conftest '/home/nsmith/tt-forge-remediation/tt-xla/tests/conftest.py'.
tests/conftest.py:25: in <module>
    from infra import DeviceConnectorFactory, Framework
E   ModuleNotFoundError: No module named 'infra'
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

## Root cause
The failure occurs only when pytest is invoked without activating the tt-xla venv via `source venv/activate`. The custom `venv/activate` script (not the standard `venv/bin/activate`) sets `PYTHONPATH="$(pwd):$(pwd)/tests:..."`, which makes `tests/infra` importable as `infra`. The `tests/__init__.py` causes pytest's default import-mode to add the project root (not `tests/`) to sys.path, so without the PYTHONPATH export from `venv/activate`, `from infra import ...` in `tests/conftest.py` fails with `ModuleNotFoundError`.

The reported failure message `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute` is a harmless SWIG internal-type warning that appears both in passing and failing runs; it is not an error.

When run correctly (after `source venv/activate` or with `PYTHONPATH=tests` set), the test passes cleanly on the configured branch.

## Fix
No fix required. The test passes as-is on the configured branch.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    85.88s
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
