# Remediation Summary: chronos2-pytorch-Chronos_2_Synth-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[chronos2/pytorch-Chronos_2_Synth-single_device-inference]

## Result
SILICON_PASS — pytest.ini missing `pythonpath = tests`; adding it plus SWIG warning filter unblocked all prior compiler fixes

## Stack layer
loader, tt-mlir

## Tier
A

## Bug fingerprint
pytest-missing-pythonpath-tests

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
`tests/__init__.py` exists (added in commit 5175d7fc5 for superset integration), which causes pytest to treat `tests/` as a package and add only the parent directory (`tt-xla/`) to `sys.path`. This prevents `from infra import ...` in `tests/conftest.py` from finding `tests/infra/`. The fix is `pythonpath = tests` in `pytest.ini`, which is a pytest 7.0+ feature. The SWIG DeprecationWarning about `swigvarlink` is cosmetic — it appears as the reported failure message because it was the last thing printed before the collection error caused pytest to exit.

Five prior remediation sessions had applied this same fix to their respective remediation branches but it was never merged to the tt-xla main branch. This session applies it to the chronos2 remediation branch on top of four existing Tier A tt-mlir compiler fixes (SDPA verifier, Gather guard, expandMaskQueryDim, decode guard) and one loader fix (chronos-forecasting requirements.nodeps.txt).

## Fix
**tt-xla** (`remediation/chronos2-pytorch-Chronos_2_Synth-single_device-inference`, commit `92e486b06`):
- `pytest.ini`: added `pythonpath = tests` and `filterwarnings` to suppress SwigPy DeprecationWarnings

**tt-mlir** (`remediation/chronos2-pytorch-Chronos_2_Synth-single_device-inference`, commits `b88e5b71f`–`6929e98f8`):
- `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`: guard `GatherToSliceRepeatConcat` against `maxIndex==0` and concat size mismatch
- `lib/Dialect/TTIR/IR/TTIROps.cpp`, `lib/Dialect/TTNN/IR/TTNNOps.cpp`: allow broadcast dim[2]=1 in SDPA attention mask verifier
- `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`: expand broadcast SDPA mask dim[2] and guard decode path for `kv_seq_len % 32 != 0`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    65.08s (0:01:05)
- Tier A attempts: 1 (applying previously-developed fixes from remediation branch)

## Files changed
- `tt-xla/pytest.ini` (pythonpath + filterwarnings)
- `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`
- `tt-mlir/lib/Dialect/TTIR/IR/TTIROps.cpp`
- `tt-mlir/lib/Dialect/TTNN/IR/TTNNOps.cpp`
- `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 6929e98f8c9e965bd2402aff7519da67e236046f |
| tt-xla          | 92e486b06480a8e2e6e37ae220b52048eec5c313 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
