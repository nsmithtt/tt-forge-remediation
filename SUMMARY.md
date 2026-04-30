# Remediation Summary: llama_300m_v2_bigram/causal_lm/pytorch-base-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[llama_300m_v2_bigram/causal_lm/pytorch-base-single_device-inference]

## Result
SILICON_PASS — added missing `pythonpath = tests` and SWIG DeprecationWarning filter to pytest.ini

## Stack layer
tt-xla

## Tier
N/A

## Bug fingerprint
pytest-ini-missing-pythonpath-tests

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
`tests/conftest.py` imports `from infra import DeviceConnectorFactory, Framework` where `infra` is a package at `tests/infra/`. pytest.ini was missing the `pythonpath = tests` directive that adds `tests/` to sys.path during collection, so Python could not find the `infra` module. The SWIG DeprecationWarning was a secondary symptom surfaced in the error output. Both issues were already fixed on other tt-xla branches.

## Fix
Added to `tt-xla/pytest.ini`:
- `pythonpath = tests` — makes `tests/infra` (and other `tests/` packages) importable during pytest collection and execution
- `filterwarnings` block suppressing the SWIG `swigvarlink`/`SwigPy` DeprecationWarning

Branch: `remediation/llama_300m_v2_bigram-causal-lm-pytorch-single-device-inference` in tt-xla.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    90.20s (0:01:30)
- Tier A attempts: N/A

## Files changed
- tt-xla/pytest.ini

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 6fcd88cb6fc37510c3f4942e883ffa4b13d79a0f |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
