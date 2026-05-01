# Remediation Summary: mobilenetv3-pytorch-TF_Large_100-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mobilenetv3/pytorch-TF_Large_100-single_device-inference]

## Result
SILICON_PASS — two loader-layer bugs fixed: pytest pythonpath missing so infra package import failed, and sys.path shadowing broke spacy namespace

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
pytest-pythonpath-tests-missing

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
Two bugs in the tt-xla test infrastructure:

1. **Missing pythonpath in pytest.ini**: `tests/conftest.py` does `from infra import DeviceConnectorFactory, Framework`, relying on `tests/` being in `sys.path`. Because `tests/__init__.py` exists, pytest treats `tests/` as a package (rooted at the repo root `tt-xla/`), and does NOT automatically insert `tests/` into `sys.path`. Without `pythonpath = tests` in `pytest.ini`, the bare `from infra import ...` fails with `ModuleNotFoundError`. The SWIG DeprecationWarning was the visible message but the real crash was the `ModuleNotFoundError`.

2. **sys.path shadowing by models_root**: `DynamicLoader._setup_models_root()` in `tests/runner/utils/dynamic_loader.py` inserted `models_root` (the `tt_forge_models/` directory) at `sys.path[0]`. Because `tt_forge_models/spacy/` exists as a namespace directory, this shadowed the real `spacy` package, causing `datasets._dill` import failures that can break model collection.

## Fix
Two changes in tt-xla on branch `remediation/mobilenetv3-pytorch-tf-large-100-single-device-inference`:

1. **`pytest.ini`** -- Added `pythonpath = tests` so pytest adds `tests/` to `sys.path` before conftest.py runs, enabling `from infra import ...`. Also added `filterwarnings` to suppress the distracting SWIG DeprecationWarnings.

2. **`tests/runner/utils/dynamic_loader.py`** -- Removed the `sys.path.insert(0, models_root)` call. The `tt_forge_models` namespace package is already registered correctly via the module spec mechanism; adding `models_root` to `sys.path[0]` is not needed and causes `tt_forge_models/spacy/` to shadow the real `spacy` package.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    74.63s (0:01:14)
- Tier A attempts: N/A

## Files changed
- tt-xla/pytest.ini
- tt-xla/tests/runner/utils/dynamic_loader.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a6022d20f2c7b178658a10673768f24392e39001 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
