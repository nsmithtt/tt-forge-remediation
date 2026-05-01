# Remediation Summary: mobileclip_b_lt-pytorch-B_LT-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mobileclip_b_lt/pytorch-B_LT-single_device-inference]

## Result
SILICON_PASS — removed models_root from sys.path, eliminating spacy namespace shadowing

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
spacy-namespace-shadows-real-package

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
venv/lib/python3.12/site-packages/datasets/utils/_dill.py:42: in save
    if issubclass(obj_type, spacy.Language):
                            ^^^^^^^^^^^^^^
E   AttributeError: module 'spacy' has no attribute 'Language'
```
(Reported as `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute` from pytest summary line — this is a red herring; the actual exception is the spacy AttributeError above.)

## Root cause
`DynamicLoader.setup_models_path` in `tests/runner/utils/dynamic_loader.py` inserts `models_root` (the `tt_forge_models/` directory) into `sys.path`. Because `tt_forge_models/spacy/` is a directory without `__init__.py`, Python registers a namespace package named `spacy` from it — overriding the real spaCy package. When `datasets._dill.save()` checks `issubclass(obj_type, spacy.Language)`, the namespace package lacks `Language` and raises `AttributeError`. The `sys.path` insertion was never necessary; relative imports in loaders work via `__package__` + the manually-registered `tt_forge_models` namespace in `sys.modules`.

## Fix
Removed the `sys.path.insert(0, models_root)` call from `DynamicLoader.setup_models_path` in `tt-xla/tests/runner/utils/dynamic_loader.py`. Added a comment explaining why it must not be re-added.

**Branch:** `remediation/mobileclip_b_lt-pytorch-B_LT-single_device-inference` in tt-xla

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    121.03s
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/utils/dynamic_loader.py` — removed `sys.path.insert(0, models_root)`, added explanatory comment

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a4ea566eb7a00521b4fe2a7db55ec47c1ae736d6 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
