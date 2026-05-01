# Remediation Summary: eva-pytorch-Large_Patch14_336_IN22K_FT_IN22K_IN1K-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[eva/pytorch-Large_Patch14_336_IN22K_FT_IN22K_IN1K-single_device-inference]

## Result
SILICON_PASS

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
E   AttributeError: module 'spacy' has no attribute 'Language'

## Root cause
`DynamicLoader.setup_models_path()` in `tests/runner/utils/dynamic_loader.py` inserted `models_root` (the `tt_forge_models/` directory) into `sys.path`. Because `tt_forge_models/spacy/` is a directory without `__init__.py`, Python registered it as a namespace package named `spacy`, shadowing the real spaCy library. When `huspacy/pytorch/loader.py` imported `spacy` during model discovery, the namespace package (without `Language`) was cached in `sys.modules['spacy']`. Later, `datasets.utils._dill.save()` called `issubclass(obj_type, spacy.Language)`, which raised `AttributeError: module 'spacy' has no attribute 'Language'`. The `models_parent` directory (one level up from `models_root`) is already added to `sys.path` in `import_model_loader`, so the `models_root` insertion was redundant and harmful.

## Fix
Removed the `sys.path.insert(0, models_root)` block from `DynamicLoader.setup_models_path()` in `tt-xla/tests/runner/utils/dynamic_loader.py`, adding a comment explaining why `models_root` must not be on `sys.path`. The namespace package registration for `tt_forge_models` that follows was left in place — it is not affected by this change.

File: `tests/runner/utils/dynamic_loader.py` in the `tt-xla` subproject, remediation branch `remediation/eva-pytorch-Large_Patch14_336_IN22K_FT_IN22K_IN1K-single_device-inference`.

## Verification
- pytest exit: PASS
- Hardware:    wormhole
- Duration:    107.77s
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/utils/dynamic_loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 4e5949fc440b3dc4dcb583cef16256f32fe32f63 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
