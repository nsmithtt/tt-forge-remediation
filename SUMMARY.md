# Remediation Summary: dino-feature_extraction-pytorch-Base_8-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[dino/feature_extraction/pytorch-Base_8-single_device-inference]

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
```
AttributeError: module 'spacy' has no attribute 'Language'
```

Traceback path: `load_dataset("huggingface/cats-image")` → `datasets._dill.save()` → `issubclass(obj_type, spacy.Language)` → `AttributeError`.

Reported failure message: `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute` (a pytest exit-line warning; the real failure was the AttributeError above).

## Root cause
`DynamicLoader.setup_models_path()` in `tests/runner/utils/dynamic_loader.py` inserts `models_root` (the `tt_forge_models/` directory) at the front of `sys.path`. Because `tt_forge_models/spacy/` is a directory without `__init__.py`, Python treats it as a namespace package and registers it in `sys.modules['spacy']`. When `datasets._dill.save()` later checks `if issubclass(obj_type, spacy.Language)`, it finds the hollow namespace package instead of the real spaCy library and raises `AttributeError: module 'spacy' has no attribute 'Language'`.

The `sys.path` insertion is not needed: relative imports in loaders work through `__package__` and the manually-registered `tt_forge_models` module whose `__path__` already points to `models_root`.

## Fix
Removed the `sys.path.insert(0, models_root)` block from `DynamicLoader.setup_models_path()` and added an explanatory comment. Changed in `tt-xla`:

- `tests/runner/utils/dynamic_loader.py` — delete the 3-line `sys.path.insert` block, add comment explaining why models_root must not be in sys.path.

Remediation branch: `remediation/dino-feature_extraction-pytorch-Base_8-single_device-inference` in tt-xla.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    46.24s
- Tier A attempts: N/A

## Files changed
- tt-xla: `tests/runner/utils/dynamic_loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 39e60e09725fc4f2fca664d92234d7ce82756308 |
| tt-forge-models | 0f7b734348 |
