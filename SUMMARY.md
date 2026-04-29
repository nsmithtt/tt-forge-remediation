# Remediation Summary: facial_emotions_detection-pytorch-Base-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[facial_emotions_detection/pytorch-Base-single_device-inference]

## Result
SILICON_PASS — spacy namespace-shadowing bug fixed by removing models_root from sys.path

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
E   AttributeError: module 'spacy' has no attribute 'Language'
```

Full traceback originates in `loader.py:92` calling `load_dataset("huggingface/cats-image", split="test")`. Deep inside `datasets._dill.save()` it checks `if issubclass(obj_type, spacy.Language):` but `spacy` resolves to the shadowed namespace package from `tt_forge_models/spacy/`, which has no `Language` attribute.

The test's reported failure message (`sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute`) is emitted as a warning at process exit; the actual error is the `AttributeError` above.

## Root cause
`DynamicLoader.setup_models_path()` in `tests/runner/utils/dynamic_loader.py` inserts `models_root` (the `tt_forge_models/` directory) into `sys.path`. Because `tt_forge_models/spacy/` exists as a directory without `__init__.py`, Python creates a namespace package `spacy` from it whenever any loader causes model discovery to scan that path. When `huspacy/pytorch/loader.py` or similar imports `spacy` at the top level during discovery, this namespace (not the real spaCy library) ends up in `sys.modules['spacy']` without `Language`. Later, `datasets._dill.save()` hits the `issubclass(obj_type, spacy.Language)` guard and crashes.

The fix is a loader-layer fix: relative imports in loaders work via `__package__` plus the manually-registered `tt_forge_models` namespace module (which already has `__path__ = [models_root]`). The `sys.path` insertion is unnecessary.

## Fix
Removed the `sys.path.insert(0, models_root)` block (3 lines) from `DynamicLoader.setup_models_path()` in `tests/runner/utils/dynamic_loader.py` in the tt-xla repo. Added a comment explaining why `sys.path` insertion must not be used here.

File changed: `tests/runner/utils/dynamic_loader.py`
Commit: `19da006991e86435ff2cf3a373433693fb4342a7` on branch `remediation/facial_emotions_detection-pytorch-Base-single_device-inference` in tt-xla.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    38.76s
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/utils/dynamic_loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 19da006991e86435ff2cf3a373433693fb4342a7 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
