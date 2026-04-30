# Remediation Summary: dfn_clip-pytorch-ViT_H_14_378-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[dfn_clip/pytorch-ViT_H_14_378-single_device-inference]

## Result
SILICON_PASS — loader-layer fix: removed models_root from sys.path to stop spacy/ namespace shadowing the real spaCy package

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
Traceback through `datasets._dill.save()` → `issubclass(obj_type, spacy.Language)` where `spacy` was a namespace package (no `Language` attribute) created by `sys.path.insert(0, models_root)`.

## Root cause
`DynamicLoader.setup_models_path()` in `tests/runner/utils/dynamic_loader.py` calls `sys.path.insert(0, models_root)` where `models_root` is the `tt_forge_models/` directory. Because `tt_forge_models/spacy/` exists as a directory without `__init__.py`, Python creates a namespace package for `spacy` from it. During model discovery, `huspacy/pytorch/loader.py` does `import spacy` at the top level, registering this hollow namespace package in `sys.modules['spacy']`. Later, when `load_dataset("huggingface/cats-image")` is called inside `dfn_clip`'s `load_inputs()`, `datasets._dill.save()` checks `if issubclass(obj_type, spacy.Language)`, which fails with `AttributeError: module 'spacy' has no attribute 'Language'` because the namespace package has no `Language` attribute.

The fix is straightforward: the `sys.path.insert` is not needed because relative imports in loaders already work through the `__package__` attribute and the manually-registered `tt_forge_models` namespace package (`sys.modules["tt_forge_models"].__path__ = [models_root]`).

## Fix
Removed `sys.path.insert(0, models_root)` from `DynamicLoader.setup_models_path()` in `tests/runner/utils/dynamic_loader.py` (tt-xla repo). Added a comment explaining why models_root must NOT be added to sys.path.

Remediation branch: `remediation/dfn_clip-pytorch-ViT_H_14_378-single_device-inference` in tt-xla.

## Verification
- pytest exit: PASS
- Hardware: blackhole-p150b
- Duration: 197.39s (0:03:17)
- Tier A attempts: N/A

## Files changed
- `tests/runner/utils/dynamic_loader.py` (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 2e304750f58990a6718abd068119ff731ec83d33 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
