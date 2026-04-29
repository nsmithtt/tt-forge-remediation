# Remediation Summary: dinov3-feature_extraction-pytorch-Large-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[dinov3/feature_extraction/pytorch-Large-single_device-inference]

## Result
SILICON_PASS — spacy namespace shadowing fixed by removing models_root from sys.path

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
Footer: `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute`

## Root cause
`DynamicLoader.setup_models_path` in `tests/runner/utils/dynamic_loader.py` calls
`sys.path.insert(0, models_root)` where `models_root` is the `tt_forge_models/`
directory. Because `tt_forge_models/spacy/` exists as a directory without
`__init__.py`, Python treats it as a namespace package and registers it in
`sys.modules['spacy']`. The real spaCy library (with `spacy.Language`) is then
shadowed. When `datasets._dill.save()` later checks
`if issubclass(obj_type, spacy.Language)`, it finds the broken stub and raises
`AttributeError: module 'spacy' has no attribute 'Language'`.

Relative imports in loaders already work through `__package__` and the
manually-registered `tt_forge_models` namespace package (`__path__ = [models_root]`);
the `sys.path` insertion was redundant and harmful.

## Fix
Removed `sys.path.insert(0, models_root)` from `DynamicLoader.setup_models_path`
in `tt-xla/tests/runner/utils/dynamic_loader.py`. Added a NOTE comment explaining
why models_root must not be added to sys.path. The `tt_forge_models` namespace
package registration block immediately below was left intact since it is the
correct mechanism for loader relative imports.

Branch: `remediation/dinov3-feature_extraction-pytorch-Large-single_device-inference` in tt-xla.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    74.39s
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/utils/dynamic_loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | eb7d2625e9733f9e9da4bc61a830a8590f4b3ed4 |
| tt-forge-models | bd076f15a0ea7b6d8e40c54a2e847ca30e9e4987 |
