# Remediation Summary: mobileclip-pytorch-S1-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mobileclip/pytorch-S1-single_device-inference]

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
AttributeError: module 'spacy' has no attribute 'Language'
```
(reported as: `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute`)

## Root cause
`DynamicLoader.setup_models_path` in `tests/runner/utils/dynamic_loader.py` was inserting `models_root` (the `tt_forge_models/` directory) at position 0 of `sys.path`. Because `tt_forge_models/spacy/` is a directory without `__init__.py`, Python creates a namespace package named `spacy` from it. When any loader imports `spacy` during model discovery, this namespace shadows the real spaCy library. Later, `datasets._dill.save()` checks `if issubclass(obj_type, spacy.Language)` and fails because the shadowed `spacy` module has no `Language` attribute.

The `sys.path` insertion was not needed: relative imports inside loaders work via `__package__` and the manually-registered `tt_forge_models` namespace package (`sys.modules["tt_forge_models"]`) — `sys.path` insertion is redundant and harmful.

## Fix
Removed 4 lines from `tests/runner/utils/dynamic_loader.py` in `tt-xla`:

```python
- # Add the models root to sys.path so relative imports work
- if models_root not in sys.path:
-     sys.path.insert(0, models_root)
-
```

Commit: `a88702686` on branch `remediation/mobileclip-pytorch-S1-single_device-inference` in tt-xla.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    270.22s (0:04:30)
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/utils/dynamic_loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a887026863313e38a6df2d405f9b48a7bf5c7d32 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
