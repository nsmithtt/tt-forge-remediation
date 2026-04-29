# Remediation Summary: bioclip-pytorch-ViT_Base_Patch16_224-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bioclip/pytorch-ViT_Base_Patch16_224-single_device-inference]

## Result
SILICON_PASS — spacy namespace-package shadowing fixed by removing sys.path.insert of models_root in dynamic_loader

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
AttributeError: module 'spacy' has no attribute 'Language'

Traceback in datasets/utils/_dill.py:42:
  if issubclass(obj_type, spacy.Language):

Triggered during load_dataset("huggingface/cats-image") inside bioclip loader's load_inputs().

## Root cause
`DynamicLoader.setup_models_path()` in `tests/runner/utils/dynamic_loader.py` called `sys.path.insert(0, models_root)`, where `models_root` is the `tt_forge_models/` directory. Because `tt_forge_models/spacy/` exists as a directory without `__init__.py`, Python registers it as a namespace package named `spacy` in `sys.modules`. This shadows the real spaCy library. When `datasets._dill.save()` later checks `if issubclass(obj_type, spacy.Language):`, the namespace package has no `Language` attribute, causing `AttributeError`. The `sys.path` insertion is unnecessary because relative imports in loaders work via `__package__` and the manually-registered `tt_forge_models` namespace package (which already has `__path__ = [models_root]`).

## Fix
Removed `sys.path.insert(0, models_root)` from `DynamicLoader.setup_models_path()` in `tt-xla/tests/runner/utils/dynamic_loader.py`. A comment was added explaining why `sys.path` must not include `models_root`. The fix was cherry-picked from commit `c87bc9bb2` (originally landed in the `aesthetic_shadow` remediation branch) onto a new branch `remediation/bioclip-pytorch-ViT_Base_Patch16_224-single_device-inference` in tt-xla.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    94.46s (0:01:34)
- Tier A attempts: N/A

## Files changed
- tt-xla: `tests/runner/utils/dynamic_loader.py` (removed sys.path.insert of models_root, added explanatory comment)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 17f8cad778de3dc0fdf92766320c5de89967256a |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
