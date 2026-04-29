# Remediation Summary: clip_vit_eurosat-feature_extraction-pytorch-Base_Patch32_EuroSAT-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[clip_vit_eurosat/feature_extraction/pytorch-Base_Patch32_EuroSAT-single_device-inference]

## Result
SILICON_PASS — removed sys.path.insert(models_root) that caused spacy/ directory to shadow real spaCy package

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
third_party/tt_forge_models/clip_vit_eurosat/feature_extraction/pytorch/loader.py:87: in load_inputs
    dataset = load_dataset("huggingface/cats-image", split="test")
...
venv/lib/python3.12/site-packages/datasets/utils/_dill.py:42: in save
    if issubclass(obj_type, spacy.Language):
AttributeError: module 'spacy' has no attribute 'Language'
```

## Root cause
`DynamicLoader.setup_models_path()` in `tests/runner/utils/dynamic_loader.py` inserted `models_root` (the `tt_forge_models/` directory) at the front of `sys.path`. Because `tt_forge_models/spacy/` is a directory without `__init__.py`, Python created a namespace package named `spacy` from it. When any loader triggers `import spacy` (e.g. `huspacy` during model discovery), this empty namespace package was registered in `sys.modules['spacy']` instead of the real spaCy library. Later, `datasets._dill.save()` checks `issubclass(obj_type, spacy.Language)` and fails with `AttributeError: module 'spacy' has no attribute 'Language'`. The `clip_vit_eurosat` loader calls `load_dataset()` which triggers the `_dill` code path.

## Fix
Removed the `sys.path.insert(0, models_root)` call from `setup_models_path` in `tests/runner/utils/dynamic_loader.py`. Relative imports in model loaders work via `__package__` and the manually-registered `tt_forge_models` namespace module (which already sets `__path__ = [models_root]`); the `sys.path` insertion was not needed for them.

File changed: `tt-xla/tests/runner/utils/dynamic_loader.py`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    46.53s
- Tier A attempts: N/A

## Files changed
- tt-xla/tests/runner/utils/dynamic_loader.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | ed4f8448a1267750f1abc3753fc166d853633db2 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
