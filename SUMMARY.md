# Remediation Summary: dinov3-feature_extraction-pytorch-small_plus_patch16_timm-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[dinov3/feature_extraction/pytorch-Small_Plus_Patch16_TIMM-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
load-dataset-spacy-namespace-package-pollution

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
venv/lib/python3.12/site-packages/datasets/utils/_dill.py:42: if issubclass(obj_type, spacy.Language):
E   AttributeError: module 'spacy' has no attribute 'Language'
```

Surfaced as: `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute` (the trailing warning after the test failed).

## Root cause
`huspacy/pytorch/loader.py` had `import spacy` at module level. During pytest collection, this executes before any test runs, importing `tt_forge_models/spacy/` (a namespace directory on sys.path) as `sys.modules['spacy']` instead of the real spaCy package. Later, when the dinov3 TIMM loader calls `load_dataset("huggingface/cats-image", split="test")`, the `datasets` dill hasher checks `issubclass(obj_type, spacy.Language)`, but the namespace package has no `Language` attribute, causing the AttributeError.

## Fix
In `tt_forge_models/huspacy/pytorch/loader.py`:
- Removed `import spacy` from the module-level imports
- Added `import spacy` as a lazy import inside the `_load_nlp()` method, so it is only imported when actually loading the huspacy model (not during pytest collection)

Repository: `tt-forge-models`
Branch: `remediation/dinov3-feature_extraction-pytorch-small_plus_patch16_timm-single_device-inference`
Commit: `42abeba498`
File: `huspacy/pytorch/loader.py`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    60.30s
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/huspacy/pytorch/loader.py` — made `import spacy` lazy (moved into `_load_nlp`)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 14cb2a6e303b0e4d90e96429f1a6c6d28ce20bc2 |
| tt-forge-models | 42abeba498c2c5a6f95c35024fda2b4e22c9f41d |
