# Remediation Summary: aiornot-image_classification-pytorch-AIorNot-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[aiornot/image_classification/pytorch-AIorNot-single_device-inference]

## Result
SILICON_PASS — two loader-layer bugs fixed: ViTFeatureExtractor removal in transformers 5.x and sys.path spacy namespace shadowing

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
transformers-5x-vit-feature-extractor-removed, spacy-namespace-shadows-real-package

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The reported failure message was `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute`.

With the correct venv activation (`source venv/activate` sets PYTHONPATH to include `tests/`), the real failures are:

1. `ValueError: Unrecognized image processor in Nahrawy/AIorNot` — transformers 5.x removed `ViTFeatureExtractor`; `AutoImageProcessor` can no longer resolve `"image_processor_type": "ViTFeatureExtractor"` from the model's `preprocessor_config.json`.

2. `AttributeError: module 'spacy' has no attribute 'Language'` — `setup_models_path` in `dynamic_loader.py` added `models_root` (`tt_forge_models/`) to `sys.path`, making `tt_forge_models/spacy/` a namespace package named `spacy` that shadows the real spaCy library. `datasets._dill` then failed checking `spacy.Language`.

## Root cause
Both bugs are in the loader layer:

**Bug 1 (tt-forge-models loader):** `aiornot/image_classification/pytorch/loader.py` called `AutoImageProcessor.from_pretrained("Nahrawy/AIorNot")`. The model's `preprocessor_config.json` specifies `"image_processor_type": "ViTFeatureExtractor"`, a class that was removed in transformers 5.x. The successor class `ViTImageProcessor` works correctly for this model.

**Bug 2 (tt-xla test runner):** `tests/runner/utils/dynamic_loader.py::setup_models_path` called `sys.path.insert(0, models_root)`. Because `tt_forge_models/spacy/` exists as a directory without `__init__.py`, Python created a namespace package `spacy` from it. This shadowed the real spaCy library. The `sys.path.insert` was redundant: `tt_forge_models` is already registered explicitly as a namespace package in `sys.modules` on the next line.

## Fix
**Fix 1 — tt-forge-models** (`aiornot/image_classification/pytorch/loader.py`):
- Replaced `AutoImageProcessor` import and call with `ViTImageProcessor` (the canonical transformers 5.x replacement for `ViTFeatureExtractor`).
- Branch: `remediation/aiornot-image_classification-pytorch-AIorNot-single_device-inference`

**Fix 2 — tt-xla** (`tests/runner/utils/dynamic_loader.py`):
- Removed the `if models_root not in sys.path: sys.path.insert(0, models_root)` block from `setup_models_path`. The namespace package registration in `sys.modules` that immediately follows is sufficient for loader relative imports to resolve.
- Branch: `remediation/aiornot-image_classification-pytorch-AIorNot-single_device-inference`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    143.18s (0:02:23)
- Tier A attempts: N/A

## Files changed
- `aiornot/image_classification/pytorch/loader.py` (tt-forge-models)
- `tests/runner/utils/dynamic_loader.py` (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 1cd99572bd8c5d5908327cdd57a4237b657ee910 |
| tt-forge-models | 6cd8f629fd9fecfa4e8e0ea8f1839a046d2998bf |
