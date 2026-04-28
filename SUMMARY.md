# Remediation Summary: bad_anatomy_realism_classifier-image_classification-pytorch-BadAnatomyRealismClassifier-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bad_anatomy_realism_classifier/image_classification/pytorch-BadAnatomyRealismClassifier-single_device-inference]

## Result
SILICON_PASS — two loader-layer bugs fixed: missing preprocessor_config.json fallback and sys.path spacy namespace package collision

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
transformers-5x-missing-preprocessor-config-json

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
OSError: Can't load image processor for 'angusleung100/bad-anatomy-realism-classifier'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure 'angusleung100/bad-anatomy-realism-classifier' is the correct path to a directory containing a preprocessor_config.json file

(The original bug report showed `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute` — this is a SWIG warning that always appears when the test fails; the actual exception follows it.)

## Root cause
Two loader-layer bugs compounded:

1. **Missing preprocessor_config.json**: The HuggingFace repo `angusleung100/bad-anatomy-realism-classifier` has no `preprocessor_config.json`. In transformers 5.x, `AutoImageProcessor.from_pretrained` no longer falls back to the base model's `_name_or_path` when this file is absent; it raises an `OSError`. The base model (`google/vit-base-patch16-224-in21k`) is recorded in the raw `config.json` under `_name_or_path`.

2. **spacy namespace package collision**: After the preprocessor fix, `load_dataset("huggingface/cats-image")` failed with `AttributeError: module 'spacy' has no attribute 'Language'`. The `DynamicLoader.setup_models_path` in tt-xla was inserting `models_root` (= `third_party/tt_forge_models/`) into `sys.path`. Because `tt_forge_models/spacy/` is a directory without `__init__.py`, Python created a namespace package named `spacy` from it. The `datasets` library's `_dill.py` checks `if "spacy" in sys.modules` and then accesses `spacy.Language`, which doesn't exist on the stub namespace package. The `models_root` entry in `sys.path` was redundant — all relative imports in loaders are already handled via the `tt_forge_models` namespace package registered with `__path__ = [models_root]`.

## Fix
**Fix 1 — tt_forge_models loader** (`bad_anatomy_realism_classifier/image_classification/pytorch/loader.py`):
In `_load_processor`, catch `OSError` from `AutoImageProcessor.from_pretrained`. On failure, read the raw config dict via `PretrainedConfig.get_config_dict` (which preserves the original `_name_or_path` before transformers overrides it at load time) and retry with the base model name.

**Fix 2 — tt-xla test infrastructure** (`tests/runner/utils/dynamic_loader.py`):
Removed the `sys.path.insert(0, models_root)` call from `setup_models_path`. The namespace package registration at the same location (lines 210–219) already provides all relative-import support via `__path__`. Adding `models_root` to `sys.path` caused every subdirectory of `tt_forge_models` that lacks `__init__.py` to become a shadowing namespace package.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    20.69s
- Tier A attempts: N/A

## Files changed
- `third_party/tt_forge_models/bad_anatomy_realism_classifier/image_classification/pytorch/loader.py` (tt_forge_models repo, remediation branch)
- `tests/runner/utils/dynamic_loader.py` (tt-xla repo, remediation branch)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 6ff22bb4fce0fc1183c7382bc8e3da5abc70a0b7 |
| tt-forge-models | 16b9924ff2b88157065d0bf0bfb57b132dc58ce1 |
