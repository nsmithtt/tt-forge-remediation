# Remediation Summary: deep_fake_detector_v2-image_classification-pytorch-Deep-Fake-Detector-v2-Model-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deep_fake_detector_v2/image_classification/pytorch-Deep-Fake-Detector-v2-Model-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
transformers-5x-use-fast-default

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The image processor of type `ViTImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.

Additionally, in the local test environment: `AttributeError: module 'spacy' has no attribute 'Language'` caused by `huspacy/loader.py` importing `spacy` at module level during test collection. Because `third_party/tt_forge_models` is added to `sys.path` by the dynamic loader, the `tt_forge_models/spacy/` subdirectory is found as a Python 3 namespace package, poisoning `sys.modules['spacy']` without the real `spacy.Language` class. When `datasets` library (via dill) later checks `if "spacy" in sys.modules` and accesses `spacy.Language`, it raises an `AttributeError`.

## Root cause
Two loader-layer bugs:

1. **transformers 5.x use_fast default** (`loader` layer): `AutoImageProcessor.from_pretrained()` in transformers 5.2.0 now loads `ViTImageProcessorFast` by default when torchvision is available, instead of the slow `ViTImageProcessor` the checkpoint was saved with. The `deep_fake_detector_v2` loader called `AutoImageProcessor.from_pretrained()` without `use_fast=False`.

2. **spacy namespace package pollution** (`loader` layer): `huspacy/loader.py` had `import spacy` at module level. The dynamic loader adds `third_party/tt_forge_models` to `sys.path` during test collection. Python 3 namespace package discovery finds `tt_forge_models/spacy/` (a real model subdirectory without `__init__.py`) as a namespace package named `spacy`, and `import spacy` succeeds — but the resulting module has no `Language` attribute. Later, `datasets._dill.Pickler.save()` checks `"spacy" in sys.modules`, finds the namespace package, and fails with `AttributeError: module 'spacy' has no attribute 'Language'` when `load_dataset()` tries to hash its arguments.

## Fix
**Repository**: `tt-forge-models`  
**Branch**: `remediation/deep_fake_detector_v2-image_classification-pytorch-Deep-Fake-Detector-v2-Model-single_device-inference`  
**Commits**: `aaebb48ad2`, `e0f8413415`

1. `deep_fake_detector_v2/image_classification/pytorch/loader.py`: Added `use_fast=False` to `AutoImageProcessor.from_pretrained()` call so the slow `ViTImageProcessor` is loaded consistently.

2. `huspacy/pytorch/loader.py`: Moved `import spacy` from module level into `_load_nlp()` (lazy import). Added a guard that checks `hasattr(spacy, 'Language')` and pops the namespace package from `sys.modules` before raising `ImportError` if real spacy is not installed. This ensures that during test collection the `spacy` namespace package is never added to `sys.modules`, and if huspacy's own tests are run without real spacy, the error surfaces cleanly at runtime.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    48.57s
- Tier A attempts: N/A

## Files changed
- `deep_fake_detector_v2/image_classification/pytorch/loader.py`
- `huspacy/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | e0f8413415ee1e3a147b4fdb85df077e67d11561 |
