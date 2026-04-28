# Remediation Summary: dinov2/image_classification/pytorch-Base_SkinDisease-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[dinov2/image_classification/pytorch-Base_SkinDisease-single_device-inference]

## Result
SILICON_PASS — two loader fixes: huspacy lazy-import and use_fast=False for Base_SkinDisease

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
The image processor of type `BitImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.

Underlying runtime failure: `AttributeError: module 'spacy' has no attribute 'Language'` in `datasets/utils/_dill.py` when calling `load_dataset("huggingface/cats-image")`.

## Root cause
Two loader-layer bugs:

1. **spacy namespace collision**: `third_party/tt_forge_models/spacy/` is a model directory. When `models_root` is added to `sys.path` by the dynamic loader, Python 3's namespace package mechanism makes it importable as `spacy`. `huspacy/pytorch/loader.py` ran `import spacy` at module level; during test discovery this placed the stub namespace package in `sys.modules["spacy"]`. The `datasets` library later found `"spacy"` in `sys.modules` and tried to access `spacy.Language` (to handle pickling), raising `AttributeError`. This caused `load_dataset("huggingface/cats-image")` to fail before any inference occurred.

2. **BitImageProcessor fast-processor default**: `Jayanth2002/dinov2-base-finetuned-SkinDisease` stores a `BitImageProcessor` config (the slow variant). In transformers 5.x, when `AutoImageProcessor.from_pretrained()` is called without `use_fast=False`, it now selects the fast `BitImageProcessorFast` by default, which may produce different preprocessing outputs affecting PCC.

## Fix
**tt_forge_models** (branch `remediation/dinov2-image_classification-pytorch-Base_SkinDisease-single_device-inference`):

1. `huspacy/pytorch/loader.py`: moved `import spacy` from module level into the `_load_nlp()` method. This defers the import to runtime, preventing the namespace package collision during test discovery.

2. `dinov2/image_classification/pytorch/loader.py`: added `use_fast=False` kwarg for the `BASE_SKIN_DISEASE` variant in `_load_processor()`, so `AutoImageProcessor.from_pretrained()` loads the original slow `BitImageProcessor` as the checkpoint specifies.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    61.80s (0:01:01)
- Tier A attempts: N/A

## Files changed
- `third_party/tt_forge_models/huspacy/pytorch/loader.py` — lazy-import spacy
- `third_party/tt_forge_models/dinov2/image_classification/pytorch/loader.py` — use_fast=False for Base_SkinDisease

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 95f6a5e09902604ce6201632610c321d2bee75f7 |
