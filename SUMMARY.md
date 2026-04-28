# Remediation Summary: gender_classification-pytorch-Leilab-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[gender_classification/pytorch-Leilab-single_device-inference]

## Result
SILICON_PASS — two loader bugs fixed: ViTImageProcessor use_fast default and spacy namespace collision

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

(Additionally, the test was blocked first by `AttributeError: module 'spacy' has no attribute 'Language'` due to a namespace collision from the `spacy/` model directory under `tt_forge_models/`.)

## Root cause
Two loader-layer bugs:

1. **transformers-5x-use-fast-default**: `AutoImageProcessor.from_pretrained("Leilab/gender_class")` resolved to `ViTImageProcessor`, which in transformers 5.x defaults to loading the fast C++-backed processor. The `Leilab/gender_class` checkpoint was saved with the slow Python processor, so loading as fast is a breaking change that triggers a warning/error.

2. **Spacy namespace collision**: `tt_forge_models/spacy/` and `tt_forge_models/huspacy/` directories exist on `sys.path` during pytest collection. A module-level `import spacy` in `huspacy/pytorch/loader.py` resolved to the `spacy/` namespace package (a model directory, not the real spaCy library), populating `sys.modules['spacy']` without the `Language` attribute. When `load_dataset` later called the dill pickler, `datasets._dill` checked `issubclass(obj_type, spacy.Language)` and crashed.

## Fix
1. `gender_classification/pytorch/loader.py`: Added `use_fast=False` to `AutoImageProcessor.from_pretrained(pretrained_model_name)`.
2. `huspacy/pytorch/loader.py`: Moved `import spacy` from module level into the `_load_nlp()` method body to defer resolution until the real spaCy library is needed.

Both changes are in the `tt_forge_models` repo on branch `remediation/gender_classification-pytorch-Leilab-single_device-inference`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    55.17s
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/gender_classification/pytorch/loader.py`
- `tt_forge_models/huspacy/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 4981e79205dde3e7ca665459e1b0b08cbcb0218f |
| tt-forge-models | 53843943718c86a18283bd114942e645bfbfa798 |
