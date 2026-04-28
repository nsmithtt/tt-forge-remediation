# Remediation Summary: cvt/pytorch-cvt-13-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[cvt/pytorch-cvt-13-single_device-inference]

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
The image processor of type `ConvNextImageProcessor` is now loaded as a fast
processor by default, even if the model checkpoint was saved with a slow
processor. This is a breaking change and may produce slightly different outputs.
To continue using the slow processor, instantiate this class with `use_fast=False`.

Additionally, a secondary failure occurred due to the `huspacy` loader importing
`spacy` at the top level. Since the dynamic loader adds `third_party/tt_forge_models`
to `sys.path`, Python resolves the `spacy/` subdirectory as an empty namespace package.
The `datasets` library then sees `"spacy" in sys.modules`, tries to access
`spacy.Language`, and fails with `AttributeError: module 'spacy' has no attribute 'Language'`.

## Root cause
Two loader-layer bugs in `tt_forge_models`:

1. **CvT loader (`cvt/pytorch/loader.py`)**: In transformers 5.x, `AutoImageProcessor.from_pretrained`
   now returns the fast variant (`ConvNextImageProcessorFast`) by default for models whose
   checkpoint was saved with the slow `ConvNextImageProcessor`. The fix is to pass
   `use_fast=False` to preserve the original processor type.

2. **HuSpaCy loader (`huspacy/pytorch/loader.py`)**: The module-level `import spacy`
   ran during test collection. Because the dynamic test loader adds
   `third_party/tt_forge_models` to `sys.path`, Python found the `spacy/` subdirectory
   in that directory as a namespace package (no `__init__.py`, no `Language` attribute).
   This stub was registered as `sys.modules['spacy']`, causing `datasets` to crash
   when loading the cats-image dataset for CvT inputs.

## Fix
Two changes in `tt_forge_models`, committed to branch
`remediation/cvt-pytorch-cvt-13-single-device-inference`:

1. `cvt/pytorch/loader.py`: Added `use_fast=False` to the `AutoImageProcessor.from_pretrained()` call in `_load_processor`.

2. `huspacy/pytorch/loader.py`: Moved `import spacy` from module top-level into the `_load_nlp()` method (lazy import), preventing namespace pollution of `sys.modules['spacy']` during test collection.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    109.70s (0:01:49)
- Tier A attempts: N/A

## Files changed
- `third_party/tt_forge_models/cvt/pytorch/loader.py`
- `third_party/tt_forge_models/huspacy/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a53cc30a8ff11e30ec5f9e7922a164f3ade28f5f |
| tt-forge-models | 162ff7a323f690fc60115bb8589ea694daff46b9 |
