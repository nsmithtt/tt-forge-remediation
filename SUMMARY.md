# Remediation Summary: dinov2/feature_extraction/pytorch-XRay_Base-single_device-inference

## Skill version
2

## Test
tests/runner/test_models.py::test_all_models_torch[dinov2/feature_extraction/pytorch-XRay_Base-single_device-inference]

## Result
SILICON_PASS

## Failure
The image processor of type `BlipImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.

## Root cause
Two loader-layer issues in `tt_forge_models`:

1. **`huspacy/pytorch/loader.py`** imports `spacy` at module level. During pytest collection the dynamic loader imports this module, placing the `tt_forge_models/spacy` namespace package (a model directory on `sys.path`) into `sys.modules['spacy']`. Later, `datasets/dill` checks `"spacy" in sys.modules` and tries `spacy.Language`, which does not exist on the stub namespace package → `AttributeError`. This prevented `load_dataset("huggingface/cats-image")` from completing in the dinov2 loader.

2. **`dinov2/feature_extraction/pytorch/loader.py`** calls `AutoImageProcessor.from_pretrained` without `use_fast=False` for the `XRay_Base` variant (`StanfordAIMI/dinov2-base-xray-224`). That checkpoint stores a `BlipImageProcessor` config. In transformers 5.x, the fast variant is now selected by default, changing preprocessing behavior and producing different pixel values — causing the PCC check to fail (or the breaking-change warning to surface as the primary visible failure).

## Fix
**Fix 1** (`huspacy/pytorch/loader.py`): Moved `import spacy` from module level to a lazy import inside `_load_nlp()`. This defers the import to runtime, so collection no longer pollutes `sys.modules['spacy']` with the stub namespace package, and `datasets/dill` no longer encounters the missing `Language` attribute.

**Fix 2** (`dinov2/feature_extraction/pytorch/loader.py`): Added `use_fast=False` to `AutoImageProcessor.from_pretrained` for the `XRAY_BASE` variant. This is the transformers 5.x breaking-change fix explicitly listed as a legitimate loader-layer fix in the skill rules.

Neither change trims the model, offloads sub-modules, skips the vision path, adjusts input shapes, lowers `required_pcc`, or suppresses warnings.

## Verification
pytest exit status: PASSED
Wall-clock duration: 55.78s
Hardware: n150 (wormhole_b0)

## Files changed
- `tt_forge_models/huspacy/pytorch/loader.py` — lazy `import spacy`
- `tt_forge_models/dinov2/feature_extraction/pytorch/loader.py` — `use_fast=False` for XRay_Base

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 1d1800a39c34e7f1ed6fb516baa9e2602d101f75 |
| tt-forge-models | 8c7d11ee409f934f33ae453017a26715a0449f59 |
