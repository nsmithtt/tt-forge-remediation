# Remediation Summary: gender_classification-pytorch-Crangana-single_device-inference

## Skill version
10

## Test
tests/runner/test_models.py::test_all_models_torch[gender_classification/pytorch-Crangana-single_device-inference]

## Result
SILICON_PASS

## Failure
```
ValueError: Unrecognized image processor in crangana/trained-gender. Should have a
`image_processor_type` key in its preprocessor_config.json of config.json, or one of
the following `model_type` keys in its config.json: ...convnext...
```

(Originally reported as a `ValueError` about `ConvNextFeatureExtractor` being loaded as a
fast processor; by the time of reproduction the message had evolved to "Unrecognized image
processor", but the root cause is the same.)

## Root cause

Two loader-layer bugs, both in `tt_forge_models`:

**Bug 1 – `gender_classification/pytorch/loader.py`:**
`AutoImageProcessor.from_pretrained` is called for `crangana/trained-gender`. The model's
`preprocessor_config.json` stores `image_processor_type: "ConvNextFeatureExtractor"`, the
legacy class name. In transformers 5.x the auto-mapping no longer recognises
`ConvNextFeatureExtractor` as a valid `image_processor_type`, so `from_pretrained` raises
`ValueError`. `ConvNextImageProcessor.from_pretrained` handles this config correctly.

**Bug 2 – `huspacy/pytorch/loader.py`:**
The huspacy loader does `import spacy` at module level. When pytest collects all model
loaders, `dynamic_loader.setup_models_path` inserts `third_party/tt_forge_models/` into
`sys.path`. Python then resolves `import spacy` to the `tt_forge_models/spacy/` directory
(a namespace package with no `__init__.py`), creating `sys.modules['spacy']` as an empty
namespace module without `Language`. Later, when `load_dataset("huggingface/cats-image")`
hashes its config with dill, `datasets/utils/_dill.py` checks `spacy.Language` and raises
`AttributeError: module 'spacy' has no attribute 'Language'`, aborting the test before the
model is ever run.

## Fix

**Fix 1** (`gender_classification/pytorch/loader.py`):
Replace `AutoImageProcessor.from_pretrained(pretrained_model_name)` with
`ConvNextImageProcessor.from_pretrained(pretrained_model_name)`. This targets the correct
processor class directly, bypassing the broken auto-mapping for the legacy class name.
Not a forbidden workaround: it is a standard transformers 5.x breaking-change adaptation.

**Fix 2** (`huspacy/pytorch/loader.py`):
Move `import spacy` from the module top level into the `_load_nlp()` method (lazy import).
This prevents `sys.modules['spacy']` from being populated with the wrong namespace package
during pytest collection. Not a forbidden workaround: it fixes the import-time side-effect
that corrupts the global module state.

Both fixes are in `tt_forge_models` on branch
`remediation/gender_classification-pytorch-Crangana-single_device-inference`.

## Verification
pytest exit status: **PASSED**
Wall-clock duration: **78.10 s**
Hardware: **n150** (single device, wormhole_b0)

## Files changed
- `gender_classification/pytorch/loader.py` — use `ConvNextImageProcessor` directly
- `huspacy/pytorch/loader.py` — lazy `import spacy` inside `_load_nlp()`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 4a58c9fe89f7c414ab1a34ccce8806509cfe00c8 |
| tt-forge-models | 80a7050b3ce99f3d026a7e6fbcc91a6a0c6d82cd |
