# Remediation Summary: clipseg/pytorch-Rd64_Refined_Fp16-single_device-inference

## Skill version
2

## Test
tests/runner/test_models.py::test_all_models_torch[clipseg/pytorch-Rd64_Refined_Fp16-single_device-inference]

## Result
SILICON_PASS

## Failure
The image processor of type `ViTImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.

## Root cause
Loader layer (tt_forge_models), three issues stacked:

1. **spacy namespace shadowing**: `huspacy/pytorch/loader.py` had a top-level `import spacy` that resolved to `tt_forge_models/spacy/` (a namespace directory on sys.path) instead of the real spacy package. The `datasets` fingerprinting code then crashed with `AttributeError: module 'spacy' has no attribute 'Language'` when trying to hash the dataset builder.

2. **ViTImageProcessor fast-processor breaking change**: `CLIPSegProcessor.from_pretrained()` was called without `use_fast=False`. transformers 5.x now loads the image processor as a fast processor by default, which is a breaking change.

3. **`return_dict=False` propagation**: `CLIPSegForImageSegmentation.from_pretrained(..., return_dict=False)` set `config.return_dict=False` on the model, which propagated to all internal sub-model calls. The model's `get_conditional_embeddings()` method then called the text encoder (which returned a tuple) and crashed with `AttributeError: 'tuple' object has no attribute 'pooler_output'`.

## Fix
All three fixes are in `tt_forge_models` (loader layer):

1. Moved `import spacy` from top-level to inside `_load_nlp()` in `huspacy/pytorch/loader.py`, and added `spacy/__init__.py` to make the directory a proper package — this prevents the namespace from shadowing the real spacy module.

2. Added `use_fast=False` to `CLIPSegProcessor.from_pretrained()` in `clipseg/pytorch/loader.py` — addresses transformers 5.x breaking change.

3. Removed `return_dict=False` from `model_kwargs` in `clipseg/pytorch/loader.py`'s `load_model()` — the kwarg was being passed to `from_pretrained` which set it on the model config, causing all internal sub-model calls to return tuples instead of dataclasses.

None of these are forbidden workarounds: they are all legitimate loader-layer fixes for transformers 5.x breaking changes and namespace management.

## Verification
pytest exit code: 0 (PASSED)
Wall-clock duration: 135.45s (2:15)
Hardware: n150

## Files changed
- `tt_forge_models/huspacy/pytorch/loader.py` — lazy-import spacy
- `tt_forge_models/spacy/__init__.py` — new file, makes spacy/ a proper package
- `tt_forge_models/clipseg/pytorch/loader.py` — use_fast=False, remove return_dict=False

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 2b2e332c0ca8fcfb65a3af0f36156e468639abbe |
| tt-forge-models | f7b1cf9eeab559434f7642720cbd6064368f9497 |
