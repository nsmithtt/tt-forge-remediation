# Remediation Summary: clipseg-pytorch-Rd64_Refined-single_device-inference

## Skill version
2

## Test
tests/runner/test_models.py::test_all_models_torch[clipseg/pytorch-Rd64_Refined-single_device-inference]

## Result
SILICON_PASS

## Failure
The image processor of type `ViTImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.

## Root cause
Two cascading loader-layer bugs in `tt_forge_models`, both introduced by transformers 5.x:

**Bug 1 — spacy namespace shadowing (huspacy loader)**
`setup_models_path` in the test runner inserts `tt_forge_models/` into `sys.path` so
relative model imports work.  The `huspacy/pytorch/loader.py` had a top-level
`import spacy`, which Python resolved to the `tt_forge_models/spacy/` directory
(a namespace package) instead of the real spaCy package (which is not installed).
This registered a bare `spacy` namespace module in `sys.modules["spacy"]` — one
with no `Language` attribute.  Later, when clipseg's `load_inputs` called
`load_dataset("huggingface/cats-image")`, the `dill` serialiser checked
`if "spacy" in sys.modules:` (True), then tried `issubclass(obj_type, spacy.Language)`
→ `AttributeError: module 'spacy' has no attribute 'Language'`.

**Bug 2 — return_dict=False propagation (clipseg loader)**
After fixing Bug 1, the next failure was:
`AttributeError: 'tuple' object has no attribute 'pooler_output'`

The clipseg loader passed `return_dict=False` to `CLIPSegForImageSegmentation.from_pretrained()`,
which sets `config.use_return_dict = False` globally on the model.  In transformers 5.x,
helper methods `get_text_features` and `get_image_features` are decorated with
`@can_return_tuple`, which converts their output to a tuple when `config.return_dict`
is False.  `get_conditional_embeddings` (called internally during forward) calls
these helpers without an explicit `return_dict` kwarg and then accesses `.pooler_output`
on the result — failing because the result is now a tuple, not a ModelOutput object.

**Bug 3 — ViTImageProcessor fast-processor warning (clipseg loader)**
`CLIPSegProcessor.from_pretrained()` without `use_fast=False` loads a fast
`ViTImageProcessor` by default in transformers 5.x even though the CIDAS/clipseg-rd64-refined
checkpoint was saved with a slow processor, producing a breaking-change warning and
potentially different outputs.

## Fix
All fixes are in `tt_forge_models`, `clipseg/pytorch/loader.py` and `huspacy/pytorch/loader.py`.

1. **huspacy/pytorch/loader.py**: moved `import spacy` from module level into
   `_load_nlp()`, so it is only imported when the huspacy model is actually loaded.
   This prevents the `tt_forge_models/spacy/` namespace package from polluting
   `sys.modules["spacy"]` at import time.

2. **clipseg/pytorch/loader.py**: removed `return_dict=False` from the model-loading
   kwargs.  Setting it via `from_pretrained` propagates into all internal sub-calls
   through the transformers 5.x `@can_return_tuple` decorator, breaking
   `get_conditional_embeddings`.  The model now returns a `CLIPSegOutput` object
   from `forward()`; the test framework handles ModelOutput objects correctly.

3. **clipseg/pytorch/loader.py**: added `use_fast=False` to
   `CLIPSegProcessor.from_pretrained()` so the slow `ViTImageProcessor` is loaded
   consistently with the saved checkpoint.

None of these changes trim the model, offload sub-modules, alter input shapes,
lower PCC thresholds, or suppress exceptions.

## Verification
pytest exit status: PASSED
Wall-clock duration: 127.69 s (2 m 7 s)
Hardware: n150 (wormhole_b0)

## Files changed
- `clipseg/pytorch/loader.py` (in tt_forge_models)
- `huspacy/pytorch/loader.py` (in tt_forge_models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 36c015c8a96ad2998afa4c5825a3403c5de15a7a |
| tt-forge-models | 56fc038b02f5b9b1beda4a5dce86f599bc994c16 |
