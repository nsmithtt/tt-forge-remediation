# Remediation Summary: depth_anything_v2/pytorch-Small-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[depth_anything_v2/pytorch-Small-single_device-inference]

## Result
SILICON_PASS

## Failure
The image processor of type `DPTImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.

Secondary failure: `AttributeError: module 'spacy' has no attribute 'Language'` in `datasets._dill` when calling `load_dataset`.

## Root cause
Two loader-layer bugs:

1. **DPTImageProcessor fast processor (transformers 5.x breaking change):** `depth_anything_v2/pytorch/loader.py` called `AutoImageProcessor.from_pretrained(pretrained_model_name)` without `use_fast=False`. In transformers 5.x, `DPTImageProcessor` defaults to the fast variant, which prints a breaking-change message.

2. **spacy namespace package pollution:** `huspacy/pytorch/loader.py` had `import spacy` at module-level. The test infrastructure's `dynamic_loader.py` adds `tt_forge_models/` to `sys.path` during test collection so that model loader modules can use relative imports. This caused `import spacy` to resolve to the `tt_forge_models/spacy/` subdirectory (a namespace package) rather than the real spacy library. Once that fake module was in `sys.modules["spacy"]`, the `datasets` fingerprinting code (`_dill.py:42`) checked `if "spacy" in sys.modules` and then tried to call `spacy.Language`, which does not exist on the namespace package.

## Fix
Both fixes are in `tt-forge-models`:

1. `depth_anything_v2/pytorch/loader.py`: Added `use_fast=False` to `AutoImageProcessor.from_pretrained(...)` so the slow DPTImageProcessor is loaded, matching the checkpoint's saved configuration.

2. `huspacy/pytorch/loader.py`: Moved `import spacy` from module-level into the `_load_nlp()` method body. This defers the import to runtime (when the huspacy model is actually used), preventing the `tt_forge_models/spacy/` namespace package from being inserted into `sys.modules` during test collection.

Neither change trims the model, offloads submodules, or lowers `required_pcc`.

## Verification
pytest exit: PASSED  
Wall-clock duration: 122.51s (2:02)  
Hardware: n150 (wormhole_b0)

## Files changed
- `tt-forge-models/depth_anything_v2/pytorch/loader.py`
- `tt-forge-models/huspacy/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a6f7401ecffdc1fabc5f5bc6d7699769a0dccc4f |
| tt-forge-models | bb5e06ebc5b9f17d68906630867c5a28e9a41359 |
