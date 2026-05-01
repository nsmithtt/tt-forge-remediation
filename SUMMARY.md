# Remediation Summary: mexma_siglip2-image_text_similarity-pytorch-visheratin-mexma-siglip2-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mexma_siglip2/image_text_similarity/pytorch-visheratin/mexma-siglip2-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
transformers-5x-all-tied-weights-keys-missing

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
AttributeError: 'MexmaSigLIP' object has no attribute 'all_tied_weights_keys'. Did you mean: '_tied_weights_keys'?

(The originally reported failure message "sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute" is a trailing pytest warning summary line; the real error was the AttributeError above.)

## Root cause
Two loader-layer bugs:

1. **`all_tied_weights_keys` missing (transformers 5.x)**: The `MexmaSigLIP` remote model class (`visheratin/mexma-siglip2`) inherits from `PreTrainedModel` but its `__init__` does not call `self.post_init()`. In transformers 5.x, `post_init()` initializes `self.all_tied_weights_keys = {}`, which `_finalize_model_loading` then accesses via `_adjust_tied_keys_with_tied_pointers`. Because `post_init()` was never called, the attribute is missing, crashing at load time.

2. **`spacy` namespace shadowing (known bug)**: After fixing bug 1, `load_inputs` called `load_dataset("huggingface/cats-image")` which triggered `datasets._dill` to check `spacy.Language`. The `dynamic_loader.py::setup_models_path` was inserting `models_root` into `sys.path`, causing `tt_forge_models/spacy/` (a directory without `__init__.py`) to shadow the real spaCy package, so `spacy.Language` raised `AttributeError`.

## Fix

**Fix 1** — `tt_forge_models/mexma_siglip2/image_text_similarity/pytorch/loader.py`:
Wrapped `AutoModel.from_pretrained(...)` with a temporary patch of `PreTrainedModel._adjust_tied_keys_with_tied_pointers` that initializes `self.all_tied_weights_keys = {}` if the attribute is missing. Restored after the call via `try/finally`.

**Fix 2** — `tt-xla/tests/runner/utils/dynamic_loader.py`:
Removed the `sys.path.insert(0, models_root)` call from `setup_models_path`. Relative imports in loaders work via the manually registered `tt_forge_models` namespace package; the `sys.path` insertion was unnecessary and caused the spacy namespace shadowing.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    167.31s (0:02:47)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/mexma_siglip2/image_text_similarity/pytorch/loader.py` (loader fix 1)
- `tt-xla/tests/runner/utils/dynamic_loader.py` (loader fix 2)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 158702c5bf81217f75608855ce5c822bc1f7dbc3 |
| tt-forge-models | bec599261064481b8c7173c31baa3d88e05f96fb |
