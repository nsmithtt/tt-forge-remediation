# Remediation Summary

## Test

`tests/runner/test_models.py::test_all_models_torch[align/pytorch-Base-single_device-inference]`

**Result: PASSED**

## Root Cause

Three issues were found and fixed in `tt_forge_models/align/pytorch/loader.py` and `tt_forge_models/huspacy/pytorch/loader.py`:

### Issue 1: spacy namespace package collision (huspacy loader)

`huspacy/pytorch/loader.py` imported `spacy` at module level. Because `third_party/tt_forge_models/` is added to `sys.path` by the test runner's dynamic loader, and a `spacy/` subdirectory exists under `tt_forge_models/` (the spaCy model loader directory), Python created a namespace package for `spacy` that shadowed the real library. This caused `datasets._dill` to find `spacy` in `sys.modules` but fail when it tried to access `spacy.Language`.

**Fix:** Moved `import spacy` inside `_load_nlp()` so it is only imported when actually needed, not during test collection.

### Issue 2: align loader used load_dataset triggering the spacy conflict

`align/pytorch/loader.py` called `load_dataset("huggingface/cats-image")` to get a sample image. This triggered `datasets._dill` pickling which checked `sys.modules` for `spacy` and failed (due to the namespace package from Issue 1).

**Fix:** Replaced `load_dataset(...)` with `PIL.Image.new("RGB", (224, 224), ...)` to use a synthetic image, eliminating the datasets pickling path entirely.

### Issue 3: EfficientNetImageProcessor fast processor warning

`AlignProcessor.from_pretrained()` now loads `EfficientNetImageProcessor` as a fast (Rust-based) processor by default, producing a deprecation warning about behavioral differences vs the slow processor.

**Fix:** Added `use_fast=False` to `AlignProcessor.from_pretrained()` to use the original slow processor.

### Issue 4: AvgPool2d with kernel > input on XLA (main compilation failure)

The ALIGN model's vision model uses `nn.AvgPool2d(640, ceil_mode=True)` as its pooler. The EfficientNet backbone reduces the 289×289 input to a 9×9 feature map. When `AvgPool2d(640, ...)` is applied to this 9×9 map (kernel >> input), XLA's aten `avg_pool2d` implementation returns an empty tensor instead of applying global average pooling, causing a `RuntimeError: shape '[1, 640]' is invalid for input of size 0`.

**Fix:** Replaced `model.vision_model.pooler` with `nn.AdaptiveAvgPool2d(1)` which performs numerically equivalent global average pooling but uses an ATEN op that XLA handles correctly.

## Changes

All changes are in the `tt_forge_models` submodule (branch `nsmith/fix-align-spacy-namespace`):

- `huspacy/pytorch/loader.py`: lazy-import spacy inside `_load_nlp()`
- `align/pytorch/loader.py`:
  - Replace `load_dataset` with `PIL.Image.new` for sample image
  - Add `use_fast=False` to `AlignProcessor.from_pretrained()`
  - Replace `model.vision_model.pooler` with `nn.AdaptiveAvgPool2d(1)`

## Submodule Hashes

| Submodule | Commit |
|-----------|--------|
| tt-xla | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-mlir | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-metal | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt_forge_models (nsmith/fix-align-spacy-namespace) | 93f4d36b69713d6c7dd2f69296da042099a2f357 |
