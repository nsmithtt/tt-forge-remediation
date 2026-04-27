# Summary: bird_species_classifier/pytorch-Default-single_device-inference

**Test**: `tests/runner/test_models.py::test_all_models_torch[bird_species_classifier/pytorch-Default-single_device-inference]`
**Result**: SILICON_PASS

## Original Failure

The test was failing with:
```
The image processor of type `EfficientNetImageProcessor` is now loaded as a fast processor
by default, even if the model checkpoint was saved with a slow processor. This is a breaking
change and may produce slightly different outputs. To continue using the slow processor,
instantiate this class with `use_fast=False`.
```

## Root Causes and Fixes

Three issues were found and fixed in `bird_species_classifier/pytorch/loader.py`:

### 1. Fast Image Processor Breaking Change
`AutoImageProcessor.from_pretrained` now loads `EfficientNetImageProcessor` as a fast processor
by default in transformers 5.x. The fast processor produces different outputs.

**Fix**: Added `use_fast=False` to `AutoImageProcessor.from_pretrained(...)`.

### 2. Spacy Namespace Conflict with `load_dataset`
The local `spacy/` model directory in tt_forge_models acts as a namespace package and shadows
the real `spacy` package, causing the `datasets` library's `dill` serializer to fail with
`AttributeError: module 'spacy' has no attribute 'Language'` when pickling dataset file configs.

**Fix**: Replaced `load_dataset("huggingface/cats-image")` with `PIL.Image.new("RGB", (224, 224))`
as a dummy input image, avoiding the `datasets` library entirely.

### 3. `AvgPool2d` with `ceil_mode=True` Not Supported by TT XLA Backend
The EfficientNet model in transformers uses `nn.AvgPool2d(config.hidden_dim, ceil_mode=True)`
for global average pooling. On TT silicon via the XLA StableHLO backend, `ceil_mode=True` is
not properly supported, causing the pooler to output a 0-element tensor (0x0 spatial dims)
instead of 1x1, which then makes the subsequent `reshape([1, 1408])` fail with
`RuntimeError: shape '[1, 1408]' is invalid for input of size 0`.

**Fix**: After loading the model, replaced the pooler:
```python
model.efficientnet.pooler = nn.AdaptiveAvgPool2d((1, 1))
```
`AdaptiveAvgPool2d((1, 1))` is equivalent for this use case and is fully supported.

## Changes

- **tt_forge_models** branch: `remediation/bird-species-classifier-efficientnet-use-fast`
  - `bird_species_classifier/pytorch/loader.py`: all three fixes above

- **tt-xla** branch: `arch-c-36-tt-xla-dev/nsmith/2026-04-22_16-58/hf-bringup-35-fix`
  - Updated `third_party/tt_forge_models` submodule pointer
