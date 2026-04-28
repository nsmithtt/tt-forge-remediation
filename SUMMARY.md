# Remediation Summary: efficientnet-image_classification-pytorch-B1-single_device-inference

## Skill version
11

## Test
tests/runner/test_models.py::test_all_models_torch[efficientnet/image_classification/pytorch-B1-single_device-inference]

## Result
FAIL — `AvgPool2d(kernel_size=1280, ceil_mode=True)` returns a 0-size tensor in the TT XLA compilation path, causing a subsequent `view` to `[1, 1280]` to crash

## Failure
The loader originally failed with two issues fixed in this remediation:

1. **Loader failure (fixed):** `AttributeError: module 'spacy' has no attribute 'Language'`
   Caused by `load_dataset("huggingface/cats-image")` in `load_inputs`. The `datasets` library tries `import spacy` and finds the `tt_forge_models/spacy/` namespace package (on `sys.path` via the model discovery mechanism) instead of real spacy. `datasets._dill` then fails with `AttributeError` when it accesses `spacy.Language`.

2. **Loader warning (fixed):** `The image processor of type EfficientNetImageProcessor is now loaded as a fast processor by default` — transformers 5.x breaking change requiring `use_fast=False`.

3. **Compiler-stack failure (not fixed):** After the loader fixes the test reaches compilation and fails:
   ```
   RuntimeError: shape '[1, 1280]' is invalid for input of size 0
   While executing %view : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%avg_pool2d, [1, 1280]), kwargs = {})
   ```

## Root cause

**Compiler frontend (tt-xla) — `AvgPool2d(ceil_mode=True)` output-shape bug**

The EfficientNet model's pooler is:
```python
self.pooler = nn.AvgPool2d(config.hidden_dim, ceil_mode=True)
# → AvgPool2d(kernel_size=1280, stride=1280, padding=0, ceil_mode=True)
```

For B1 with 240×240 input, the feature map before pooling is `[1, 1280, 8, 8]`.

- With `ceil_mode=False`: output_h = floor((8 − 1280) / 1280 + 1) = 0 → error
- With `ceil_mode=True`:  output_h = ceil((8 − 1280) / 1280 + 1) = ceil(0.00625) = 1 → `[1, 1280, 1, 1]` ✓

CPU PyTorch handles `ceil_mode=True` correctly. The TT XLA compilation path (`dynamo_bridge.partition_fx_graph_for_cpu_fallback` → `collector.run`) computes output shape as 0, causing the subsequent `view([1, 1280])` to raise `RuntimeError: invalid shape`.

The bug is in how the tt-xla compiler frontend lowers `aten.avg_pool2d` with `ceil_mode=True` to StableHLO — the StableHLO `reduce_window` (or equivalent) does not implement the ceil rounding for output-size computation.

## Fix

**Loader fixes (applied, in tt_forge_models):**
- Replaced `load_dataset("huggingface/cats-image")` with `PIL.Image.new("RGB", (240, 240), ...)` to avoid the spacy namespace-package conflict.
- Added `use_fast=False` to `AutoImageProcessor.from_pretrained` to comply with transformers 5.x.
- Replaced deprecated `torch_dtype` kwarg with `dtype` in `AutoModelForImageClassification.from_pretrained`.

**Compiler fix (proposed, in tt-xla):**
The `aten.avg_pool2d` lowering in tt-xla (or the StableHLO bridge) must honor the `ceil_mode` argument when computing output window dimensions. The fix should produce output size 1 (not 0) for the `ceil_mode=True` case where kernel > input.

## Verification
FAIL — test does not pass after loader fixes. Hardware: blackhole (p150).

## Files changed
- `tt_forge_models/efficientnet/image_classification/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | fb3a96c7c3c7942246c092048e8f1216f705cdc7 |
| tt-forge-models | 36f3e02332 |
