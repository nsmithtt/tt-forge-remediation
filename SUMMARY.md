# Remediation Summary: efficientnet-image_classification-pytorch-B5-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[efficientnet/image_classification/pytorch-B5-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader + tt-xla

## Tier
A

## Bug fingerprint
avg-pool2d-ceil-mode-zero-output

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Two sequential failures:

**1. Loader failure (fixed first):**
```
AttributeError: module 'spacy' has no attribute 'Language'
```
Raised in `datasets/utils/_dill.py:42` during `load_dataset("huggingface/cats-image")`.
The `datasets` library checks `"spacy" in sys.modules` and then accesses `spacy.Language`.
Because the dynamic model loader inserts `third_party/tt_forge_models` into `sys.path[0]`,
Python finds `tt_forge_models/spacy/` as a namespace package (no `__init__.py` needed in
Python 3). The `huspacy/pytorch/loader.py` top-level `import spacy` then imports this stub
namespace package. Later, `datasets._dill` finds `"spacy"` in `sys.modules`, imports it, and
fails with `AttributeError: module 'spacy' has no attribute 'Language'`.

Additionally, the image processor loaded with a transformers 5.x breaking-change warning:
```
The image processor of type `EfficientNetImageProcessor` is now loaded as a fast processor
by default, even if the model checkpoint was saved with a slow processor.
```

**2. Compiler failure (fixed second, after loader fix):**
```
RuntimeError: shape '[1, 2048]' is invalid for input of size 0

While executing %view : [num_users=1] = call_function[target=torch.ops.aten.view.default](
    args = (%avg_pool2d, [1, 2048]), kwargs = {})
```

## Root cause

**Loader bugs:**

1. `load_dataset("huggingface/cats-image")` in `efficientnet/image_classification/pytorch/loader.py`
   triggers the spacy namespace-package conflict. The real spacy is not installed; the
   `tt_forge_models/spacy/` directory is imported as a namespace package via Python 3 implicit
   namespace packages (PEP 420). `datasets._dill` then fails accessing `spacy.Language`.

2. `AutoImageProcessor.from_pretrained(pretrained_model_name)` without `use_fast=False`:
   transformers 5.x changed the default to load fast processors, producing a warning and
   potentially different outputs.

**Compiler-stack bug (Tier A) — tt-xla `decompositions.py`:**

EfficientNet-B5's pooler is:
```python
self.pooler = nn.AvgPool2d(config.hidden_dim, ceil_mode=True)
# → AvgPool2d(kernel_size=2048, stride=2048, padding=0, ceil_mode=True)
```

For B5 with 456×456 input, the feature map before pooling is `[1, 2048, 15, 15]`.
- `ceil_mode=False`: `output_h = floor((15 − 2048)/2048 + 1) = floor(0.0073) = 0` → empty
- `ceil_mode=True`:  `output_h = ceil((15 − 2048)/2048 + 1) = ceil(0.0073) = 1` → `[1, 2048, 1, 1]` ✓

The existing `avg_pool2d` decomposition in
`tt-xla/python_package/tt_torch/backend/decompositions.py` only handled the exact-match
case (`stride == kernel_size == input_size`). When `kernel_size > input_size`, it returned
`NotImplemented`, delegating to torch-xla's default lowering which computes output size 0
for the `ceil_mode=True` case and produces an empty tensor.

## Fix

**Loader fixes (in tt_forge_models, commit `5cc4ce7d70`):**
- Replaced `load_dataset("huggingface/cats-image")` with `PIL.Image.new("RGB", (240, 240), ...)`
  to avoid the spacy namespace-package conflict entirely.
- Added `use_fast=False` to `AutoImageProcessor.from_pretrained` (transformers 5.x requirement).
- Replaced deprecated `torch_dtype` kwarg with `dtype` in `AutoModelForImageClassification.from_pretrained`.

**Compiler fix (in tt-xla, commit `27e945f9a0`):**
Extended the `avg_pool2d` decomposition in
`python_package/tt_torch/backend/decompositions.py` to also handle the `ceil_mode=True`
global-pool case. When `padding==0`, `kernel_size >= input_size`, and no `divisor_override`,
the single output window covers the entire spatial input, making the result equivalent to
`input.mean(dim=[-2, -1], keepdim=True)` (verified against CPU PyTorch output).

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    251.73s (0:04:11)
- Tier A attempts: 1

## Files changed
- `tt_forge_models/efficientnet/image_classification/pytorch/loader.py`
- `tt-xla/python_package/tt_torch/backend/decompositions.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 27e945f9a05f6c3ed16bd3b0bb9da6f328cea60e |
| tt-forge-models | 5cc4ce7d70e5d690e7feb24ae01a59ee028edf61 |
