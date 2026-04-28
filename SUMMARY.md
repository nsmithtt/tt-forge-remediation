# Remediation Summary: cxformer/feature_extraction/pytorch-Small-single_device-inference

## Skill version
9

## Test
`tests/runner/test_models.py::test_all_models_torch[cxformer/feature_extraction/pytorch-Small-single_device-inference]`

## Result
SILICON_PASS

## Failure
```
The image processor of type `CustomBitImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.
```

## Root cause
**Loader layer** — three issues in `cxformer/feature_extraction/pytorch/loader.py`:

1. **Missing `use_fast=False`**: `AutoImageProcessor.from_pretrained` was called without `use_fast=False`. In transformers 5.x, this triggers a `FutureWarning` that becomes the reported failure and also causes transformers to attempt loading `CustomBitImageProcessorFast` (which does not exist in the remote model code). Adding `use_fast=False` suppresses the warning and loads the correct slow processor.

2. **`datasets` dependency conflict**: `load_dataset("huggingface/cats-image")` was used to obtain a sample image. This pulled in the `datasets` package, which during pickling checks `if "spacy" in sys.modules` and then attempts `spacy.Language`. Because `dynamic_loader.setup_models_path()` adds `third_party/tt_forge_models` to `sys.path`, Python's namespace-package resolution makes `tt_forge_models/spacy` (a model directory without `__init__.py`) visible as the `spacy` module — causing `AttributeError: module 'spacy' has no attribute 'Language'`. Replacing `load_dataset` with `PIL.Image.new` eliminates the dependency and the conflict.

3. **Missing `requirements.txt`**: The remote `custom_processor.py` imports `cv2`, but no `requirements.txt` existed for the loader. Added `opencv-python-headless`.

## Fix
**Repo**: `tt-forge-models`, branch `remediation/cxformer-feature_extraction-pytorch-Small-single_device-inference`

- `cxformer/feature_extraction/pytorch/loader.py`:
  - Removed `from datasets import load_dataset`
  - Added `from PIL import Image`
  - Added `use_fast=False` to `AutoImageProcessor.from_pretrained`
  - Replaced `load_dataset("huggingface/cats-image")[...]` with `Image.new("RGB", (224, 224), color=(128, 128, 128))`
- `cxformer/feature_extraction/pytorch/requirements.txt` (new): `opencv-python-headless`

None of these are forbidden workarounds: no model trimming, no CPU offload, no shape changes, no PCC lowering. The full model runs on TT silicon with unmodified weights and architecture.

## Verification
- pytest exit status: **PASSED**
- Wall-clock duration: **52.30 s**
- Hardware: **n150**

## Files changed
- `cxformer/feature_extraction/pytorch/loader.py`
- `cxformer/feature_extraction/pytorch/requirements.txt` (new)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | ebe39917e5da1edd373e303f68ca54f2c303ecaa |
