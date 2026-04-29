# Remediation Summary: clip_vit_base_patch32_sun397-pytorch-Base_Patch32_SUN397-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[clip_vit_base_patch32_sun397/pytorch-Base_Patch32_SUN397-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
spacy-namespace-package-shadows-real-spacy-load-dataset

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute

## Root cause
The `spacy/` model directory in tt_forge_models creates a Python namespace package
that shadows the real `spacy` library when `models_root` is added to `sys.path`
during test discovery. The `load_inputs()` method called
`load_dataset("huggingface/cats-image")` which triggers `datasets._dill`
fingerprinting. Inside `_dill.py:42`, the serializer calls
`issubclass(obj_type, spacy.Language)`, which raises
`AttributeError: module 'spacy' has no attribute 'Language'` because `spacy`
resolves to the empty namespace package at `tt_forge_models/spacy/` instead of
the real spacy installation.

The `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute`
in the failure detail is a harmless Python 3.12 warning emitted by torch_xla's SWIG
C extension at process exit — it is the last stderr line printed after the real error
caused the test to fail, which is why the failure_category was classified as `unknown`
(the AttributeError occurred in the dataset fingerprinting path and was the last
recognizable error before this cosmetic warning).

## Fix
Replaced `load_dataset("huggingface/cats-image")` with a direct HTTP fetch of a
COCO image URL in `clip_vit_base_patch32_sun397/pytorch/loader.py`:

```python
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(BytesIO(requests.get(url, timeout=30).content))
```

Added `import requests`, `from io import BytesIO`, `from PIL import Image`;
removed `from datasets import load_dataset`.

This matches the identical fix applied to `clip_vit_base_patch32_svhn` (commit
`60ad46a1de`). The fix is in `tenstorrent/tt-forge-models` on branch
`remediation/clip_vit_base_patch32_sun397-pytorch-Base_Patch32_SUN397-single_device-inference`.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    48.95s
- Tier A attempts: N/A

## Files changed
- `clip_vit_base_patch32_sun397/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 280ec966ab12a078f4ab12c470fafdfd6918470e |
