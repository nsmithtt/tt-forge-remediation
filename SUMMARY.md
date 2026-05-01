# Remediation Summary: efficientnetv2-pytorch-Tf_Efficientnetv2_M.in21k_ft_in1k-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[efficientnetv2/pytorch-Tf_Efficientnetv2_M.in21k_ft_in1k-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
load-dataset-spacy-namespace-pollution

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
AttributeError: module 'spacy' has no attribute 'Language'
```

Traceback from `datasets/utils/_dill.py:42`: `if issubclass(obj_type, spacy.Language):`

Reported at test session end as: `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute`

## Root cause
The `tt_forge_models/spacy/` directory (a namespace package for the SpaCy NER model loader) is present in the Python path, causing `sys.modules['spacy']` to be populated with a stub namespace package that has no `Language` attribute.

When `load_dataset("huggingface/cats-image")` is called, the `datasets` library uses `dill` to fingerprint its arguments. During serialization, `datasets/utils/_dill.py` checks `issubclass(obj_type, spacy.Language)` to decide how to serialize objects. Since `sys.modules['spacy']` is the stub namespace package rather than the real spaCy library, this raises `AttributeError: module 'spacy' has no attribute 'Language'`.

The fix is a loader-level change: replace `load_dataset` with a direct `PIL.Image.open` via `requests` so the `dill` fingerprinting path that checks `spacy.Language` is never triggered.

## Fix
In `tt_forge_models/efficientnetv2/pytorch/loader.py`:
- Removed `from datasets import load_dataset`
- Added `import requests` and `from PIL import Image`
- Replaced `dataset = load_dataset("huggingface/cats-image")["test"]; image = dataset[0]["image"].convert("RGB")` with `image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw).convert("RGB")`

Committed on branch `remediation/efficientnetv2-pytorch-Tf_Efficientnetv2_M.in21k_ft_in1k-single_device-inference` in tt-forge-models at commit `90a625ed7bf3c63ac42eaa52ce36e0bfe121c6ce`.

## Verification
- pytest exit: PASS
- Hardware:    wormhole
- Duration:    213.92s (0:03:33)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/efficientnetv2/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 8adb3fc100c976e4adc3e115fb0ba400975a5426 |
| tt-forge-models | 90a625ed7bf3c63ac42eaa52ce36e0bfe121c6ce |
