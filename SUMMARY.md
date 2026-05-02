# Remediation Summary: mobilenetv1-pytorch-Optimum_Mobilenet_v1_0.75_192-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mobilenetv1/pytorch-Optimum_Mobilenet_v1_0.75_192-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
load-dataset-spacy-dill-namespace-collision

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
Raised inside `datasets/utils/_dill.py` during dill fingerprinting triggered by `load_dataset("huggingface/cats-image", split="test")`.

(The reported failure message `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute` is an unrelated warning emitted at session end; the actual pytest FAILED reason is the AttributeError above.)

## Root cause
`tt_forge_models/spacy/` is a namespace package that gets placed in `sys.modules['spacy']` before the real spacy package is imported. When `datasets` attempts to fingerprint the data-files config using `dill`, it executes `issubclass(obj_type, spacy.Language)` — but `sys.modules['spacy']` is the stub namespace package which has no `Language` attribute, so the call raises `AttributeError`.

The loader's `load_inputs` method called `load_dataset("huggingface/cats-image", split="test")` to obtain a sample PIL Image for preprocessing. This is unnecessary because `VisionPreprocessor.preprocess(image=None, ...)` already handles the `None` case by downloading a default COCO image (`http://images.cocodataset.org/val2017/000000039769.jpg`).

## Fix
**`tt_forge_models/mobilenetv1/pytorch/loader.py`**
- Removed `from datasets import load_dataset` import.
- Simplified `load_inputs` to pass `image` directly to `input_preprocess` without the `load_dataset` call; when `image=None` the `VisionPreprocessor` falls back to its built-in COCO default URL.

## Verification
- pytest exit: PASS
- Hardware:    wormhole
- Duration:    98.63s
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/mobilenetv1/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 43ee5b2d9af2634aa3e523c136f4ea7b5b734637 |
| tt-forge-models | 99856dd30fea2dfbdeee5667621d69fc398b4b02 |
