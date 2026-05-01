# Remediation Summary: megadescriptor-feature_extraction-pytorch-Base_224-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[megadescriptor/feature_extraction/pytorch-Base_224-single_device-inference]

## Result
SILICON_PASS — removed load_dataset call; VisionPreprocessor handles image=None via default COCO URL

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
load-dataset-spacy-namespace-collision

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
(reported as `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute`)

## Root cause
The `tt_forge_models/spacy/` directory is a namespace package that shadows the real `spacy`
package in `sys.modules` when `tt_forge_models/` is on `sys.path`. The `datasets` library's
`_dill.py` performs `if issubclass(obj_type, spacy.Language):` during fingerprint hashing of
`load_dataset` arguments. Since the shadowed `spacy` module (a bare namespace package) has no
`Language` attribute, this raises `AttributeError`.

The bug lives in the loader: `megadescriptor/feature_extraction/pytorch/loader.py` uses
`load_dataset("huggingface/cats-image", split="test")` to fetch a sample image, triggering the
crash. The `VisionPreprocessor` class already handles `image=None` gracefully by downloading
the default COCO image from `http://images.cocodataset.org/val2017/000000039769.jpg`.

## Fix
Removed the `from datasets import load_dataset` import and the `load_dataset` call block from
`load_inputs` in
`third_party/tt_forge_models/megadescriptor/feature_extraction/pytorch/loader.py`.

When `image=None`, `load_inputs` now passes `None` directly to `input_preprocess`, which
passes it to `VisionPreprocessor.preprocess`, which fetches the default COCO image via
`get_file(default_image_url)`.

Branch: `remediation/megadescriptor-feature_extraction-pytorch-Base_224-single_device-inference`
in `tt-forge-models`.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    129.80s (0:02:09)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/megadescriptor/feature_extraction/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d75355 |
| tt-mlir         | 553c0632b |
| tt-xla          | 94362e631 |
| tt-forge-models | d9d261807d |
