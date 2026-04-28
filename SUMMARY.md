# Remediation Summary: megadescriptor-pytorch-L_384-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[megadescriptor/pytorch-L_384-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
spacy-namespace-collision-load-dataset

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.5876576076287366. Required: pcc=0.95.

Locally reproduced as:
E   AttributeError: module 'spacy' has no attribute 'Language'

## Root cause
The `load_inputs` method called `load_dataset("huggingface/cats-image", split="test")`.
`dynamic_loader.py` adds `tt_forge_models/` to `sys.path`, and `tt_forge_models/spacy/`
exists as a directory, making it a namespace package that shadows the real `spacy`
library. When `datasets._dill` later tries `issubclass(obj_type, spacy.Language)`, it
fails with `AttributeError: module 'spacy' has no attribute 'Language'`. In some CI
test-collection orders, the collision may corrupt inputs rather than raising immediately,
producing the observed PCC=0.5876.

The `VisionPreprocessor` already handles default image loading (COCO cat image via URL)
when `image=None`, so the `load_dataset` call was unnecessary.

## Fix
Removed the `load_dataset` import and the `load_dataset("huggingface/cats-image")`
call in `load_inputs`. The `VisionPreprocessor.preprocess(image=None, ...)` path now
handles image loading via its built-in COCO URL default.

- `tt-xla/third_party/tt_forge_models/megadescriptor/pytorch/loader.py`: removed
  `from datasets import load_dataset` and the `if image is None: dataset = ...` block.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    145.91s (0:02:25)
- Tier A attempts: N/A

## Files changed
- tt-xla/third_party/tt_forge_models/megadescriptor/pytorch/loader.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 87303ceb54fa08e62c9d62975a37d021b697406a |
| tt-forge-models | 370a0704f0ff1f671406e55d33a10b0116b7307a |
