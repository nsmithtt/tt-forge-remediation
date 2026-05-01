# Remediation Summary: megadescriptor-pytorch-T_224-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[megadescriptor/pytorch-T_224-single_device-inference]

## Result
SILICON_PASS

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
third_party/tt_forge_models/megadescriptor/pytorch/loader.py:100: in load_inputs
    if issubclass(obj_type, spacy.Language):
E   AttributeError: module 'spacy' has no attribute 'Language'
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

## Root cause
The `megadescriptor/pytorch/loader.py` imported `load_dataset` from the `datasets`
library and called it in `load_inputs` to fetch a sample image. The `tt_forge_models`
package contains a `spacy/` subdirectory that is a namespace package for model loaders.
When `tt_forge_models` is on `sys.path`, this directory shadows the real `spacy`
PyPI package. The `datasets` library's internal `_dill` module imports `spacy.Language`
at runtime, which fails with `AttributeError: module 'spacy' has no attribute 'Language'`
because it hits the fake `tt_forge_models/spacy/` instead of the real spacy package.
The `VisionPreprocessor` used by the loader already handles default image loading
(via COCO URL cache), so `load_dataset` is entirely unnecessary.

This is the same root cause as the previously fixed `megadescriptor/pytorch-L_384` variant.

## Fix
Removed `from datasets import load_dataset` import and the `load_dataset` call in
`load_inputs` (the `if image is None:` block that fetched the cats-image dataset).

File changed: `megadescriptor/pytorch/loader.py` in `tt-forge-models`.

## Verification
- pytest exit: PASS
- Hardware:    wormhole
- Duration:    107.24s (0:01:47)
- Tier A attempts: N/A

## Files changed
- `megadescriptor/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | dae304b04ab52db00826ea23f2e59bd34ac248ab |
| tt-forge-models | 82089c6b21e2fcf35a3f639e55028f3a21f65b81 |
