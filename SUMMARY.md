# Remediation Summary: man_woman_face_detection-pytorch-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[man_woman_face_detection/pytorch-Base-single_device-inference]

## Result
SILICON_PASS — lazy spacy import in huspacy loader fixes namespace package collision causing load_dataset() to crash

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
load-dataset-spacy-namespace-package-collision

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
Surfaced as:
```
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```
with the actual traceback showing `datasets/utils/_dill.py:42` checking `issubclass(obj_type, spacy.Language)` during `load_dataset("huggingface/cats-image", ...)` in `man_woman_face_detection/pytorch/loader.py`.

## Root cause
`huspacy/pytorch/loader.py` had `import spacy` at module level (line 14). During pytest collection, Python imported this file, which resolved `spacy` to the `tt_forge_models/spacy/` namespace package (on `sys.path` from the dynamic loader's `third_party/tt_forge_models/` path) instead of the real spacy installation. This placed the stub namespace package object in `sys.modules['spacy']`. When `man_woman_face_detection/pytorch/loader.py` subsequently called `load_dataset("huggingface/cats-image")`, the `datasets` library's `dill` hasher checked `issubclass(obj_type, spacy.Language)` and crashed because the namespace package has no `Language` attribute.

## Fix
In `tt-forge-models` `huspacy/pytorch/loader.py`: removed module-level `import spacy` and moved `import spacy` inside the `_load_nlp()` method (the only call site). This ensures the real spacy package is only imported when `_load_nlp()` is actually called during model loading, not at collection time, so `sys.modules['spacy']` remains unset when `load_dataset` is called.

Remediation branch: `remediation/man_woman_face_detection-pytorch-single_device-inference` in `tenstorrent/tt-forge-models`.

## Verification
- pytest exit: PASS
- Hardware:    wormhole
- Duration:    47.28s
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/huspacy/pytorch/loader.py` — move `import spacy` from module level into `_load_nlp()`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | e79c438797b66c7cdb78bb6adaca75640984066b |
| tt-forge-models | b29354e100e67d7cb6cac582b80103b060c2551a |
