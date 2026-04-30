# Remediation Summary: edgenext-pytorch-x-small-in1k-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[edgenext/pytorch-X_Small_IN1K-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
load-dataset-spacy-namespace-package-shadowing

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
AttributeError: module 'spacy' has no attribute 'Language'

The error arose during `load_dataset("huggingface/cats-image", split="test")` inside `edgenext/pytorch/loader.py:load_inputs`. The `datasets` dill hasher (`_dill.py:42`) checks `if issubclass(obj_type, spacy.Language)`, which crashes because `sys.modules['spacy']` holds the `tt_forge_models/spacy/` namespace package (which has no `Language` attribute) instead of the real spacy library.

## Root cause
The `huspacy/pytorch/loader.py` had `import spacy` at module level (line 14). During pytest collection, the dynamic loader scans all loader files, triggering this top-level import before the real `spacy` package is on `sys.path`. The `tt_forge_models/spacy/` directory (a namespace package) is resolved first, putting a stub module into `sys.modules['spacy']`. When the EdgeNeXt loader later calls `load_dataset(...)`, the `datasets` dill hasher finds `sys.modules['spacy']` and tries to access `spacy.Language`, which does not exist on the namespace stub.

This is a loader-layer bug: a module-level import in an unrelated loader (`huspacy`) pollutes `sys.modules` and breaks a different loader (`edgenext`) that calls `load_dataset`.

## Fix
In `tt_forge_models/huspacy/pytorch/loader.py`, removed the module-level `import spacy` and moved it inside `_load_nlp()`, the only method that uses it. This ensures `spacy` is only imported when the HuSpaCy loader is actually instantiated, preventing namespace-package shadowing during collection.

File changed: `huspacy/pytorch/loader.py`
Branch: `remediation/edgenext-pytorch-x-small-in1k-inference` in `tt-forge-models`

## Verification
- pytest exit: PASS
- Hardware:    n300
- Duration:    97.88s (0:01:37)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/huspacy/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | ca9f10cc70fce7612b19c219783d18e849d5b5ae |
