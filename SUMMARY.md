# Remediation Summary: donut/document_image_classification/pytorch-rvlcdip-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[donut/document_image_classification/pytorch-rvlcdip-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
huspacy-spacy-namespace-pollution

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The image processor of type `DonutImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.

## Root cause
Two loader-layer bugs combined to cause the failure:

**Bug 1 (actual crash):** `huspacy/pytorch/loader.py` had a top-level `import spacy`. The `dynamic_loader` inserts `models_root` (`third_party/tt_forge_models`) at the front of `sys.path` during test collection. Since `models_root/spacy/` exists as a directory without `__init__.py`, Python treats it as a namespace package. When the huspacy loader was imported during pytest collection, `import spacy` resolved to this namespace package and registered it in `sys.modules['spacy']` — a module with no `Language` attribute. Later, when the Donut `load_inputs()` called `load_dataset("hf-internal-testing/example-documents", ...)`, the datasets library's `_dill.py` checked `if "spacy" in sys.modules` (True), then tried `issubclass(obj_type, spacy.Language)` — crashing with `AttributeError: module 'spacy' has no attribute 'Language'`.

**Bug 2 (reported, potential PCC impact):** `DonutProcessor.from_pretrained()` in transformers 5.x now loads `DonutImageProcessor` as a fast image processor by default, even when the checkpoint was saved with the slow processor. This produces subtly different pixel values and could cause PCC failures. The fix is to pass `use_fast=False`.

## Fix
**Fix 1:** Moved `import spacy` from module-level inside `_load_nlp()` in `huspacy/pytorch/loader.py`. This ensures the import only runs when the huspacy model is actually loaded (not during collection), preventing the namespace package from poisoning `sys.modules['spacy']`.

**Fix 2:** Added `use_fast=False` to `DonutProcessor.from_pretrained()` in `donut/document_image_classification/pytorch/loader.py` to use the slow image processor as the checkpoint expects.

Both changes are in the `tt_forge_models` repo on branch `remediation/donut-document_image_classification-pytorch-rvlcdip-single_device-inference`.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    272.98s (0:04:32)
- Tier A attempts: N/A

## Files changed
- `huspacy/pytorch/loader.py` — moved `import spacy` inside `_load_nlp()`
- `donut/document_image_classification/pytorch/loader.py` — added `use_fast=False` to `DonutProcessor.from_pretrained()`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 7cd0554d988fcd4594427e0829dde1f2a0e08a28 |
