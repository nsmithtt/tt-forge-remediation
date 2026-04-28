# Remediation Summary: colpali/vision_retrieval/pytorch-colpali-v12-random-testing-single_device-inference

## Skill version
3

## Test
tests/runner/test_models.py::test_all_models_torch[colpali/vision_retrieval/pytorch-colpali-v12-random-testing-single_device-inference]

## Result
SILICON_PASS

## Failure
The image processor of type `SiglipImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.

## Root cause
Loader layer (`tt_forge_models`). The `huspacy/pytorch/loader.py` had a top-level `import spacy` statement. The `dynamic_loader` inserts `models_root` (`third_party/tt_forge_models`) at the front of `sys.path` so all loaders can be discovered. Since `models_root/spacy/` exists as a directory without `__init__.py`, Python treats it as a namespace package. When the huspacy loader was imported during pytest collection, `import spacy` resolved to this namespace package and registered it in `sys.modules['spacy']` — a module with no `Language` attribute.

Later, when the colpali `load_inputs()` called `load_dataset("huggingface/cats-image")`, the datasets library's `_dill.py` checked `if "spacy" in sys.modules` (True), then tried `issubclass(obj_type, spacy.Language)` — crashing with `AttributeError: module 'spacy' has no attribute 'Language'`.

The reported `SiglipImageProcessor` warning does not raise an exception in the current version of transformers; it is a harmless `FutureWarning` that predates the actual failure root cause.

## Fix
Moved `import spacy` from module-level to inside the `_load_nlp()` method in `huspacy/pytorch/loader.py`. This ensures the import only runs when the huspacy model is actually loaded (not during collection), preventing the namespace package from poisoning `sys.modules['spacy']`.

This is a legitimate loader-layer fix (lazy import to avoid import-time side-effects). No forbidden workarounds were used.

Changed file: `huspacy/pytorch/loader.py` in `tt_forge_models` repo, branch `remediation/colpali-vision-retrieval-siglip-fix`.

## Verification
pytest exit status: PASSED  
Wall-clock duration: 69.53s  
Hardware: blackhole (p150)

## Files changed
- `huspacy/pytorch/loader.py` — moved `import spacy` inside `_load_nlp()`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | b73a206ae0354fcba18718cfbc61a676b9096146 |
| tt-forge-models | fa48f1a14024cad41e2e1730ad4f0a5158fd4912 |
