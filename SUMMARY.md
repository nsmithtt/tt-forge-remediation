# Remediation Summary: depth_anything-pytorch-Base-single_device-inference

## Skill version
2

## Test
tests/runner/test_models.py::test_all_models_torch[depth_anything/pytorch-Base-single_device-inference]

## Result
SILICON_PASS

## Failure
The image processor of type `DPTImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.

## Root cause
Two loader-layer bugs in `tt_forge_models`:

1. **DPTImageProcessor breaking change (transformers 5.x)**: `AutoImageProcessor.from_pretrained` for `LiheYoung/depth-anything-base-hf` now loads `DPTImageProcessor` as a fast processor by default. This is a transformers 5.x breaking change.

2. **spaCy namespace package pollution**: `tt_forge_models/spacy/` is a directory (model category) without `__init__.py`, making it a Python 3 namespace package named `spacy`. When `huspacy/pytorch/loader.py` does `import spacy` at module level during test collection, it finds this namespace package (which has no `Language` attribute) and adds it to `sys.modules`. Later, `datasets/utils/_dill.py` checks `if "spacy" in sys.modules: import spacy; issubclass(obj_type, spacy.Language)` and crashes with `AttributeError: module 'spacy' has no attribute 'Language'` when `depth_anything/load_inputs` calls `load_dataset`.

## Fix
**Fix 1** (`depth_anything/pytorch/loader.py`): Added `use_fast=False` to `AutoImageProcessor.from_pretrained()` to retain the slow `DPTImageProcessor` behaviour required by this model checkpoint.

**Fix 2** (`huspacy/pytorch/loader.py`): Added a module-level `importlib.util.find_spec("spacy")` guard. Namespace packages have `loader=None`; real installed packages have a non-None loader. If the namespace package is detected, an `ImportError` is raised before `import spacy` executes, preventing `sys.modules["spacy"]` from being populated with the stub namespace package. This causes `huspacy` to fail to import (correctly skipped as a missing-dependency model) and removes the `sys.modules` pollution that was crashing `datasets._dill.py`.

Neither fix is a forbidden workaround: both are genuine loader-layer fixes for a transformers API breaking change and a namespace package collision bug.

## Verification
pytest exit: PASSED
Wall-clock duration: 123.58s (0:02:03)
Hardware: Blackhole (bh-lb-13)

## Files changed
- `depth_anything/pytorch/loader.py` — added `use_fast=False`
- `huspacy/pytorch/loader.py` — added namespace package guard

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a9c0e94819ee86171e4db7f1fa33fc8816f266b4 |
| tt-forge-models | 997b6c291ccceea9d95af1bc54a924aca5a9f6f6 |
