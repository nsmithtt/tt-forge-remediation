# Remediation Summary: depth_anything_v2_metric_indoor/pytorch-Small-single_device-inference

## Skill version
2

## Test
tests/runner/test_models.py::test_all_models_torch[depth_anything_v2_metric_indoor/pytorch-Small-single_device-inference]

## Result
SILICON_PASS

## Failure
The image processor of type `DPTImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.

## Root cause
Two loader-layer bugs:

1. **DPTImageProcessor (transformers 5.x breaking change)** — `AutoImageProcessor.from_pretrained()` in `depth_anything_v2_metric_indoor/pytorch/loader.py` now returns `DPTImageProcessorFast` by default, causing output differences.

2. **spacy namespace collision (masked the above locally)** — `dynamic_loader.setup_models_path()` adds `third_party/tt_forge_models` to `sys.path`. The directory `third_party/tt_forge_models/spacy/` is then importable as a Python namespace package named `spacy`, shadowing the real spacy package. `huspacy/pytorch/loader.py` had `import spacy` at module level; during test discovery this import succeeds (finds the namespace package), polluting `sys.modules["spacy"]`. Later, `datasets._dill.py` checks `if "spacy" in sys.modules` and then calls `spacy.Language`, which does not exist on the namespace package, raising `AttributeError`.

## Fix
1. In `depth_anything_v2_metric_indoor/pytorch/loader.py`: added `use_fast=False` to `AutoImageProcessor.from_pretrained()` to retain the slow `DPTImageProcessor`.

2. In `huspacy/pytorch/loader.py`: moved `import spacy` from module-level to inside `_load_nlp()` so it is only evaluated when the model is actually used, preventing the namespace collision from poisoning `sys.modules` during test collection.

Neither change is a forbidden workaround — both are legitimate loader-layer fixes for a transformers 5.x breaking change and a Python path namespace collision.

## Verification
pytest exit status: PASSED
Wall-clock duration: 137.55s (2:17)
Hardware: n150

## Files changed
- `third_party/tt_forge_models/depth_anything_v2_metric_indoor/pytorch/loader.py` — add `use_fast=False`
- `third_party/tt_forge_models/huspacy/pytorch/loader.py` — lazy `import spacy`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | eb01d156e76e8d3ce2ce30e81d4e14f7a68e9adb |
| tt-forge-models | dae039d44a318747bab5b61376572dc3da481fef |
