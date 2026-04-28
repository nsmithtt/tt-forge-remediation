# Remediation Summary: depth_anything_v2-pytorch-Base-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[depth_anything_v2/pytorch-Base-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
transformers-5x-use-fast-default

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The image processor of type `DPTImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.

Additionally, a secondary failure was observed locally: `AttributeError: module 'spacy' has no attribute 'Language'`, caused by `huspacy/pytorch/loader.py` importing `spacy` at module level. Since `third_party/tt_forge_models` is added to `sys.path` by the dynamic loader, the `spacy/` namespace directory within `tt_forge_models` shadows the real spacy package, creating a stub module in `sys.modules` with no `Language` attribute. This causes `datasets._dill` to crash when serializing objects.

## Root cause
Two loader-layer bugs in `tt_forge_models`:

1. **DPTImageProcessor (transformers 5.x)**: `transformers` 5.x changed `AutoImageProcessor.from_pretrained()` to return a fast processor by default for `DPTImageProcessor`. The loader did not pass `use_fast=False`, resulting in a `FutureWarning` (or error, depending on transformers version) about the breaking change.

2. **spacy namespace collision**: `dynamic_loader.setup_models_path()` adds `third_party/tt_forge_models` to `sys.path`. The `spacy/` subdirectory in `tt_forge_models` is a Python namespace package (no `__init__.py`), which shadows the real `spacy` package. `huspacy/pytorch/loader.py` had `import spacy` at module level, so when the huspacy loader was discovered, this namespace stub was placed in `sys.modules['spacy']` — a module with no `Language` attribute. The `datasets._dill` serializer later checked `if "spacy" in sys.modules` and crashed on `spacy.Language`.

## Fix
Two commits in `tt_forge_models` on branch `remediation/depth_anything_v2-pytorch-Base-single_device-inference`:

1. `depth_anything_v2/pytorch/loader.py`: Added `use_fast=False` to `AutoImageProcessor.from_pretrained()` call in `_load_processor()`.

2. `huspacy/pytorch/loader.py`: Moved `import spacy` from module level into the `_load_nlp()` function body, so it is only executed when the huspacy model is actually loaded (not during test discovery).

Both fixes were cherry-picked from `remediation/depth_anything_v2-pytorch-Small-single_device-inference` which addressed the same bugs for the Small variant.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    135.80s (0:02:15)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/depth_anything_v2/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/huspacy/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 1dcd7de8b8768418957412f2c3d292acb9ca0a01 |
