# Remediation Summary: community_forensics_deepfake_det_vit-pytorch-Base-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[community_forensics_deepfake_det_vit/pytorch-Base-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
transformers-5x-vit-image-processor-center-crop-removed

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute

The reported failure message masked three cascading loader errors:
1. `ModuleNotFoundError: No module named 'infra'` — tests/conftest.py imports from infra (tests/infra/) but tests/ was not in sys.path.
2. `AttributeError: module 'spacy' has no attribute 'Language'` — dynamic_loader added models_root to sys.path, causing the spacy/ stub directory in tt_forge_models to shadow the real spaCy library.
3. `ValueError: Input image size (440*440) doesn't match model (384*384)` — transformers 5.x removed do_center_crop from ViTImageProcessor.preprocess(); the processor resized to 440×440 (the preprocessor_config.json default) but never cropped to the model's expected 384×384.

## Root cause
Three independent loader-layer bugs on the `ip-172-31-30-232-tt-xla-dev/ubuntu/hf-bringup-9` branch:

1. **Missing `pythonpath = tests` in pytest.ini**: `tests/conftest.py:25` does `from infra import DeviceConnectorFactory, Framework`. The `infra` package lives at `tests/infra/`. Without `pythonpath = tests` in pytest.ini, pytest does not add `tests/` to sys.path (because `tests/__init__.py` makes it a package, not a rootdir-rooted directory), so the import fails at conftest load time.

2. **models_root added to sys.path in dynamic_loader**: `DynamicLoader.setup_models_path()` (tests/runner/utils/dynamic_loader.py:207) inserted `models_root` into `sys.path`. The `tt_forge_models/spacy/` directory (a spaCy language-model data dir) was then importable as the top-level `spacy` module, shadowing the real library. `datasets._dill` checks `issubclass(obj_type, spacy.Language)` during fingerprinting, which raised `AttributeError: module 'spacy' has no attribute 'Language'`.

3. **ViTImageProcessor center-crop removed in transformers 5.x**: The model's `preprocessor_config.json` specifies `size: 440, crop_size: 384`. The loader called `ViTImageProcessor.from_pretrained(..., do_center_crop=True)` intending to enable center-crop to 384×384, but in transformers 5.x `preprocess()` no longer accepts `do_center_crop` as a parameter (decorated with `@filter_out_non_signature_kwargs`). The image was resized to 440×440 but not cropped, so the model forward pass raised `ValueError: Input image size (440*440) doesn't match model (384*384)`.

## Fix
Three commits across two repos:

**tt-xla** (`remediation/community_forensics_deepfake_det_vit-pytorch-Base-single_device-inference`):

1. `pytest.ini`: Added `pythonpath = tests` so pytest includes `tests/` in sys.path, making `from infra import ...` resolvable.
   - File: `pytest.ini`

2. `tests/runner/utils/dynamic_loader.py`: Removed `sys.path.insert(0, models_root)` from `setup_models_path()`. Relative imports in loaders work via `__package__` + the `tt_forge_models` namespace package already registered in `sys.modules`; the sys.path insertion is not needed and causes spacy namespace pollution.
   - File: `tests/runner/utils/dynamic_loader.py`

**tt-forge-models** (`remediation/community_forensics_deepfake_det_vit-pytorch-Base-single_device-inference`):

3. `community_forensics_deepfake_det_vit/pytorch/loader.py`: Changed `_load_processor()` to load with `size={"height": 384, "width": 384}, do_center_crop=False` instead of `do_center_crop=True`. This makes the processor resize directly to the model's expected 384×384 input size, bypassing the no-longer-functional center-crop path in transformers 5.x.
   - File: `community_forensics_deepfake_det_vit/pytorch/loader.py`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    28.53s
- Tier A attempts: N/A

## Files changed
- tt-xla: `pytest.ini`
- tt-xla: `tests/runner/utils/dynamic_loader.py`
- tt-forge-models: `community_forensics_deepfake_det_vit/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d75355 |
| tt-mlir         | 553c0632b |
| tt-xla          | d3f6041ec |
| tt-forge-models | 6ac073148d |
