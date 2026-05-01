# Remediation Summary: internlm_xcomposer2d5_clip-pytorch-ViT_Large_Patch14_560-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[internlm_xcomposer2d5_clip/pytorch-ViT_Large_Patch14_560-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
load-dataset-spacy-namespace-pollution

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
(Reported as `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute` in the summary, which is the last line emitted after the actual exception.)

## Root cause
The loader's `load_inputs` called `load_dataset("huggingface/cats-image")` from the HuggingFace `datasets` library. Internally, `datasets` uses `dill` for serialization, and `dill` checks `issubclass(obj_type, spacy.Language)` during pickling. The `tt_forge_models/spacy/` namespace package installed alongside `tt_forge_models` pollutes `sys.modules['spacy']` with a stub module that has no `Language` attribute, causing `AttributeError: module 'spacy' has no attribute 'Language'`. This is a loader bug — the model itself (CLIPVisionModel) does not need `datasets`; any PIL image suffices for testing.

## Fix
Replaced `load_dataset("huggingface/cats-image")` with `PIL.Image.new("RGB", (560, 560), color=(128, 128, 128))` in `load_inputs`, and removed the `from datasets import load_dataset` import.

**File changed:** `internlm_xcomposer2d5_clip/pytorch/loader.py` in `tt-forge-models` repo.

## Verification
- pytest exit: PASS
- Hardware: wormhole
- Duration: 87.67s
- Tier A attempts: N/A

## Files changed
- `tt-forge-models`: `internlm_xcomposer2d5_clip/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 46753c71efe79ea446d600940084ef642593634e |
| tt-forge-models | 3b8924caf61975d7c788319861685fa6ceacd394 |
