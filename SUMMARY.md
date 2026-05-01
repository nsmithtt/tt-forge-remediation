# Remediation Summary: mistral-pixtral-pytorch-12B_2409-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mistral/pixtral/pytorch-12B_2409-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
loader-wrong-model-id-native-mistral-format

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Two loader bugs in `mistral/pixtral/pytorch/loader.py`:

**Bug 1 (immediate):**
```
AttributeError: type object 'ModelVariant' has no attribute 'PIXTRAL_12B_2409_BNB_4BIT'
```

**Bug 2 (after fixing Bug 1):**
```
OSError: mistralai/Pixtral-12B-2409 does not appear to have a file named
pytorch_model.bin or model.safetensors.
```

## Root cause
Commit `c401c320ed` renamed the `PIXTRAL_12B_2409_BNB_4BIT` enum member to
`PIXTRAL_12B_2409` and changed the model from `unsloth/Pixtral-12B-2409-bnb-4bit`
to `mistralai/Pixtral-12B-2409`, but introduced two bugs:

1. The `load_model` function still referenced the old `PIXTRAL_12B_2409_BNB_4BIT`
   name in a dead `device_map="cpu"` guard, causing `AttributeError` immediately.

2. `mistralai/Pixtral-12B-2409` stores weights in native Mistral format
   (`consolidated.safetensors` + `params.json`, no `config.json`) and cannot be
   loaded with `LlavaForConditionalGeneration.from_pretrained`. The HuggingFace-
   compatible conversion of those same weights is `mistral-community/pixtral-12b`.

## Fix
Both fixes are in `tt_forge_models/mistral/pixtral/pytorch/loader.py`:

**Fix 1** — removed the stale `device_map` guard block (4 lines):
```python
# Quantized variants need device_map="cpu" for CPU-based loading
if self._variant in (ModelVariant.PIXTRAL_12B_2409_BNB_4BIT,):
    model_kwargs["device_map"] = "cpu"
```
Commit: `095b544434` on branch
`remediation/mistral-pixtral-pytorch-12B_2409-single_device-inference`
in `tenstorrent/tt-forge-models`.

**Fix 2** — corrected the model ID:
```python
# Before
pretrained_model_name="mistralai/Pixtral-12B-2409",
# After
pretrained_model_name="mistral-community/pixtral-12b",
```
`mistral-community/pixtral-12b` is the standard HuggingFace-format conversion
of the same September 2024 checkpoint; it shares architecture
(`LlavaForConditionalGeneration`) with the existing community variant.

Commit: `893fc6841d` on the same branch.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    165.99s (0:02:45)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/mistral/pixtral/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a3619a515 (bumps tt-forge-models to 893fc6841d) |
| tt-forge-models | 893fc6841d5468b1d36ac76630fa3db63d987d1c |
