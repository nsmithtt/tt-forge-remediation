# Remediation Summary: ministral_3_8b-pytorch-unsloth-Ministral-3-8B-Instruct-2512-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[ministral_3_8b/pytorch-unsloth/Ministral-3-8B-Instruct-2512-single_device-inference]

## Result
XFAIL — Ministral-3-8B (7.95B params, ~15.9 GB BF16) + Pixtral vision encoder with standard-resolution image (~9240 visual patches) requires >45 GB DRAM activation tensors; p150b has 32 GB total

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
ministral3-load-shard-spec-wrong-attr-paths

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
AttributeError: 'Mistral3ForConditionalGeneration' object has no attribute 'language_model'
```
Raised at `third_party/tt_forge_models/ministral_3_8b/pytorch/loader.py:157: in load_shard_spec` when iterating `model.language_model.layers`.

After fixing the loader, the terminal failure is:
```
TT_FATAL: Out of Memory: Not enough space to allocate 48586817536 B DRAM buffer across 8 banks,
where each bank needs to store 6073352192 B, but bank size is 4273390016 B
(allocated: 2541411392 B, free: 1731978624 B, largest free block: 1700473728 B)
```

## Root cause

**Loader bug (fixed):** `load_shard_spec` accessed `model.language_model.layers` and `model.vision_tower.vision_model.encoder.layers`. In `Mistral3ForConditionalGeneration`, the inner model is wrapped in `model.model`; the correct paths are `model.model.language_model.layers` and `model.model.vision_tower.transformer.layers`. The vision encoder layer attribute structure was also wrong: Pixtral uses `.attention.{q,k,v,o}_proj` and `.feed_forward.{gate,up,down}_proj`, not `.self_attn.*` / `.mlp.fc1/fc2`.

**Hardware-class OOM (XFAIL):** After fixing the loader, the model compiles and begins executing. The Pixtral vision encoder processes the sample `candy.JPG` image (~1540×1540 pixels) into approximately 9240 visual patches. The forward pass — vision encoder attention maps for 9240 patches across 24 layers plus LLM processing — requires more than 45 GB of DRAM for intermediate activation tensors. The p150b has 32 GB total DRAM, of which approximately 20 GB is already occupied by model weights (15.9 GB BF16 LLM + 0.8 GB vision tower). The remaining ~12 GB is insufficient for the peak activation memory. This is a hardware-class capacity ceiling; there is no single-device hardware configuration in the fleet that can run inference for this model-image combination.

An additional loader fix (`_patch_get_image_features`) was committed to the remediation branch to handle a TT silicon integer arithmetic bug: `(image_sizes // downsample_ratio).prod(-1)` returns 2320 instead of the correct 2310 when computed on the TT device. This patch moves the scalar `split_sizes` computation to CPU. It does not affect the OOM (which occurs during compilation before `get_image_features` is reached), but the fix is preserved for the case where the OOM is resolved on more capable hardware.

## Fix

**Loader fixes** — `tt-forge-models`, branch `remediation/ministral_3_8b-pytorch-unsloth-Ministral-3-8B-Instruct-2512-single_device-inference`:
- `ministral_3_8b/pytorch/loader.py` — `load_shard_spec`: corrected `model.language_model` → `model.model.language_model`, `model.vision_tower.vision_model.encoder` → `model.model.vision_tower.transformer`, and updated attention/MLP attribute names for the Pixtral layer structure.
- `ministral_3_8b/pytorch/loader.py` — `_patch_get_image_features`: added instance method patch to compute `split_sizes` on CPU, avoiding TT silicon integer arithmetic inaccuracy.
- Added `import torch` inside the patched function (required for `torch.cat` and `torch.split`).

**Test config XFAIL** — `tt-xla`, branch `remediation/ministral_3_8b-pytorch-unsloth-Ministral-3-8B-Instruct-2512-single_device-inference`:
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` — marked `ministral_3_8b/pytorch-unsloth/Ministral-3-8B-Instruct-2512-single_device-inference` as `KNOWN_FAILURE_XFAIL` with the OOM reason string.

## Verification
- pytest exit: FAIL (OOM after loader fix)
- Hardware:    blackhole-p150b
- Duration:    243.16s (0:04:03)
- Tier A attempts: N/A

## Files changed
- `ministral_3_8b/pytorch/loader.py` (tt-forge-models, 3 commits)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla, 1 commit)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 889670020485930807b70fb7aa486fa0b892c96c |
| tt-forge-models | 13e7a25f115bb14f1d8e301b76b39270f712d1bc |
