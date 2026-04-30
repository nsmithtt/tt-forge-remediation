# Remediation Summary: glm_ocr-image_to_text-pytorch-mlx_8bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[glm_ocr/image_to_text/pytorch-mlx_8bit-single_device-inference]

## Result
SILICON_PASS — four loader bugs fixed; test passes on TT silicon in 237.27s

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
mlx-affine-8bit-checkpoint-incompatibility

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute

(The DeprecationWarning is a harmless SWIG warning emitted after test completion; the actual test failure was the loader crashing before the model ran.)

## Root cause
The `mlx-community/GLM-OCR-8bit` checkpoint was quantized with Apple MLX and has four
incompatibilities with the standard transformers 5.x `GlmOcrForConditionalGeneration` loader:

1. **quantization_config without quant_method**: The config.json contains
   `quantization_config: {group_size: 64, bits: 8, mode: "affine"}` but no `quant_method`.
   Transformers 5.x raises `ValueError` when it encounters this.

2. **uint32-packed int8 weights**: Linear layer weights are stored as `torch.uint32`-packed
   int8 values with per-group bf16 scales and biases.  Transformers expects standard float
   weights; the model must be dequantized manually before loading.

3. **Stale key names**: The checkpoint was quantized from an older model revision that used
   a different attribute layout: `language_model.lm_head.*`, `language_model.model.*`, and
   `vision_tower.*` instead of the current `lm_head.*`, `model.language_model.*`, and
   `model.visual.*`.

4. **MLX channel-last conv weights**: MLX stores Conv2d weights as `[out, H, W, in]` and
   Conv3d weights as `[out, D, H, W, in]`.  PyTorch expects channel-first (`[out, in, ...]`).
   Two conv layers failed `load_state_dict` with shape mismatches:
   `model.visual.patch_embed.proj.weight` (Conv3d, 5-D) and
   `model.visual.downsample.weight` (Conv2d, 4-D).

## Fix
All four bugs fixed in `tt_forge_models/glm_ocr/image_to_text/pytorch/loader.py` on branch
`remediation/glm_ocr-image_to_text-pytorch-mlx_8bit-single_device-inference` in the
tt-forge-models repo.

- **Bug 1**: Load `AutoConfig.from_pretrained`, delete `config.quantization_config` if
  present, then call `AutoModelForImageTextToText.from_config(config)` instead of
  `from_pretrained`.
- **Bug 2**: Added `_dequantize_mlx_affine_8bit()` — unpacks uint32 → uint8, applies
  per-group scale/bias, casts to bfloat16.
- **Bug 3**: Added `_remap_mlx_keys()` — remaps the three stale key prefixes to their
  current transformers 5.x equivalents.
- **Bug 4**: Added `_permute_mlx_conv_weight()` — permutes 4-D weights with
  `permute(0, 3, 1, 2)` and 5-D weights with `permute(0, 4, 1, 2, 3)` in the
  non-quantized path of `_dequantize_mlx_affine_8bit`.

Files changed:
- `tt-xla/third_party/tt_forge_models/glm_ocr/image_to_text/pytorch/loader.py`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    237.27s (0:03:57)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/glm_ocr/image_to_text/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 16d4b2a5bea75e89237f1ef9d531f065f2ce962f |
