# Remediation Summary: anima_fp8/pytorch-preview3-base-mxfp8-single_device-inference

## Test

`tests/runner/test_models.py::test_all_models_torch[anima_fp8/pytorch-preview3-base-mxfp8-single_device-inference]`

## Status: SILICON_PASS

## Background

This test targets the MXFP8-quantized variant of Anima (Bedovyy/Anima-FP8), a 2B parameter
text-to-image diffusion transformer derived from NVIDIA Cosmos-Predict2-2B-Text2Image.
The MXFP8 variant uses the same loader as the FP8 variant — both fixes applied to
`anima_fp8/pytorch/loader.py` therefore cover all three model variants
(`preview3-base-fp8`, `preview3-base-mxfp8`, `preview3-base-nvfp4mixed`).

## Problems and Fixes

Two root causes were identified and fixed in an earlier session (for `preview3-base-fp8`).
The same fixes resolve the MXFP8 variant.

### 1. `from_single_file` model type detection failure

`CosmosTransformer3DModel.from_single_file()` uses key detection heuristics to
determine the Cosmos model type. All FP8/MXFP8 safetensors files are missing
`net.pos_embedder.dim_spatial_range` (stripped during quantization), so diffusers
falls back to the `"v1"` (Stable Diffusion 1.x) type, which tries to load a config
from `stable-diffusion-v1-5/stable-diffusion-v1-5` — causing an `OSError`.

**Fix:** Replace `from_single_file` with direct `CosmosTransformer3DModel`
instantiation using hardcoded Cosmos-Predict2-2B architecture parameters
(`num_attention_heads=16, num_layers=28, text_embed_dim=1024, adaln_lora_dim=256`),
then apply the standard `convert_cosmos_transformer_checkpoint_to_diffusers` key
mapping and load with `strict=False`. Also added a required `padding_mask` input
(shape `[B, 1, H, W]`) because `concat_padding_mask=True` in the model config.

### 2. TT hardware SDPA chunk size too small

The original `load_inputs` used `latent_height=2, latent_width=2`. With
`patch_size=(1, 2, 2)`, this produces only 1 spatial patch — a sequence length of
1 for self-attention. The TT hardware SDPA decode kernel requires `k_chunk_size` to
be a multiple of 32, but with seq_len=1 it calculated `k_chunk_size=2`, causing:
```
TT_FATAL: Chunk size must be multiple of 32, but the maximum calculated k_chunk_size is: 2
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

**Fix:** Increased `latent_height=16, latent_width=16` (giving 8×8=64 patches ≥ 32)
and `encoder_hidden_states` sequence from 8 to 32 tokens (for cross-attention).

## Changes Made

All changes are in `tt-xla/third_party/tt_forge_models` on branch:
`remediation/anima-fp8-fix-from-single-file-detection`

### `anima_fp8/pytorch/loader.py`
- Replaced `CosmosTransformer3DModel.from_single_file()` with direct construction
  using known Cosmos-Predict2-2B config parameters
- Added manual checkpoint conversion via
  `convert_cosmos_transformer_checkpoint_to_diffusers`
- Added `safetensors.torch.load_file` import for checkpoint loading
- Added `padding_mask` tensor to `load_inputs` output (required by `concat_padding_mask=True`)
- Increased `latent_height` and `latent_width` from 2 to 16 (64 self-attn patches)
- Increased encoder hidden states sequence from 8 to 32 (cross-attn constraint)

## Submodule Hashes

| Submodule | Hash |
|-----------|------|
| tt-metal | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla | 79046e376fdb77856cb6426577361057c1489f23 |
| tt-xla/third_party/tt_forge_models | 97dbda0a90388aa213cab5df4c15a75377651c2b |

## Commits in tt_forge_models

```
97dbda0a90 Fix anima_fp8 load_inputs: increase spatial dims to 16x16 and encoder seq to 32 for TT hardware SDPA minimum chunk size
b6f9cd05f0 Fix anima_fp8: bypass from_single_file auto-detection by loading CosmosTransformer3DModel directly with hardcoded config and add padding_mask to inputs
```
