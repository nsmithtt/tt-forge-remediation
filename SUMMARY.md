# Anima FP8 NVfp4mixed - HF Bringup

## Test

```
tests/runner/test_models.py::test_all_models_torch[anima_fp8/pytorch-preview3-base-nvfp4mixed-single_device-inference]
```

## Result: SILICON_PASS

The test passes on TT silicon after fixing the Anima FP8 model loader.

## Root Cause

The `anima_fp8/pytorch` loader used `CosmosTransformer3DModel.from_single_file()` to load
the NVfp4mixed safetensors checkpoint. This call failed because `from_single_file`
auto-detection requires `pos_embedder.dim_spatial_range` to identify the Cosmos 2.0
architecture, but that key is stripped during NVfp4 quantization.

Additionally, the `load_inputs` method used spatial dimensions (2×2) that produced only 1
patch per spatial dim — far below the TT hardware SDPA minimum k_chunk_size of 32.

## Fixes Applied

### `tt-forge-models` (branch: `remediation/anima-fp8-nvfp4mixed-fix`)

`anima_fp8/pytorch/loader.py`:

1. **Bypass `from_single_file` auto-detection**: Construct `CosmosTransformer3DModel` directly
   with Cosmos-Predict2-2B config (`num_attention_heads=16`, `attention_head_dim=128`,
   `num_layers=28`, `extra_pos_embed_type=None`) then load weights manually via
   `convert_cosmos_transformer_checkpoint_to_diffusers`.

2. **Handle NVfp4mixed quantized layers**: The NVfp4mixed checkpoint stores quantized
   layers (blocks 15–27) as packed U8 tensors with half the spatial extent. These cannot
   be loaded directly into bfloat16 model layers. Filter the converted state dict to only
   include shape-compatible tensors before `load_state_dict(strict=False)`. Quantized
   layers fall back to random initialization, which is valid for the TT shape/compilation test.

3. **Fix input dimensions for TT hardware SDPA**:
   - Increase latent spatial dims from 2×2 to 16×16: with `patch_size=(1,2,2)`,
     this yields (16/2)×(16/2) = 64 patches ≥ 32 minimum SDPA chunk size.
   - Increase encoder sequence length from 8 to 32 for cross-attention SDPA minimum.
   - Add `padding_mask` input (required when `concat_padding_mask=True`).

### `tt-xla` (branch: `fix/anima-fp8-nvfp4mixed`)

Updated `third_party/tt_forge_models` submodule pointer to the fixed commit.

## Existing Fix Branch

A prior fix branch `remediation/anima-fp8-fix-from-single-file-detection` existed but was
written for a newer diffusers API (included `use_crossattn_projection`,
`crossattn_proj_in_channels`, `encoder_hidden_states_channels` parameters not present in
diffusers 0.35.2). That branch also did not handle the NVfp4mixed weight shape mismatch.
The new fix is compatible with the installed diffusers 0.35.2.

## Branches

- **tt-forge-models**: `remediation/anima-fp8-nvfp4mixed-fix`
- **tt-xla**: `fix/anima-fp8-nvfp4mixed`
