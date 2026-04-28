# Remediation Summary: flux_1_krea_dev_gguf-pytorch-Q4_K_M-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[flux_1_krea_dev_gguf/pytorch-Q4_K_M-single_device-inference]

## Result
SILICON_PASS — loader fixed to use local config and dequantized GGUF weights

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-metadata-points-to-wrong-gated-repo

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
2026-04-24 08:36:10.259 | critical |          Always | TT_THROW: Fabric Router Sync: Timeout after 10000 ms. Device 2: Expected status 0xa2b2c2d2, got 0xa1b1c1d1 (assert.hpp:104)

The failure could not be reproduced as reported: before reaching silicon execution, the
loader itself was crashing with:

  OSError: black-forest-labs/FLUX.1-Depth-dev is not a local folder and is not a valid
  model identifier listed on 'https://huggingface.co/models'

Two cascading loader bugs were present. Once both were fixed, the test passed on silicon
without ever reproducing the original timeout.

## Root cause
Two bugs in the `flux_1_krea_dev_gguf/pytorch/loader.py` loader:

**Bug 1 — wrong config repo from GGUF metadata.**
`FluxTransformer2DModel.from_single_file` reads the GGUF file's `general.architecture`
metadata key (value: `"flux"`), then calls diffusers' `fetch_diffusers_config` which in
version 0.37.1 maps the `"flux"` architecture to `black-forest-labs/FLUX.1-Depth-dev`.
That repo is gated and inaccessible without an HF token, causing the OSError above.
The actual model weights, after dequantization, have standard FLUX.1-dev architecture
(`in_channels=64`, `pooled_projection_dim=768`, `guidance_embeds=True`).

**Bug 2 — raw byte shape mismatch with no GGUF quantizer.**
The GGUF file stores weights as BF16 (`GGMLQuantizationType.BF16 = type 30`). diffusers'
`load_gguf_checkpoint` wraps them in `GGUFParameter` with uint8 raw data (each BF16
element occupies 2 bytes), so the raw byte shape is 2× the logical shape (e.g. weight
with logical shape [3072, 256] has raw shape [3072, 512]). Without a `GGUFQuantizationConfig`
the model loading code compares model-expected shapes against raw byte shapes and raises
a `ValueError` shape mismatch. With the config, the GGUF quantizer's `check_quantized_param_shape`
compares against logical shapes instead.

**Secondary: pipeline loading for inputs used gated repos.**
The original loader loaded a full `FluxPipeline` from `black-forest-labs/FLUX.1-dev`
(also gated) to produce tokenized text embeddings for `load_inputs`. This was unnecessary
for inference testing; synthetic random embeddings in the correct shapes are sufficient.

## Fix
Changes to `flux_1_krea_dev_gguf/pytorch/loader.py` in `tt_forge_models`:

1. Added `_TRANSFORMER_CONFIG` dict with the standard FLUX.1-dev architecture (identical
   to the arcticlatent_flux1 loader). Added `_make_local_config_dir()` that writes
   `transformer/config.json` to a temp dir, and passed `config=config_dir,
   subfolder="transformer"` to `from_single_file` to bypass GGUF metadata → wrong repo.

2. Passed `quantization_config=GGUFQuantizationConfig(compute_dtype=dtype)` so the
   GGUF quantizer replaces `nn.Linear` with `GGUFLinear` and uses logical shapes for the
   shape compatibility check.

3. Called `_dequantize_gguf_and_restore_linear(self._transformer)` and reset
   `self._transformer.is_quantized = False` before casting to BF16, so the model
   returned to the test harness has standard `nn.Linear` layers with BF16 parameters
   (compatible with TT silicon compilation).

4. Replaced pipeline-based `load_inputs` (requiring gated repos) with synthetic random
   inputs of correct shapes, matching the arcticlatent_flux1 loader pattern.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    0:11:42
- Tier A attempts: N/A

## Files changed
- `flux_1_krea_dev_gguf/pytorch/loader.py` (tt_forge_models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 03b942fb36d37b4130c45874de3798ac334322d9 |
| tt-forge-models | bd4969616b70b2593a01f36b500b84cff6973965 |
