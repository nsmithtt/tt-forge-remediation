# Remediation Summary: flux1_dev_dedistilled_mix_tuned_v4/pytorch-Default-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[flux1_dev_dedistilled_mix_tuned_v4/pytorch-Default-single_device-inference]

## Result
SILICON_PASS — fixed loader to use FluxTransformer2DModel.from_single_file() instead of FluxPipeline.from_pretrained() because the HF repo only ships a single safetensors file

## Stack layer
loader

## Tier
A

## Bug fingerprint
flux-single-file-repo-pipeline-pretrained-mismatch

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
huggingface_hub.errors.RemoteEntryNotFoundError: 404 Client Error. (Request ID: Root=1-69f07290-2154b33878e4c33b3924a6c6;e7ffcea7-9580-42db-858c-08e99e0868cd)

Entry Not Found for url: https://huggingface.co/wikeeyang/Flux1-Dev-DedistilledMixTuned-v4/resolve/main/model_index.json.
```

## Root cause
The `wikeeyang/Flux1-Dev-DedistilledMixTuned-v4` HuggingFace repository is not a Diffusers pipeline directory. It contains only a single `Flux1-Dev-DedistilledMixTuned-V4-fp8.safetensors` weight file (plus GGUF quantized variants). There is no `model_index.json`, no `transformer/config.json`, and none of the component subdirectories required by `FluxPipeline.from_pretrained()`. The original loader called `FluxPipeline.from_pretrained("wikeeyang/Flux1-Dev-DedistilledMixTuned-v4")`, which immediately 404s because model_index.json does not exist.

Note: the CI failure message says `pcc=nan` (the originally reported error), which would have followed from the loader crash in an older test harness version; on the current harness the crash surfaces directly as the 404 `RemoteEntryNotFoundError`.

## Fix
Rewrote `tt-xla/third_party/tt_forge_models/flux1_dev_dedistilled_mix_tuned_v4/pytorch/loader.py`:

- Removed `FluxPipeline` and `AutoencoderTiny` imports; replaced with `FluxTransformer2DModel`.
- Added an inline `_TRANSFORMER_CONFIG` dict matching the FLUX.1-dev architecture (identical to the config used by `flux_fp8/pytorch/loader.py`), with `guidance_embeds=True` for the dev-based model.
- Added `_make_local_config_dir()` helper that writes the config to a temp directory (same pattern as `flux_fp8` loader).
- Added `_load_transformer()` that calls `FluxTransformer2DModel.from_single_file(_SAFETENSORS_URL, config=config_dir, subfolder="transformer", torch_dtype=dtype)`.
- Changed `load_model()` to operate on `self._transformer` instead of `self.pipe.transformer`.
- Changed `load_inputs()` to generate fully synthetic text embeddings (random tensors with correct shapes) instead of running the CLIP/T5 text encoders, removing the pipeline dependency entirely. Dimensions are read from the loaded transformer config.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    ~9m 41s (580.87s)
- Tier A attempts: 1

## Files changed
- `tt-xla/third_party/tt_forge_models/flux1_dev_dedistilled_mix_tuned_v4/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348 |
