# Remediation Summary: flux_1_krea_dev_gguf-pytorch-Q4_K_S-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[flux_1_krea_dev_gguf/pytorch-Q4_K_S-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-metadata-wrong-config-repo

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   OSError: black-forest-labs/FLUX.1-Depth-dev is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'

Note: the originally reported failure (`raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")`) was superseded by the above OSError once `gguf>=0.10.0` was present in the environment. The root cause is the same loader bug.

## Root cause
The `flux_1_krea_dev_gguf` loader (loader layer) called `FluxTransformer2DModel.from_single_file` without an explicit `config=` argument. diffusers' `from_single_file` reads `general.architecture` from the GGUF metadata and looks it up in its internal `fetch_diffusers_config` mapping. The FLUX.1-Krea-dev GGUF files produced by InvokeAI use the architecture tag `flux-depth`, which diffusers maps to `black-forest-labs/FLUX.1-Depth-dev`. That repo does not exist (or is gated), so `cls.load_config(...)` raises `OSError`. The fix writes a minimal local `config.json` and passes it via `config=` so diffusers never queries HuggingFace for the config.

Additionally, `GGUFParameter` tensor subclasses produced by `GGUFQuantizationConfig` are incompatible with TT silicon compilation (TorchDynamo recursion). These are eagerly dequantized via `_dequantize_gguf_and_restore_linear` after loading, and the pipeline dependency on the gated `black-forest-labs/FLUX.1-dev` repo is removed by using synthetic inputs.

## Fix
All changes are in `tt-forge-models` on branch `remediation/flux_1_krea_dev_gguf-pytorch-Q4_K_S-single_device-inference`.

**`flux_1_krea_dev_gguf/pytorch/loader.py`**:
- Removed `FluxPipeline`, `AutoencoderTiny` dependencies (gated repos, not needed for transformer-only test)
- Added hardcoded `_TRANSFORMER_CONFIG` dict with the correct FLUX.1-dev architecture parameters
- Added `_make_local_config_dir()` which writes `_TRANSFORMER_CONFIG` to a temp `transformer/config.json`
- Changed `from_single_file` call to pass `config=config_dir, subfolder="transformer"` to bypass GGUF metadata config inference
- Added `GGUFQuantizationConfig(compute_dtype=dtype)` and `_dequantize_gguf_and_restore_linear` to properly handle GGUF weights, then cast to bf16 via `torch.nn.Module.to()`
- Replaced pipeline-based `load_inputs` with synthetic random embeddings (avoids all gated repo dependencies)

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    649.13s (0:10:49)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models/flux_1_krea_dev_gguf/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 62a985ab62c16f3421f743f65edf2c93d608dcfd |
| tt-forge-models | cb6f63d5645dbcfb48511e912e0f5f6cb04ab815 |
