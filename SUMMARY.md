# Remediation Summary: flux_1_fill_dev_gguf-pytorch-Q6_K-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[flux_1_fill_dev_gguf/pytorch-Q6_K-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-parameter-torch-function-dynamo-recursion

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   torch._dynamo.exc.InternalTorchDynamoError: RecursionError: maximum recursion depth exceeded

The failure reproduced as a cascade of three loader bugs:
1. `OSError: YarvixPA/FLUX.1-Fill-dev-GGUF does not appear to have a file named resolve/main/flux1-fill-dev-Q6_K.gguf` (double resolve/main URL)
2. `GatedRepoError: 403 Cannot access gated repo black-forest-labs/FLUX.1-Fill-dev` (GGUF metadata pointing to gated repo)
3. After fixing the above: the original RecursionError would surface under TorchDynamo (GGUFParameter.__torch_function__ infinite recursion)

## Root cause
Three bugs in `flux_1_fill_dev_gguf/pytorch/loader.py` (loader layer):

1. **resolve/main URL doubling**: `from_single_file` was called with a `resolve/main`-style URL. diffusers 0.37.x strips `blob/main/` from filenames but not `resolve/main/`, causing the path component to be doubled into the download URL (`…/resolve/main/resolve/main/flux1-fill-dev-Q6_K.gguf`).

2. **Gated config repo**: The GGUF metadata (and the explicit `from_single_file` call) tried to load the FluxTransformer2DModel config from `black-forest-labs/FLUX.1-Fill-dev`, which is a gated HuggingFace repo returning 403 in this environment. No explicit `config=` was provided, so diffusers fell back to the GGUF metadata pointer, which also pointed to the gated repo.

3. **GGUFParameter TorchDynamo recursion**: `FluxTransformer2DModel` loaded via `GGUFQuantizationConfig` has `GGUFParameter` tensor subclasses. `GGUFParameter.__torch_function__` calls `super().__torch_function__()`, which recurses infinitely under TorchDynamo tracing. Fix: call `_dequantize_gguf_and_restore_linear()` eagerly after loading to convert all parameters to plain tensors, then cast to compute_dtype via `torch.nn.Module.to()` directly (diffusers `ModelMixin.to()` raises `ValueError` on quantized models even after dequantization because `_hf_quantizer` is still set).

A fourth sub-bug emerged when providing the local config: the FLUX.1-Fill-dev transformer has `in_channels=384` (concatenated noisy+masked+mask input) but `out_channels=64` (outputs only denoised latents). Setting `out_channels=null` (which diffusers defaults to `in_channels`) caused a weight shape mismatch for `proj_out.weight` (`[384, 3072]` expected vs `[64, 3072]` in the GGUF).

## Fix
All fixes are in `tt-forge-models` on branch `remediation/flux_1_fill_dev_gguf-pytorch-Q6_K-single_device-inference`.

**`flux_1_fill_dev_gguf/pytorch/loader.py`**:
- Changed `from_single_file` URL from `resolve/main` to `blob/main`
- Added `config=_TRANSFORMER_CONFIG_DIR` (local path) to `from_single_file`
- Added `from diffusers.quantizers.gguf.utils import _dequantize_gguf_and_restore_linear`
- After `from_single_file`: call `_dequantize_gguf_and_restore_linear(self.transformer)` then `torch.nn.Module.to(self.transformer, compute_dtype)`
- Removed the `FluxFillPipeline` load entirely (only the transformer is returned; `load_inputs` now uses synthetic random embeddings with correct shapes, avoiding the gated pipeline repo)

**`flux_1_fill_dev_gguf/pytorch/transformer_config/config.json`** (new file):
- Local FluxTransformer2DModel config with `in_channels=384`, `out_channels=64`, and standard FLUX architecture parameters — bypasses the gated `black-forest-labs/FLUX.1-Fill-dev` repo that GGUF metadata points to.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    599.22s (0:09:59)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models/flux_1_fill_dev_gguf/pytorch/loader.py`
- `tt-forge-models/flux_1_fill_dev_gguf/pytorch/transformer_config/config.json` (new)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550 |
| tt-mlir         | 553c0632b |
| tt-xla          | 7ac5581da |
| tt-forge-models | 0cc1bd759b |
