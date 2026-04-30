# Remediation Summary: flux_dev_gguf-pytorch-eviation_caesar_Q5_K_S-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[flux_dev_gguf/pytorch-eviation_caesar_Q5_K_S-single_device-inference]

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

## Root cause
Four loader bugs in `flux_dev_gguf/pytorch/loader.py`:

1. **URL format**: `resolve/main` in HuggingFace URLs was incompatible with diffusers 0.37.1's `_extract_repo_id_and_weights_name`, which only strips `blob/main`. The `resolve/main` prefix remained in the filename, causing a malformed double-path URL.

2. **Gated config repo**: diffusers 0.37.1 infers the model config source from GGUF metadata, resolving to the gated `black-forest-labs/FLUX.1-dev` repository. This causes an authentication error during model loading.

3. **TorchDynamo recursion** (the primary reported failure): `GGUFParameter.__torch_function__` calls `super().__torch_function__()` which under TorchDynamo tracing causes infinite recursion (`RecursionError: maximum recursion depth exceeded` wrapped as `InternalTorchDynamoError`). The model must be dequantized in eager mode before compilation so dynamo only sees plain `nn.Linear` layers.

4. **Dtype mismatch after dequantization**: F16-stored GGUF tensors dequantize to `float16`, not the requested `bfloat16` compute dtype. `diffusers.ModelMixin.to()` raises `ValueError` on quantized models even after dequantization (because `hf_quantizer` is still set), so `torch.nn.Module.to()` must be called directly.

## Fix
All four fixes are in `flux_dev_gguf/pytorch/loader.py` in `tt_forge_models` on branch `remediation/flux_dev_gguf-pytorch-eviation_caesar_Q5_K_S-single_device-inference` (commit `d6d72b9693`):

1. Change URL from `resolve/main` to `blob/main`.
2. Add `config=_FLUX_DEV_CONFIG_REPO` (`BBuf/flux1-dev-modelopt-nvfp4-sglang-transformer`) to `from_single_file()`.
3. Import and call `_dequantize_gguf_and_restore_linear(self.transformer)` after loading.
4. Call `torch.nn.Module.to(self.transformer, compute_dtype)` after dequantization.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    674.00s (0:11:14)
- Tier A attempts: N/A

## Files changed
- `flux_dev_gguf/pytorch/loader.py` (tt_forge_models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | aecc3c84820cbe13c6069b1b8500449e1125e0f2 |
| tt-forge-models | d6d72b9693e467b74726f572c4190033ffc988d9 |
