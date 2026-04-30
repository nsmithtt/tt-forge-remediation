# Remediation Summary: flux_dev_gguf-pytorch-Dev_Q4_K_S-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[flux_dev_gguf/pytorch-Dev_Q4_K_S-single_device-inference]

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
Three loader bugs in `flux_dev_gguf/pytorch/loader.py`:

1. **URL format**: `resolve/main` instead of `blob/main` in the HuggingFace URL passed to `FluxTransformer2DModel.from_single_file`. diffusers 0.37.1 strips `blob/main/` from the filename during URL parsing; using `resolve/main` doubles the path segment, producing an invalid download URL.

2. **Gated config repo**: diffusers 0.37.1 infers the model config from GGUF metadata, which points to the gated `black-forest-labs/FLUX.1-dev` repo. Without an explicit `config=` argument pointing to a non-gated repo, loading fails for machines without access to that gated repo.

3. **GGUFParameter TorchDynamo recursion** (root cause of the reported error): After `from_single_file` with `GGUFQuantizationConfig`, model parameters are `GGUFParameter` instances (a diffusers tensor subclass). `GGUFParameter.__torch_function__` calls `super().__torch_function__()`, which under TorchDynamo tracing recurses infinitely through the dispatch chain → `RecursionError: maximum recursion depth exceeded`. Fix: call `_dequantize_gguf_and_restore_linear(transformer)` in eager mode immediately after loading to convert all `GGUFParameter` instances to plain `nn.Linear` layers with regular tensors.

4. **Post-dequantization dtype mismatch**: `_dequantize_gguf_and_restore_linear` preserves the original storage dtype for F16-stored GGUF tensors (float16) rather than converting to `compute_dtype` (bfloat16), causing `RuntimeError: self and mat2 must have the same dtype, but got BFloat16 and Half` during the forward pass. Fix: call `torch.nn.Module.to(transformer, compute_dtype)` after dequantization. diffusers' `ModelMixin.to()` must be bypassed via the unbound `nn.Module.to()` call because `ModelMixin.to()` raises `ValueError: Casting a quantized model to a new dtype is unsupported` when `_hf_quantizer` is still set (even after dequantization).

## Fix
All changes in `tt-xla/third_party/tt_forge_models` on branch `remediation/flux_dev_gguf-pytorch-Dev_Q4_K_S-single_device-inference`:

- `flux_dev_gguf/pytorch/loader.py`:
  - Change URL from `resolve/main` to `blob/main`
  - Add `config=_FLUX_DEV_CONFIG_REPO` (`BBuf/flux1-dev-modelopt-nvfp4-sglang-transformer`) to bypass gated config
  - Import and call `_dequantize_gguf_and_restore_linear(self.transformer)` after `from_single_file`
  - Call `torch.nn.Module.to(self.transformer, compute_dtype)` to normalize dtypes

All four fixes combined into commit `330ac232f3` in tt_forge_models.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    710.73s (0:11:50)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/flux_dev_gguf/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a67933b274a28309120fad3d7d5b18d9d6666768 |
| tt-forge-models | 330ac232f3bb411429f9bc1dd27f003c47940b6e |
