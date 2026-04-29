# Remediation Summary: flux_dev_gguf-pytorch-eviation_caesar_Q5_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[flux_dev_gguf/pytorch-eviation_caesar_Q5_K_M-single_device-inference]

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

1. **URL format**: `resolve/main` instead of `blob/main` in the HuggingFace URL passed to `FluxTransformer2DModel.from_single_file`. diffusers 0.37.1 strips `blob/main/` from the filename but doubles `resolve/main/`, producing an invalid path.

2. **Gated config repo**: diffusers 0.37.1 infers the config from GGUF metadata, which points to the gated `black-forest-labs/FLUX.1-dev` repo. Without an explicit `config=` argument, loading fails on machines without access to that repo.

3. **GGUFParameter TorchDynamo recursion** (root cause of the reported error): After loading, model parameters are `GGUFParameter` instances (diffusers tensor subclass). `GGUFParameter.__torch_function__` calls `super().__torch_function__()` which, under TorchDynamo tracing, recurses infinitely through the dispatch chain → `RecursionError`. Fix: call `_dequantize_gguf_and_restore_linear(transformer)` in eager mode before compilation to convert all `GGUFParameter` instances to plain `nn.Linear` layers with regular tensors.

4. **Post-dequantization dtype mismatch**: `_dequantize_gguf_and_restore_linear` preserves the original storage dtype for F16-stored GGUF tensors (float16) rather than converting to `compute_dtype` (bfloat16). This causes `RuntimeError: self and mat2 must have the same dtype, but got BFloat16 and Half` during the CPU forward pass. Fix: call `torch.nn.Module.to(transformer, compute_dtype)` after dequantization. diffusers' `ModelMixin.to()` must be bypassed via the unbound `nn.Module.to()` call because it rejects dtype changes when `_hf_quantizer` is still set (even post-dequantization).

## Fix
All changes in `tt-xla/third_party/tt_forge_models` on branch `remediation/flux_dev_gguf-pytorch-eviation_caesar_Q5_K_M-single_device-inference`:

- `flux_dev_gguf/pytorch/loader.py`: Change URL to `blob/main`, add `config=_FLUX_DEV_CONFIG_REPO`, call `_dequantize_gguf_and_restore_linear` + `torch.nn.Module.to(transformer, compute_dtype)` after loading.

Commits:
- `bf6f572d44`: fix blob/main URL
- `423027be09`: add explicit config repo
- `2f834f7a2d`: call _dequantize_gguf_and_restore_linear to avoid TorchDynamo recursion
- `808970825c`: cast to compute_dtype via nn.Module.to() after dequantization (dtype mismatch fix)

## Verification
- pytest exit: PASS
- Hardware: n150
- Duration: 867.82s (0:14:27)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/flux_dev_gguf/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 12da707f8595d58b8606be22b35f020ff40c4f37 |
| tt-forge-models | 808970825ca600ccae05b15a842da047c5ed2371 |
