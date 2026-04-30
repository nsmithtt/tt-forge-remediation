# Remediation Summary: flux_dev_gguf-pytorch-eviation_caesar_Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[flux_dev_gguf/pytorch-eviation_caesar_Q4_K_M-single_device-inference]

## Result
SILICON_PASS — four loader bugs fixed; test passes on TT silicon in 0:11:48

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
The `flux_dev_gguf` loader had four bugs, each blocking progress to the next:

1. **URL double-path bug**: `from_single_file` was called with a `resolve/main/` URL. diffusers 0.37.1 `_extract_repo_id_and_weights_name` strips `blob/main/` but not `resolve/main/`, so the weights_name retained `resolve/main/experimental-from-f16-caesar/flux1-dev-Q4_K_M.gguf` and HF hub prepended another `resolve/main/`, yielding a 404.

2. **Gated config repo**: The GGUF metadata embeds `config_source = black-forest-labs/FLUX.1-dev` (a gated repo). Without a valid token, diffusers raises `GatedRepoError`. Fix: pass `config=BBuf/flux1-dev-modelopt-nvfp4-sglang-transformer` (a public alternative with identical architecture config).

3. **GGUFParameter TorchDynamo recursion**: After loading, all linear weights are `GGUFParameter` tensor subclass instances. `GGUFParameter.__torch_function__` calls `super().__torch_function__()`, which under TorchDynamo tracing recurses infinitely → `RecursionError`. Fix: eagerly dequantize via `_dequantize_gguf_and_restore_linear(transformer)` before compilation.

4. **dtype mismatch after dequantization**: F16-stored GGUF tensors dequantize to float16, not bfloat16. `diffusers.ModelMixin.to()` refuses to cast quantized models (checks `is_quantized`). Fix: call `torch.nn.Module.to(transformer, compute_dtype)` directly, bypassing ModelMixin's guard.

## Fix
All four fixes are in `flux_dev_gguf/pytorch/loader.py` in tt-forge-models, on branch `remediation/flux_dev_gguf-pytorch-eviation_caesar_Q4_K_M-single_device-inference` (cherry-picked from the already-reviewed Q5_K_M branch):

1. `fix: use blob/main URL for diffusers 0.37.1 GGUF from_single_file compatibility` (c72e66d705) — change `resolve/main` → `blob/main` in URL
2. `fix: pass explicit config repo to bypass gated black-forest-labs/FLUX.1-dev` (2a501d7115) — add `_FLUX_DEV_CONFIG_REPO = "BBuf/flux1-dev-modelopt-nvfp4-sglang-transformer"` and pass as `config=`
3. `fix: dequantize GGUF tensors before compilation to avoid TorchDynamo recursion` (f52360efb7) — call `_dequantize_gguf_and_restore_linear(self.transformer)` after `from_single_file`
4. `fix: cast to compute_dtype after GGUF dequantization to resolve dtype mismatch` (51fabd392c) — call `torch.nn.Module.to(self.transformer, compute_dtype)` after dequantization

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    708.30s (0:11:48)
- Tier A attempts: N/A

## Files changed
- `flux_dev_gguf/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 51fabd392c1a2f5f088c2c34a1c66ef3b0095a1d |
