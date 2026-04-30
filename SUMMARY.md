# Remediation Summary: flux_dev_gguf-pytorch-eviation_caesar_Q8_0-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[flux_dev_gguf/pytorch-eviation_caesar_Q8_0-single_device-inference]

## Result
NO_FIX_NEEDED — test passes on the configured branch; failure was already resolved by prior Q5_K_S loader fixes

## Stack layer
n/a

## Tier
N/A

## Bug fingerprint
n/a

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
The RecursionError was caused by `GGUFParameter.__torch_function__` calling
`super().__torch_function__()` infinitely under TorchDynamo tracing. The
prior remediation for `flux_dev_gguf/pytorch-eviation_caesar_Q5_K_S`
(commit `d6d72b9693` in tt_forge_models) applied four loader fixes that also
cover the Q8_0 variant:

1. **blob/main URL**: diffusers 0.37.1 `_extract_repo_id_and_weights_name` strips
   only `blob/main/` from HF URLs; `resolve/main/` produced a malformed double-path
   filename.
2. **Explicit config repo**: `from_single_file` infers config from GGUF metadata
   pointing to the gated `black-forest-labs/FLUX.1-dev` repo; passing
   `BBuf/flux1-dev-modelopt-nvfp4-sglang-transformer` bypasses the gate.
3. **`_dequantize_gguf_and_restore_linear`**: replaces all `GGUFLinear` modules
   with plain `nn.Linear` before dynamo tracing, eliminating the infinite
   `__torch_function__` dispatch.
4. **`torch.nn.Module.to()`**: casts all parameters to compute_dtype using the
   base `nn.Module.to()` rather than `ModelMixin.to()`, which raises on quantized
   models even after dequantization.

Because the Q8_0 GGUF for FLUX.1-dev stores non-linear parameters (norm scales)
as plain F32 in the GGUF file, they are loaded as regular `torch.Tensor`s — not
`GGUFParameter`. Therefore the three GGUFLinear-only fix is sufficient and no
`_dequantize_bf16_params` step is needed.

## Fix
No fix required. The configured branch (`arch-c-36-tt-xla-dev/nsmith/2026-04-22_16-58/hf-bringup-33`,
tt-xla commit `aecc3c84820cbe13c6069b1b8500449e1125e0f2`) already contains the
loader fixes in tt_forge_models commit `d6d72b9693`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    645.17s (0:10:45)
- Tier A attempts: N/A

## Files changed
None (no fix required)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | aecc3c84820cbe13c6069b1b8500449e1125e0f2 |
| tt-forge-models | d6d72b9693e467b74726f572c4190033ffc988d9 |
