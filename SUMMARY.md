# Remediation Summary: moody_real_mix_v3_gguf-pytorch-moodyRealMix_zitV3_q4_k_m-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[moody_real_mix_v3_gguf/pytorch-moodyRealMix_zitV3_q4_k_m-single_device-inference]

## Result
FAIL — Tier A fix resolved complex-gather Error 13; second Error 13 from aten.view.dtype (uint8→float16 bitcast) in GGUF Q4_K dequantization is Tier B new-infrastructure

## Stack layer
loader, tt-mlir

## Tier
A

## Bug fingerprint
gguf-q4k-uint8-float16-bitcast-cross-size

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

While executing %view_204 : [num_users=1] = call_function[target=torch.ops.aten.view.dtype](args = (%slice_10, torch.float16), kwargs = {})
Original traceback:
  diffusers/quantizers/gguf/utils.py:361, in dequantize_blocks_Q4_K
      dmin = dmin.view(torch.float16).to(dtype)

(Earlier, before the Tier A fix was applied, the first Error 13 came from stablehlo.gather on complex<f64>
 tensors in Lumina2 _get_freqs_cis. That was resolved by the Tier A fix in tt-mlir.)

## Root cause

**First Error 13 (fixed, Tier A — tt-mlir):** Lumina2 RoPE (`_precompute_freqs_cis`) uses
`torch.polar(float64, float64)` → complex128 tensors, then `_get_freqs_cis` does `torch.gather`
on those complex<f64> tensors. `StableHLOComplexDataTypeConversionPass` had no pattern for
`stablehlo::GatherOp`, so complex-typed gathers survived into later passes that cannot handle
them, causing an INTERNAL:13 compilation failure.

**Second Error 13 (unfixed, Tier B — tt-mlir/ttnn):** GGUF Q4_K dequantization in
`dequantize_blocks_Q4_K` calls `dmin.view(torch.float16)` — a uint8→float16 bitcast (different
element sizes). This lowers to `stablehlo.bitcast_convert` in tt-mlir, which then maps to
`ttnn::bitcast`. TTNN's bitcast implementation only supports same-size element type pairs;
cross-size pairs (1-byte uint8 → 2-byte float16) have no kernel and return INTERNAL:13.

**Loader fixes (loader — tt_forge_models):** The original loader had two bugs blocking model load:
(1) `resolve/main/` URL passed to diffusers `from_single_file`, which double-appended the path;
(2) wrong 5B Lumina2 architecture parameters (CAP_FEAT_DIM=2304 instead of 2560, adaLN
bottleneck dim wrong) requiring 4 monkey-patches into diffusers to match the GGUF checkpoint.

## Fix

**Loader fix** (tt_forge_models branch `remediation/moody_real_mix_v3_gguf_pytorch_q4_k_m_single_device_inference`):
- `moody_real_mix_v3_gguf/pytorch/loader.py`: changed URL from `resolve/main/` to `blob/main/`;
  set CAP_FEAT_DIM=2560, _HIDDEN_SIZE=3840; added `_patch_lumina2_5b_gguf()` with 4 patches
  (MHA QKV split key fix, LuminaRMSNormZero cond_dim=256, TimestepEmbedding out_dim=256,
  LuminaLayerNormContinuous cond_dim=256); fixed GGUFParameter.as_tensor() subclass escape.

**Tier A fix** (tt-mlir branch `remediation/moody-real-mix-v3-gguf-complex-gather`):
- `lib/Dialect/StableHLO/Transforms/ComplexDataTypeConversion.cpp`: added
  `ComplexGatherOpConversionPattern` which converts `stablehlo::GatherOp` with complex-typed
  operand/result to operate on the real-pairs representation: appends 2 to `slice_sizes`
  (new trailing real/imag dimension) and appends `old_result_rank` to `offset_dims` (new
  trailing dim in result). Registered the pattern and added GatherOp to the dynamically
  illegal set in `StableHLOComplexDataTypeConversionPass`.

**Second bug (proposed fix location):** `ttnn/cpp/ttnn/operations/data_movement/` — implement
cross-size bitcast decomposition for `stablehlo.bitcast_convert`, or intercept
`aten.view.dtype` in tt-xla's `torch_overrides.py` and emit a reshape+pack sequence.

## Tier B justification

new-infrastructure: `ttnn::bitcast` only supports same-size element type pairs. Supporting
uint8→float16 (and similar cross-size pairs) requires either new ttnn kernel infrastructure or a
non-trivial decomposition pass (pack pairs of uint8 values into a float16 buffer). No existing
lowering handles this case.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    756.96s (0:12:36)
- Tier A attempts: 1

## Files changed
- tt-mlir: `lib/Dialect/StableHLO/Transforms/ComplexDataTypeConversion.cpp`
- tt_forge_models: `moody_real_mix_v3_gguf/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 9901eca2a749bf7dc01a022bc48f0672e65868c3 |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 129a1a105fb34f81d45c974b9e41d135e8777549 |
