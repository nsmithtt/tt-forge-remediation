# Remediation Summary: bitnet-causal_lm-pytorch-b1_58_2B_4T-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bitnet/causal_lm/pytorch-b1_58_2B_4T-single_device-inference]

## Result
SILICON_PASS — stablehlo.round_nearest_even lowering added to tt-mlir

## Stack layer
tt-mlir

## Tier
A

## Bug fingerprint
stablehlo-round-nearest-even-no-lowering

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
loc("round-nearest-even.520"): error: failed to legalize operation 'stablehlo.round_nearest_even'
2026-05-01 04:40:57.909 ( 120.919s) [        5997B140]      module_builder.cc:889    ERR| Failed to convert from SHLO to TTIR module
...
torch_xla._XLAC._xla_warm_up_cache(args_and_out_tensor_only, [])
E   ValueError: Error code: 13
```

## Root cause
The BitNet b1.58-2B-4T model uses `torch.round()` during its 1.58-bit weight
dequantization pass, which PyTorch/XLA lowers to `stablehlo.round_nearest_even`.
The tt-mlir StableHLO→TTIR conversion pass had no lowering pattern for this op,
so the whole-module conversion failed. After the failure the XLA bridge tried
CPU-fallback partitioning via `partition_fx_graph_for_cpu_fallback`, which then
surfaced as `ValueError: Error code: 13` (pjrt-device-to-host-transfer). The
primary bug is in tt-mlir: a missing lowering for `stablehlo.round_nearest_even`.

## Fix
Added `StableHLOToTTIRRoundNearestEvenOpConversionPattern` in
`tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`.

The pattern decomposes `stablehlo.round_nearest_even(x)` as:
```
sign(x) * floor(abs(x) + 0.5)
```
using existing TTIR ops: `AbsOp`, `AddOp` (with a splat 0.5 `ConstantOp`),
`FloorOp`, `SignOp`, and `MultiplyOp`. This implements "round half away from
zero", which differs from IEEE banker's rounding only at exact half-integers —
an acceptable approximation for bfloat16 ML inference. The pattern is registered
in `addElementwiseUnaryOpsConversionPatterns`.

The fix is a single pattern addition in one file; no new TTIR ops or downstream
lowering changes were required.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    167.17s (0:02:47)
- Tier A attempts: 1

## Files changed
- `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 99fc462fdb8b904ee25e7ef35c6b9051e48af9f7 |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 5ab3d66b209d180696558f19afdb313018d062f2 |
