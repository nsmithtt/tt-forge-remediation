# Remediation Summary: fnet-masked_lm-pytorch-Base-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[fnet/masked_lm/pytorch-Base-single_device-inference]

## Result
FAIL — stablehlo.fft has no lowering in tt-mlir; complex tensor materialization fails at compilation

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
stablehlo-fft-no-lowering

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
ValueError: Error code: 13

While executing %_fft_c2c : [num_users=1] = call_function[target=torch.ops.aten._fft_c2c.default](args = (%_to_copy, [1, 2], 0, True), kwargs = {})
Original traceback:
  File ".../transformers/models/fnet/modeling_fnet.py", line 176, in forward
    outputs = self.fourier_transform(hidden_states).real
```

The device log shows:
```
error: failed to legalize unresolved materialization from ('tensor<2x512x768xcomplex<f32>>') to ('tensor<2x512x768x2xf32>') that remained live after conversion
Failed to convert from SHLO to TTIR module
```

## Root cause
FNet's core attention mechanism uses Fourier transforms (`torch.fft.fft2`) instead of self-attention. This decomposes in the FX graph to `_to_copy` (real → complex64) followed by `aten._fft_c2c` (complex FFT). In the StableHLO pipeline, these become `stablehlo.convert` and `stablehlo.fft` respectively.

The `StableHLOComplexDataTypeConversionPass` in `tt-mlir/lib/Dialect/StableHLO/Transforms/ComplexDataTypeConversion.cpp` converts complex tensors to float-pair representation (`tensor<...xcomplex<f32>>` → `tensor<...x2xf32>`). It handles `complex`, `real`, `imag`, `constant`, `reshape`, `broadcast_in_dim`, `slice`, and `concatenate` ops, but has **no pattern for `stablehlo.fft`**. When the type converter maps the complex result type of `stablehlo.fft` to a float-pair type, the dialect conversion framework requires a materialization function or a conversion pattern for the op — neither exists. The MLIR dialect conversion fails with "unresolved materialization", the SHLO→TTIR compilation fails, and the PJRT plugin returns Error code 13 (INTERNAL).

The `UnsupportedNodesCollector.run_node()` in the XLA dynamo bridge does not catch this exception, so the error propagates out of `partition_fx_graph_for_cpu_fallback` and crashes the whole compilation rather than routing the op to CPU fallback.

## Fix
The proposed fix requires two steps:

1. **Lower `stablehlo.fft` to TTIR in tt-mlir** (`lib/Dialect/StableHLO/Transforms/`): Either:
   - Expand `stablehlo.fft` into a DFT via twiddle-factor matrix multiplication using existing TTIR matmul/elementwise ops. This is numerically equivalent but expensive.
   - Add a dedicated `ttir.fft` op and a corresponding TTNN kernel in tt-metal.

2. **Implement FFT arithmetic in tt-metal** if the TTIR FFT op route is taken.

This is where the fix would live:
- `tt-mlir/lib/Dialect/StableHLO/Transforms/ComplexDataTypeConversion.cpp` — add `stablehlo.fft` conversion pattern
- `tt-mlir/lib/Dialect/StableHLO/Transforms/` — possibly a new pass for FFT decomposition
- `tt-mlir/include/ttmlir/Dialect/TTIR/IR/TTIROps.td` — new FFT op definition (if dedicated op route)
- `tt-metal/` — new TTNN FFT kernel (if dedicated op route)

## Tier B justification
new-infrastructure: `stablehlo.fft` requires implementing FFT arithmetic that does not exist anywhere in tt-mlir or tt-metal. Either a DFT expansion (requires complex arithmetic lowering across multiple new patterns) or a dedicated FFT TTIR op + TTNN kernel (requires new op definitions, kernel implementation, and registration). This touches more than 3 files across multiple repos.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    33.63s
- Tier A attempts: N/A

## Files changed
None (no fix attempted — Tier B)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 99b2a770d016603614737ca4421190e4ffaf87bf |
