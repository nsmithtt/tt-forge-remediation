# Remediation Summary: cbramod_pretrained-pytorch-Pretrained-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[cbramod_pretrained/pytorch-Pretrained-single_device-inference]

## Result
SILICON_PASS — all three layers (loader, tt-xla, tt-mlir) required fixes; test passes on silicon

## Stack layer
loader, tt-xla, tt-mlir

## Tier
A

## Bug fingerprint
stablehlo-abs-complex-no-lowering

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
RuntimeError: Invalid type for NanValue (C64).
ValueError: Error code: 13
```

## Root cause

Three independent issues at three different layers:

**Layer 1 — loader (braindecode missing, float32 constraint)**
The `braindecode` package was not listed in `requirements.txt`, causing
`ModuleNotFoundError`. Additionally, CBraMod uses `torch.fft.rfft`
internally which does not support bfloat16; the loader had to keep
the model in float32 regardless of `dtype_override`.

**Layer 2 — tt-xla eager partitioner (`abs(complex)` wrong dtype)**
`UnsupportedNodesCollector` in `dynamo_bridge.py` runs the FX graph
eagerly with real XLA tensors before MLIR compilation. XLA's
implementation of `abs(complex_tensor)` returns a Python tensor
still typed as `complex64` (with imaginary part = 0) instead of
`float32`. This complex result flowed into `var_mean`, which raised
`RuntimeError: Invalid type for NanValue (C64)`.

A naïve fix of returning `result.real` recorded `Real(HLO-float)` in
XLA's lazy graph, which became `stablehlo.real(tensor<f32>)` — an
invalid StableHLO op because `real` requires complex input. The correct
fix decomposes `abs(complex_inp)` as `sqrt(real(inp)² + imag(inp)²)`
using the *input* (which is complex), recording valid `Real(complex)`
and `Imag(complex)` HLO ops.

**Layer 3 — tt-mlir `ComplexDataTypeConversion` pass (missing `abs` pattern)**
The `ComplexDataTypeConversion` pass converts `complex<f32>` function
arguments to packed `tensor<...x2xf32>` representations. `AbsOp` had no
conversion pattern, so it was left legal while its input type had already
been converted; this produced an "unresolved materialization" error at
`applyPartialConversion` time (surfaced as `Error code: 13`).

## Fix

**Loader — `tt_forge_models/cbramod_pretrained/pytorch/`**
- `requirements.txt`: added `braindecode` and `torchaudio==2.9.1+cpu`
  (CPU wheel pin avoids CUDA pull-in from braindecode).
- `loader.py`: removed `dtype_override` application so the model stays
  in native `float32`; `load_inputs` returns `torch.float32` tensors.

**tt-xla — `python_package/tt_torch/torch_overrides.py`**
Added a check in `TorchFunctionMode.__torch_function__` that intercepts
`abs` on a complex tensor and decomposes it as
`torch.sqrt(real(inp)**2 + imag(inp)**2)`. The guard
`not torch.compiler.is_compiling()` ensures the interception only fires
during eager evaluation (UnsupportedNodesCollector, extract_internal)
and not during dynamo tracing, where the graph structure is captured.

**tt-mlir — `lib/Dialect/StableHLO/Transforms/ComplexDataTypeConversion.cpp`**
Added `StablehloAbsComplexConversionPattern` that lowers
`abs(complex<f32>)` → `sqrt(re² + im²)` by:
1. Transposing the last (packed) dimension to the front.
2. Slicing out real (`offset=0`) and imaginary (`offset=1`) components.
3. Reshaping to drop the leading size-1 slice dimension.
4. Computing `sqrt(re*re + im*im)`.
Added a dynamic illegality rule for `AbsOp` when its operand is complex,
so the pattern is required rather than optional.

## Verification
- pytest exit: PASS
- Hardware: wormhole (n150)
- Duration: 84.04s (0:01:24)
- Tier A attempts: 1

## Files changed
- `tt_forge_models/cbramod_pretrained/pytorch/requirements.txt` (loader)
- `tt_forge_models/cbramod_pretrained/pytorch/loader.py` (loader)
- `tt-xla/python_package/tt_torch/torch_overrides.py` (tt-xla)
- `tt-mlir/lib/Dialect/StableHLO/Transforms/ComplexDataTypeConversion.cpp` (tt-mlir)
- `tt-xla/pytest.ini` (test infrastructure, cherry-picked from another remediation)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 29db9678bb022c3159c1e50777eb5c18e403d2be |
| tt-xla          | acbd02f5d7ec18468c412db250248ef3a243fd1c |
| tt-forge-models | ef8994710cba55fb742cb4b175279dccabef8a91 |
