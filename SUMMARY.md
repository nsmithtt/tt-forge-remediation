# Remediation Summary: biot-pytorch-shhs_prest_18chs-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[biot/pytorch-shhs_prest_18chs-single_device-inference]

## Result
FAIL — aten.stft.default returns XLAComplexFloatType which the TT XLA backend cannot handle

## Stack layer
loader, tt-xla

## Tier
B

## Bug fingerprint
aten-stft-complex-output-not-supported-xla

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
RuntimeError: Check failed: xtensor: Input tensor is not an XLA tensor: XLAComplexFloatType

While executing %stft : [num_users=1] = call_function[target=torch.ops.aten.stft.default](args = (%squeeze, 200, 100, None, None, False, True, True), kwargs = {})

## Root cause
Two loader bugs were found and fixed before reaching the original compiler-stack failure:

1. **braindecode pulls in CUDA torchaudio**: `braindecode` depends on `torchaudio`, and pip resolves this to the CUDA-linked PyPI wheel (`torchaudio 2.11.0`) which requires `libcudart.so.13`. This is absent on a CPU-only system. Fixed by pinning `torchaudio==2.9.1+cpu` from the PyTorch CPU wheel index in the BIOT `requirements.txt`.

2. **BFloat16 incompatible with MKL FFT**: The loader had `dtype_override` in both `load_model` and `load_inputs`, causing the test infra to request BFloat16. BIOT's encoder calls `torch.stft` internally; MKL FFT does not support BFloat16 tensors on CPU, so the CPU reference run failed. Fixed by removing `dtype_override` from the loader so the model runs in native float32.

After both loader fixes, the test reproduces the original failure in the TT device run:

**Root cause of the remaining failure (tt-xla layer)**: `aten.stft.default` is in the compiled FX graph. The 'tt' backend (via `bridge.extract_compiled_graph`) attempts to trace and compile this op through the XLA pipeline. `torch.stft` with `return_complex=True` produces a `torch.complex64` tensor. The TT XLA backend has no support for complex tensor types in compilation — there is no lowering for `stablehlo.fft` in tt-mlir, and the XLA runtime rejects the complex tensor with `Check failed: xtensor: Input tensor is not an XLA tensor: XLAComplexFloatType`.

## Fix
**Loader fixes (committed in tt_forge_models remediation branch)**:

- `biot/pytorch/requirements.txt`: Added `--extra-index-url https://download.pytorch.org/whl/cpu` and `torchaudio==2.9.1+cpu` so that braindecode's torchaudio dependency resolves to the CPU wheel instead of the CUDA-linked PyPI wheel.

- `biot/pytorch/loader.py`: Removed `dtype_override` parameter from `load_model` and `load_inputs`. The BIOT model's internal `torch.stft` does not support BFloat16 (MKL FFT restriction), so the model must run in float32. Removing the parameter causes the test infra to skip the BFloat16 path.

**Proposed fix for the compiler-stack bug (not implemented — Tier B)**:

The fix would require one of:
a. Full FFT op support: implement a `stablehlo.fft` → TTNN FFT kernel lowering in tt-mlir, plus complex tensor buffer management throughout the compilation and runtime pipeline.
b. Graph partitioning: add `aten.stft.default` (and other complex-output ops) to a CPU-fallback set so they are partitioned out of the XLA graph and run on host, with the real-valued output re-entered into the XLA graph.

Both approaches require new infrastructure and touch multiple files across tt-mlir and/or tt-xla.

## Tier B justification
<which Tier B indicator applies>
new-infrastructure

`torch.stft` returns `torch.complex64` tensors. The TT backend has no lowering for `stablehlo.fft` and no complex tensor type representation in the compilation or runtime paths. Supporting this would require either new FFT kernel lowering infrastructure in tt-mlir or a cross-cutting graph-partitioning mechanism to dispatch unsupported ops to CPU.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    15.89s
- Tier A attempts: N/A

## Files changed
- `biot/pytorch/requirements.txt` (tt_forge_models)
- `biot/pytorch/loader.py` (tt_forge_models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | ce8c5e3d22e450d4df7319f5684216e6367ad6ef |
| tt-forge-models | 8891068ae06a14c9e1d8c99839b7c237fba48d69 |
