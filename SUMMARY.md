# Remediation Summary: audioseal/pytorch-audioseal_detector_16bits-single_device-inference

## Skill version
10

## Test
tests/runner/test_models.py::test_all_models_torch[audioseal/pytorch-audioseal_detector_16bits-single_device-inference]

## Result
FAIL — TTNN `conv_transpose2d` DRAM auto-slicer cannot fit output width 16000 in available L1

## Failure
```
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

Underlying fatal in tt-metal:
```
TT_FATAL: DRAM Auto slice could not find valid slice configuration. Tried up to 500 slices
for width-slicing on output dimension 16000. Available L1: 1396224 bytes.
Operation requires more memory than available even with maximum slicing.
(ttnn/cpp/ttnn/operations/sliding_window/op_slicing/op_slicing.cpp:266)
```

## Root cause

Two issues were found:

**Issue 1 (loader, fixed):** The `tt_forge_models` root is prepended to `sys.path` so that loader relative imports work, but this causes `tt_forge_models/audioseal/` to shadow the pip-installed `audioseal` package. `load_model()` ran `from audioseal import AudioSeal` and hit the local `__init__.py` (which has no `AudioSeal` class), raising `ImportError`.

**Issue 2 (runtime, unfixed):** `AudioSealDetector` contains `ConvTranspose1d(128, 32, kernel_size=320, stride=320)` (`detector.0.reverse_convolution`). This layer upsamples from length 50 → 16000 (stride-320 upsample of 16 kHz mono audio). The MLIR lowering maps this to `ttnn::conv_transpose2d`. The TTNN DRAM auto-slicer (`op_slicing.cpp:266`) tries up to 500 width-slices and up to `output_height` height-slices, but even the finest slice cannot fit the input halo in the 1 396 224-byte L1 budget. Both slicing directions are exhausted and the operation is fatal.

## Fix

**Issue 1 fix (committed):** In `audioseal/pytorch/loader.py`, temporarily remove the models root from `sys.path` while importing `audioseal`, then restore it. This ensures the pip-installed package is found first. Branch: `remediation/audioseal-pytorch-audioseal_detector_16bits-single_device-inference` in `tt-forge-models`.

**Issue 2 proposed fix (not implemented — compiler-stack bug):** The fix must live in `tt-metal`. Options:
1. In `ttnn/cpp/ttnn/operations/sliding_window/op_slicing/op_slicing.cpp`: investigate why even single-element slices of the 16000-wide `conv_transpose2d` exceed L1. The `ConvTranspose1d` has `kernel_size == stride` (320), meaning there is no input overlap between output windows — each slice of the output corresponds to an independent input chunk with no halo. A kernel-aware slicer could exploit this to process width-1 output slices with a proportionally smaller input slice.
2. Alternatively, the MLIR lowering in `tt-mlir` could decompose large-stride `ConvTranspose1d` (where `kernel_size == stride`) into `scatter + conv` or `pixel_shuffle + conv` representations that the existing tiled kernels can handle.

## Verification
FAIL — pytest exited 1. Hardware: n150.

## Files changed
- `audioseal/pytorch/loader.py` (in `tt-forge-models`)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d75355 |
| tt-mlir         | 553c0632b |
| tt-xla          | 81bf3cb9d |
| tt-forge-models | 87d1bef3f2 |
