# AnimateDiff PyTorch Remediation Summary

## Test
`tests/runner/test_models.py::test_all_models_torch[animatediff/pytorch-v1.5.3-single_device-inference]`

## Failure
`RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13`

The error occurred in `torch_xla._XLAC._run_cached_graph` during TT-MLIR/hardware execution.

## Root Cause

In `animatediff/pytorch/loader.py`, the `load_inputs` method constructed the UNet sample tensor with erroneous spatial dimensions:

```python
height = 64
width = 64
sample = torch.randn(
    (batch_size, in_channels, num_frames, height // 8, width // 8),
    dtype=dtype,
)
```

`height=64` and `width=64` are already latent-space coordinates (512px image / 8 VAE scale factor = 64). Dividing by 8 again produced 8×8 latent inputs, which are far too small for the UNetMotionModel. The successive downsampling blocks reduce spatial dimensions to 1×1, causing TT hardware-level failures (INTERNAL error, code 13).

## Fix

**Repository:** `tt-forge-models` (at `animatediff/pytorch/loader.py`)  
**Branch:** `arch-c-36-tt-xla-dev/nsmith/hf-bringup-28`  
**Commit:** `f2672ec59970be30ed088f8f3da1f3f94522cb91`

Changed `height // 8` and `width // 8` to `latent_height` and `latent_width` (both 64), so the sample shape is `(1, 4, 16, 64, 64)` instead of `(1, 4, 16, 8, 8)`.

Note: The `encoder_hidden_states` batch-size fix (`batch_size * num_frames`) was already present on this branch from a prior commit.

## Result
Test passes: `1 passed in 816.54s (0:13:36)`

## Submodule Hashes
- tt-xla: `37f860df9c2acbabfb26bdbfbe11fb3f4e4d9502`
- tt-mlir: `553c0632b353f8ac457aba0d01a460a5e0f5b5ee`
- tt-metal: `3fa4d753550dba1d4aacc9af45b111ae540f63fc`
- tt-forge-models: `f2672ec59970be30ed088f8f3da1f3f94522cb91`
