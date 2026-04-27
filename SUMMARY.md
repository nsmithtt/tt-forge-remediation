# Remediation Summary: animatediff-motion-lora-pan-left/pytorch-PanLeft-single_device-inference

## Skill version
13

## Test
tests/runner/test_models.py::test_all_models_torch[animatediff_motion_lora_pan_left/pytorch-PanLeft-single_device-inference]

## Result
FAIL — TT hardware SDPA kernel requires `k_chunk_size` divisible by TILE_WIDTH (32), but the AnimateDiff temporal attention has `seq_len = num_frames = 16`, which violates this constraint.

## Failure
```
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

## Root cause
**Layer: tt-metal (runtime/compiler)**

The AnimateDiff `UNetMotionModel` has temporal motion modules that perform self-attention over the frames dimension. With the natural input of `num_frames=16`, the temporal attention's key/query sequence length is 16. The TT SDPA kernel validates that `k_chunk_size % TILE_WIDTH == 0` (TILE_WIDTH = 32). When the compiler selects `k_chunk_size = 16` (= seq_len), this assertion fires, propagating as `INTERNAL: Error code: 13` through the XLA layer.

Source: `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp` — `TT_FATAL(k_chunk_size % tt::constants::TILE_WIDTH == 0, ...)`.

The loader bug that produced the original error in CI (returning a full `AnimateDiffPipeline` instead of `pipeline.unet`, and returning a text-prompt dict instead of tensor inputs) is fixed. The residual Error code: 13 failure with the corrected loader is the SDPA k_chunk_size constraint.

## Fix
**Loader fix (tt-forge-models, `remediation/animatediff-motion-lora-pan-left`):**
- Changed `load_model` to return `self.pipeline.unet` (a `torch.nn.Module`) instead of the full pipeline object.
- Added `self.pipeline.fuse_lora()` to merge LoRA delta weights into the base linear weights and remove the PEFT adapter wrapper. Without this the wrapper's forward hook also causes Error code: 13 on TT hardware.
- Changed `load_inputs` to return proper tensor inputs `(sample, timestep, encoder_hidden_states)` matching the UNetMotionModel's forward signature, with `num_frames=16` (natural AnimateDiff frame count, matching the base `animatediff/pytorch` loader).
- Defaulted dtype to `torch.bfloat16` for TT hardware compatibility.

**Remaining compiler-stack bug (NOT fixed here):**
The TT SDPA kernel in tt-metal cannot handle sequences shorter than 32 tokens (one tile). A fix would live in `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/` to either:
1. Pad short sequences to the next multiple of TILE_WIDTH before computing attention, or
2. Fall back to a non-tiled attention kernel when `seq_len < TILE_WIDTH`.

Increasing `num_frames` to 32 would work around the hardware constraint but is explicitly forbidden by the remediation rules.

## Verification
```
FAILED tests/runner/test_models.py::test_all_models_torch[animatediff_motion_lora_pan_left/pytorch-PanLeft-single_device-inference] \
  - RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```
Duration: 528.95 s (0:08:48) — compilation succeeded; runtime execution failed on TT hardware.
Hardware: n150.

## Files changed
- `animatediff_motion_lora_pan_left/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | dd6fe408573f970f976b1f287d5a33d928af927b |
| tt-forge-models | 54853b45f8b6f0e4c7dff48a5b35637ea21cadc5 |
