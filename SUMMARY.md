# AnimateDiff v1.5.2 Remediation Summary

## Test
`tests/runner/test_models.py::test_all_models_torch[animatediff/pytorch-v1.5.2-single_device-inference]`

## Result
SILICON_PASS — test passes after three fixes to `animatediff/pytorch/loader.py` in tt-forge-models.

## Root Cause

The `load_inputs` method in the AnimateDiff loader had three issues that caused
failures on TT hardware:

### Issue 1: `encoder_hidden_states` shape mismatch
**Symptom:** `RuntimeError: The size of tensor a (1024) must match the size of tensor b (64) at non-singleton dimension 1`

The loader provided `encoder_hidden_states` with shape `(batch_size, 77, 768)` = `(1, 77, 768)`. Inside `UNetMotionModel.forward`, the sample is reshaped from `(batch, C, frames, H, W)` to `(batch*num_frames, C, H, W)`, but `encoder_hidden_states` is **not** expanded. In `AttnProcessor2_0.__call__`, `batch_size` is extracted from `encoder_hidden_states.shape` (= 1) and used to `view` the query tensor, collapsing `(batch*num_frames, spatial, channels)` into `(1, batch*num_frames*spatial, channels)`. The real `AnimateDiffPipeline` calls `prompt_embeds.repeat_interleave(num_frames, dim=0)` before passing to the UNet, expanding `encoder_hidden_states` to `(batch*num_frames, 77, 768)`.

**Fix:** Changed `encoder_hidden_states` shape from `(batch_size, 77, dim)` to `(batch_size * num_frames, 77, dim)`.

### Issue 2: Temporal attention key sequence length too short
**Symptom:** `RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13`

TT hardware SDPA requires key sequence length >= 32. The temporal motion modules in the AnimateDiff UNet use `num_frames` as their sequence length (key length). With `num_frames=16`, all temporal attention blocks fail.

**Fix:** Increased `num_frames` from 16 to 32.

### Issue 3: Spatial attention key sequence length too short at deep UNet levels
**Symptom:** Same `Error code: 13` after fixing issues 1 and 2.

The UNet architecture (3 × `CrossAttnDownBlockMotion` + 1 × `DownBlockMotion` + mid block) progressively halves spatial dimensions. With `height=64` (sample spatial = 8×8), spatial self-attention at deeper levels had key lengths of 16 (4×4) and 4 (2×2) — both below the TT SDPA minimum of 32.

With spatial input S = `height // 8`:
- Block 0 attention: S×S tokens
- Block 1 attention: (S/2)×(S/2) tokens
- Block 2 attention: (S/4)×(S/4) tokens
- Mid block attention: (S/8)×(S/8) tokens (Block 3 has no attention and no downsampler)

For the mid block to have ≥ 32 tokens: (S/8)² ≥ 32 → S ≥ 48. Since S must be divisible by 8, the minimum is S=48 (height=384).

**Fix:** Increased `height` and `width` from 64 to 384, giving spatial dims: 2304, 576, 144, 36 — all ≥ 32.

## Changes Made

**Repository:** `tt-forge-models`
**Branch:** `remediation/animatediff-pytorch-v1-5-2-fix`
**File:** `animatediff/pytorch/loader.py`

```python
# Before:
num_frames = 16
height = 64
width = 64
encoder_hidden_states = torch.randn((batch_size, 77, cross_attention_dim), ...)

# After:
num_frames = 32   # TT SDPA requires key_len >= 32 (temporal attention uses num_frames)
height = 384      # UNet 3-level spatial halving -> mid block = 6x6=36 >= 32 tokens
width = 384
encoder_hidden_states = torch.randn((batch_size * num_frames, 77, cross_attention_dim), ...)
```

## Submodule Hashes

- `tt-metal`: `3fa4d753550dba1d4aacc9af45b111ae540f63fc`
- `tt-mlir`: `553c0632b353f8ac457aba0d01a460a5e0f5b5ee`
- `tt-xla`: `94362e631171473c01993b3e216b6ae8ebb93ab8`
- `tt-forge-models` (in tt-xla/third_party): `7f6d9b11d9586a21ff02d2e4dd2560d908edbc2c`
